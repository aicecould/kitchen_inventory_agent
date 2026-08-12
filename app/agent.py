"""DeepSeek-powered LangChain tool-calling Agent."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter
from typing import Any, Literal
from zoneinfo import ZoneInfo

from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from app.actions import InventoryActionService
from app.audit import (
    AuditCode,
    AuditResult,
    append_confidence_notice,
    canonical_tool_response,
    regex_audit,
    review_candidate,
    review_tool_calls,
)
from app.config import Settings
from app.context import AgentContext, AgentResult
from app.memory import ConversationHistoryRepository
from app.profile import ProfileCategory, ProfileProposal
from app.profile_actions import (
    ProfileActionService,
    ProfileChangeOperation,
    ProfileChangeRequest,
)
from app.prompts import (
    ANSWER_REWRITE_SYSTEM_PROMPT,
    RECIPE_RECOVERY_SYSTEM_PROMPT,
    REPAIR_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
)
from app.supervisor import (
    RepairDecision,
    ResponseSupervisor,
    RetryMode,
    SupervisorCandidate,
    SupervisorRequest,
    fixed_response,
)
from app.structured_output import invoke_validated_json
from app.tools.inventory import InventoryRepository
from app.tools.recipe import RecipeRouter
from app.trace import TraceRecorder, elapsed_ms
from recommender.ingredients import normalize_search_request
from recommender.models import IngredientTranslation, RecipeRanking


class HistoryTimeRangeInput(BaseModel):
    start_at: datetime
    end_at: datetime
    limit: int = Field(default=20, ge=1, le=100)

    @model_validator(mode="after")
    def validate_range(self) -> "HistoryTimeRangeInput":
        if self.start_at.tzinfo is None or self.start_at.utcoffset() is None:
            raise ValueError("start_at must include a timezone")
        if self.end_at.tzinfo is None or self.end_at.utcoffset() is None:
            raise ValueError("end_at must include a timezone")
        if self.start_at >= self.end_at:
            raise ValueError("start_at must be earlier than end_at")
        return self


_ENGLISH_API_TERM = re.compile(r"^[A-Za-z][A-Za-z0-9 '&()/.+-]*$")
_ALLERGEN_TOOL_ERRORS = frozenset(
    {
        "MISSING_INGREDIENT_MAPPING",
        "UNEXPECTED_INGREDIENT_MAPPING",
        "DUPLICATE_INGREDIENT_MAPPING",
        "INVALID_ENGLISH_API_TERM",
    }
)


@dataclass(slots=True)
class KitchenAgent:
    model: ChatOpenAI
    inventory: InventoryRepository
    recipes: RecipeRouter
    actions: InventoryActionService
    history: ConversationHistoryRepository | None = None
    profile_actions: ProfileActionService | None = None
    application_timezone: str = "Asia/Shanghai"
    recursion_limit: int = 10
    max_input_chars: int = 60000
    model_callbacks: list[Any] = field(default_factory=list)
    repair_decider: (
        Callable[[SupervisorRequest, AuditResult], RepairDecision] | None
    ) = None

    def run(self, context: AgentContext, target_language: str = "zh") -> AgentResult:
        trace = TraceRecorder()
        user_message = self._user_message(
            context,
            target_language,
            application_timezone=self.application_timezone,
        )
        if len(user_message) > self.max_input_chars:
            trace.record("agent", "input_limit", "blocked", "input too long")
            raise ValueError(
                f"Agent input is too long: {len(user_message)} characters; "
                f"limit is {self.max_input_chars}"
            )
        trace.record("agent", "input_limit", "passed", f"{len(user_message)} character(s)")
        tools = self._create_tools(context, trace)
        agent = create_agent(
            model=self.model,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
        )
        started = perf_counter()
        try:
            invoke_config: dict[str, object] = {
                "recursion_limit": self.recursion_limit
            }
            if self.model_callbacks:
                invoke_config["callbacks"] = self.model_callbacks
            response = agent.invoke(
                {"messages": [{"role": "user", "content": user_message}]},
                config=invoke_config,
            )
        except Exception as exc:
            trace.record(
                "model",
                "deepseek_agent",
                "failed",
                type(exc).__name__,
                elapsed_ms(started),
            )
            trace.record(
                "supervisor",
                "final_fallback",
                "completed",
                AuditCode.TASK_IMPOSSIBLE.value,
            )
            return AgentResult(
                content=fixed_response(AuditCode.TASK_IMPOSSIBLE, target_language),
                audit_reason=AuditCode.TASK_IMPOSSIBLE.value,
                execution_trace=trace.events,
            )
        trace.record(
            "model",
            "deepseek_agent",
            "success",
            "response completed",
            elapsed_ms(started),
        )
        messages = response.get("messages", [])
        content = self._last_text(messages)
        history = self._tool_history(messages)
        canonical = canonical_tool_response(history, target_language=target_language)
        if canonical is not None:
            content = canonical
        elif target_language == "zh" and any(
            code in content for code in _ALLERGEN_TOOL_ERRORS
        ):
            content = "菜谱查询缺少或包含无效的过敏原过滤条件，本次没有发送查询请求。"
        content = append_confidence_notice(content, context.ingredients)
        audit = regex_audit(content)
        trace.record(
            "output",
            "regex_audit",
            "passed" if audit.passed else "blocked",
            audit.reason or "passed",
        )
        tool_audit = review_tool_calls(
            content,
            history,
            target_language=target_language,
            ingredients=context.ingredients,
            vision_confirmation_result=context.vision_confirmation_result,
        )
        trace.record(
            "output",
            "tool_call_audit",
            "passed" if tool_audit.passed else "blocked",
            tool_audit.reason or "passed",
        )
        combined_audit = review_candidate(
            content,
            history,
            intent=context.intent,
            target_language=target_language,
            ingredients=context.ingredients,
            vision_confirmation_result=context.vision_confirmation_result,
        )
        if not audit.passed or not tool_audit.passed or not combined_audit.passed:
            supervisor = ResponseSupervisor(
                audit=lambda candidate, candidate_history: review_candidate(
                    candidate,
                    candidate_history,
                    intent=context.intent,
                    target_language=target_language,
                    ingredients=context.ingredients,
                    vision_confirmation_result=context.vision_confirmation_result,
                ),
                repair=self.repair_decider or self._run_repair_subagent,
                main_retry=lambda mode, feedback, immutable_history: (
                    self._run_main_retry(
                        context=context,
                        target_language=target_language,
                        mode=mode,
                        feedback=feedback,
                        immutable_history=immutable_history,
                        trace=trace,
                    )
                ),
            )
            outcome = supervisor.review(
                SupervisorRequest(
                    content=content,
                    tool_history=history,
                    intent=context.intent,
                    target_language=target_language,
                )
            )
            for name, detail in outcome.steps:
                trace.record(
                    "supervisor",
                    name,
                    "blocked" if name == "final_fallback" else "completed",
                    detail,
                )
            return AgentResult(
                content=outcome.content,
                blocked=outcome.blocked,
                audit_reason=outcome.audit_reason,
                tool_history=outcome.tool_history,
                execution_trace=trace.events,
            )
        return AgentResult(
            content=content,
            tool_history=history,
            execution_trace=trace.events,
            profile_proposals=_profile_proposals(history),
        )

    def _run_repair_subagent(
        self, request: SupervisorRequest, audit: AuditResult
    ) -> RepairDecision:
        payload = {
            "request": request.model_dump(mode="json"),
            "audit": {
                "code": audit.code.value,
                "reason": audit.reason,
            },
        }
        return invoke_validated_json(
            self.model,
            RepairDecision,
            [
                {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, default=str),
                },
            ],
            callbacks=self.model_callbacks,
        )

    def _run_main_retry(
        self,
        *,
        context: AgentContext,
        target_language: str,
        mode: RetryMode,
        feedback: str,
        immutable_history: list[dict[str, object]],
        trace: TraceRecorder,
    ) -> SupervisorCandidate:
        payload = {
            "request": context.request_text,
            "feedback": feedback,
            "tool_history": immutable_history,
            "target_language": target_language,
        }
        user_message = json.dumps(payload, ensure_ascii=False, default=str)
        if mode == RetryMode.ANSWER_REWRITE:
            response = self.model.invoke(
                [
                    {"role": "system", "content": ANSWER_REWRITE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                config=(
                    {"callbacks": self.model_callbacks}
                    if self.model_callbacks
                    else None
                ),
            )
            content = self._last_text([response])
            history = immutable_history
        else:
            tools = [
                candidate
                for candidate in self._create_tools(context, trace)
                if candidate.name == "search_recipes"
            ]
            agent = create_agent(
                model=self.model,
                tools=tools,
                system_prompt=RECIPE_RECOVERY_SYSTEM_PROMPT,
            )
            invoke_config = {"recursion_limit": self.recursion_limit}
            if self.model_callbacks:
                invoke_config["callbacks"] = self.model_callbacks
            response = agent.invoke(
                {"messages": [{"role": "user", "content": user_message}]},
                config=invoke_config,
            )
            messages = response.get("messages", [])
            content = self._last_text(messages)
            history = [*immutable_history, *self._tool_history(messages)]
        canonical = canonical_tool_response(history, target_language=target_language)
        if canonical is not None:
            content = canonical
        content = append_confidence_notice(content, context.ingredients)
        return SupervisorCandidate(content=content, tool_history=history)

    def _create_tools(
        self, context: AgentContext, trace: TraceRecorder
    ) -> list[Any]:
        repository = self.inventory
        recipe_router = self.recipes
        action_service = self.actions
        profile_action_service = self.profile_actions
        resolved_instance_ids: set[str] = set()
        ambiguous_instance_ids: set[str] = set()
        looked_up_names: set[str] = set()

        def remember_instances(
            instances: list[dict[str, object]],
            *,
            exact_instance: bool = False,
        ) -> None:
            counts_by_name: dict[str, int] = {}
            for instance in instances:
                name = str(instance.get("name", ""))
                counts_by_name[name] = counts_by_name.get(name, 0) + 1
                looked_up_names.add(name)
            for instance in instances:
                instance_id = str(instance.get("instance_id", ""))
                name = str(instance.get("name", ""))
                if exact_instance or counts_by_name[name] == 1:
                    resolved_instance_ids.add(instance_id)
                    ambiguous_instance_ids.discard(instance_id)
                else:
                    resolved_instance_ids.discard(instance_id)
                    ambiguous_instance_ids.add(instance_id)

        def require_resolved_instance(instance_id: str) -> None:
            if instance_id in ambiguous_instance_ids:
                raise ValueError("AMBIGUOUS_INSTANCE_MATCH")
            if instance_id not in resolved_instance_ids:
                raise ValueError("INSTANCE_LOOKUP_REQUIRED")

        def user_selected_instance(instance_id: str) -> bool:
            return instance_id in context.request_text

        @tool
        def list_inventory_instances() -> list[dict[str, object]]:
            """List every stored inventory instance with its ID and exact quantity text."""
            instances = trace.run(
                "database",
                "list_inventory_instances",
                repository.list_instances,
                lambda values: f"{len(values)} item(s)",
            )
            remember_instances(instances)
            return instances

        @tool
        def find_inventory_instances(name: str) -> list[dict[str, object]]:
            """Find all inventory instances matching one exact ingredient name."""
            instances = trace.run(
                "database",
                "find_inventory_instances",
                lambda: repository.find_by_name(name),
                lambda values: f"{len(values)} item(s)",
            )
            if instances:
                remember_instances(instances)
            return instances

        @tool
        def get_inventory_instance(instance_id: str) -> dict[str, object] | None:
            """Get one inventory instance by its exact instance ID."""
            instance = trace.run(
                "database",
                "get_inventory_instance",
                lambda: repository.get_instance(instance_id),
                lambda value: "found" if value else "not found",
            )
            if instance is not None:
                if user_selected_instance(instance_id):
                    remember_instances([instance], exact_instance=True)
                else:
                    matches = repository.find_by_name(str(instance["name"]))
                    remember_instances(matches)
            return instance

        @tool
        def propose_add_inventory_instance(
            name: str,
            quantity_text: str,
            expires_at: datetime | None = None,
        ) -> dict[str, object]:
            """Propose one new instance using exact quantity text and optional user expiry."""
            action = trace.run(
                "database",
                "propose_add_inventory_instance",
                lambda: action_service.propose_add(
                    context.user_id,
                    name,
                    quantity_text,
                    user_expires_at=expires_at,
                ).model_dump(mode="json"),
                lambda value: f"status={value['status']}",
            )
            arguments = action.get("arguments")
            if (
                isinstance(arguments, dict)
                and arguments.get("expiry_source") == "unknown"
            ):
                trace.record(
                    "model",
                    "expiry_inference",
                    "degraded",
                    "unknown expiry; proposal remains pending",
                )
            return action

        @tool
        def propose_replace_inventory_instance(
            instance_id: str,
            quantity_text: str,
            expires_at: datetime | None,
            expiry_source: Literal["llm", "user", "unknown"],
        ) -> dict[str, object]:
            """Propose a complete replacement for one looked-up inventory instance."""
            require_resolved_instance(instance_id)
            return trace.run(
                "database",
                "propose_replace_inventory_instance",
                lambda: action_service.propose(
                    context.user_id,
                    "inventory.replace",
                    {
                        "instance_id": instance_id,
                        "quantity_text": quantity_text,
                        "expires_at": (
                            expires_at.isoformat() if expires_at is not None else None
                        ),
                        "expiry_source": expiry_source,
                    },
                    source_task_id=context.inventory_source_task_id,
                ).model_dump(mode="json"),
                lambda value: f"status={value['status']}",
            )

        @tool
        def propose_remove_inventory_instance(instance_id: str) -> dict[str, object]:
            """Propose deleting one looked-up inventory instance by exact ID."""
            require_resolved_instance(instance_id)
            return trace.run(
                "database",
                "propose_remove_inventory_instance",
                lambda: action_service.propose(
                    context.user_id,
                    "inventory.remove",
                    {"instance_id": instance_id},
                    source_task_id=context.inventory_source_task_id,
                ).model_dump(mode="json"),
                lambda value: f"status={value['status']}",
            )

        @tool
        def propose_remove_all_inventory_instances(name: str) -> dict[str, object]:
            """Propose deleting every looked-up inventory instance with an exact name."""
            if name not in looked_up_names:
                raise ValueError("INSTANCE_LOOKUP_REQUIRED")
            return trace.run(
                "database",
                "propose_remove_all_inventory_instances",
                lambda: action_service.propose(
                    context.user_id,
                    "inventory.remove_all",
                    {"name": name},
                ).model_dump(mode="json"),
                lambda value: f"status={value['status']}",
            )

        @tool
        def propose_confirm_inventory_instance(instance_id: str) -> dict[str, object]:
            """Propose confirming one looked-up inventory instance is still current."""
            require_resolved_instance(instance_id)
            return trace.run(
                "database",
                "propose_confirm_inventory_instance",
                lambda: action_service.propose(
                    context.user_id,
                    "inventory.confirm",
                    {"instance_id": instance_id},
                ).model_dump(mode="json"),
                lambda value: f"status={value['status']}",
            )

        @tool
        def search_recipes(
            ingredients: list[str],
            ingredient_mapping: list[IngredientTranslation],
            exclude_ingredients: list[str] | None = None,
            diet: str | None = None,
            cuisine: str | None = None,
            meal_type: str | None = None,
            max_ready_time: int | None = None,
            ranking: Literal[
                "default",
                "max-used-ingredients",
                "min-missing-ingredients",
            ] = "default",
        ) -> list[dict[str, object]]:
            """Search safe recipes with a complete Chinese-to-English ingredient mapping."""
            if not ingredients:
                raise ValueError("At least one ingredient is required")
            english_terms = [
                value for value in (diet, cuisine, meal_type) if value is not None
            ]
            if any(
                _ENGLISH_API_TERM.fullmatch(value) is None
                for value in english_terms
            ):
                raise ValueError("INVALID_ENGLISH_API_TERM")
            if max_ready_time is not None and max_ready_time <= 0:
                raise ValueError("max_ready_time must be positive")
            normalize_search_request(
                ingredients=ingredients,
                ingredient_mapping=ingredient_mapping,
                allergen_categories=list(context.allergen_intolerances),
                custom_allergens=list(context.custom_allergens),
                exclude_ingredients=exclude_ingredients or [],
                diet=diet,
                cuisine=cuisine,
                meal_type=meal_type,
                max_ready_time=max_ready_time,
                ranking=RecipeRanking(ranking),
            )
            return recipe_router.search(
                ingredients=ingredients,
                ingredient_mapping=ingredient_mapping,
                allergen_categories=list(context.allergen_intolerances),
                custom_allergens=list(context.custom_allergens),
                exclude_ingredients=exclude_ingredients or [],
                diet=diet,
                cuisine=cuisine,
                meal_type=meal_type,
                max_ready_time=max_ready_time,
                ranking=RecipeRanking(ranking),
                trace=trace.record,
            )

        tools = [
            list_inventory_instances,
            find_inventory_instances,
            get_inventory_instance,
            propose_add_inventory_instance,
            propose_replace_inventory_instance,
            propose_remove_inventory_instance,
            propose_remove_all_inventory_instances,
            propose_confirm_inventory_instance,
            search_recipes,
        ]
        if profile_action_service is not None and context.segment_id:

            @tool
            def propose_profile_change(
                category: ProfileCategory,
                key: str,
                operation: ProfileChangeOperation,
                value: str | None,
                reason: str,
            ) -> dict[str, object]:
                """Propose a durable profile change; it remains inactive until clicked."""
                proposal = trace.run(
                    "profile",
                    "propose_profile_change",
                    lambda: profile_action_service.propose(
                        context.user_id,
                        ProfileChangeRequest(
                            category=category,
                            key=key,
                            operation=operation,
                            value=value,
                            reason=reason,
                        ),
                        [context.segment_id],
                        trusted_current_segment_id=context.segment_id,
                    ),
                    lambda result: f"status={result.status}",
                )
                return proposal.model_dump(mode="json")

            tools.append(propose_profile_change)
        if self.history is not None:
            history_repository = self.history

            @tool
            def search_conversation_history_by_time(
                start_at: str,
                end_at: str,
                limit: int = 20,
            ) -> list[dict[str, object]] | dict[str, object]:
                """Recall this user's input and output in an explicit time range."""
                try:
                    validated = HistoryTimeRangeInput(
                        start_at=start_at,
                        end_at=end_at,
                        limit=limit,
                    )
                except ValidationError:
                    return {
                        "error": {
                            "code": "INVALID_HISTORY_TIME_RANGE",
                            "message": (
                                "时间范围必须使用带时区的有效起止时间，"
                                "且 limit 为 1 到 100。"
                            ),
                        }
                    }
                records = trace.run(
                    "database",
                    "search_conversation_history_by_time",
                    lambda: history_repository.search_by_time(
                        context.user_id,
                        validated.start_at,
                        validated.end_at,
                        limit=validated.limit,
                    ),
                    lambda values: f"{len(values)} record(s)",
                )
                return [item.model_dump(mode="json") for item in records]

            tools.append(search_conversation_history_by_time)
        return tools

    @staticmethod
    def _profile_values(context: AgentContext, field: str) -> list[str]:
        if field == "preferences":
            values = []
            active = False
            for line in context.profile_markdown.splitlines():
                if line.strip() == "## 饮食偏好":
                    active = True
                elif line.startswith("## "):
                    active = False
                elif active and line.strip().startswith("- "):
                    value = line.strip()[2:].strip()
                    if value not in {"", "待填写", "暂无"}:
                        values.append(value)
            return values
        return []

    @staticmethod
    def _user_message(
        context: AgentContext,
        target_language: str,
        *,
        current_time: datetime | None = None,
        application_timezone: str = "Asia/Shanghai",
    ) -> str:
        timezone = ZoneInfo(application_timezone)
        if current_time is None:
            reference = datetime.now(timezone)
        else:
            if current_time.tzinfo is None or current_time.utcoffset() is None:
                raise ValueError("current_time must be timezone-aware")
            reference = current_time.astimezone(timezone)
        payload = {
            "request": context.request_text,
            "current_date": reference.date().isoformat(),
            "current_time": reference.isoformat(),
            "application_timezone": application_timezone,
            "intent_hint": context.intent,
            "ingredients": [item.model_dump(mode="json") for item in context.ingredients],
            "allergens": context.allergens,
            "allergen_intolerances": context.allergen_intolerances,
            "custom_allergens": context.custom_allergens,
            "user_profile_markdown": context.profile_markdown,
            "recent_history": [
                segment.model_dump(mode="json")
                for segment in context.history_segments
            ],
            "cooking_task": (
                context.cooking_task.model_dump(mode="json")
                if context.cooking_task is not None
                else None
            ),
            "vision_confirmation_result": context.vision_confirmation_result,
            "target_language": target_language,
        }
        return (
            "请根据以下结构化上下文完成用户请求。最终回答使用 target_language 指定的语言。\n"
            + json.dumps(payload, ensure_ascii=False)
        )

    @staticmethod
    def _last_text(messages: list[Any]) -> str:
        for message in reversed(messages):
            if isinstance(message, ToolMessage):
                continue
            content = getattr(message, "content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                texts = [
                    str(block.get("text", ""))
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                if any(texts):
                    return "\n".join(texts).strip()
        raise RuntimeError("Agent returned no final text")

    @staticmethod
    def _tool_history(messages: list[Any]) -> list[dict[str, object]]:
        arguments_by_id: dict[str, object] = {}
        for message in messages:
            for call in getattr(message, "tool_calls", []) or []:
                if not isinstance(call, dict):
                    continue
                call_id = str(call.get("id", ""))
                if call_id:
                    arguments_by_id[call_id] = call.get("args", {})
        history: list[dict[str, object]] = []
        for message in messages:
            if isinstance(message, ToolMessage):
                call_id = message.tool_call_id
                history.append(
                    {
                        "tool": message.name or "unknown",
                        "tool_call_id": call_id,
                        "arguments": arguments_by_id.get(call_id, {}),
                        "result": message.content,
                    }
                )
        return history


def build_agent(
    settings: Settings,
    inventory: InventoryRepository,
    recipes: RecipeRouter,
    actions: InventoryActionService,
    history: ConversationHistoryRepository,
    profile_actions: ProfileActionService | None = None,
    model_callbacks: list[Any] | None = None,
) -> KitchenAgent:
    settings.require("deepseek_api_key", "deepseek_model", "deepseek_base_url")
    model = ChatOpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_model,
        temperature=0,
        timeout=settings.http_timeout_seconds,
        max_retries=1,
        max_tokens=settings.deepseek_max_output_tokens,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return KitchenAgent(
        model=model,
        inventory=inventory,
        recipes=recipes,
        actions=actions,
        history=history,
        profile_actions=profile_actions,
        application_timezone=settings.application_timezone,
        recursion_limit=settings.agent_recursion_limit,
        max_input_chars=settings.deepseek_max_input_chars,
        model_callbacks=list(model_callbacks or []),
    )


def _profile_proposals(
    history: list[dict[str, object]],
) -> list[ProfileProposal]:
    proposals: list[ProfileProposal] = []
    for entry in history:
        if entry.get("tool") != "propose_profile_change":
            continue
        value = entry.get("result")
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                continue
        if isinstance(value, dict):
            try:
                proposals.append(ProfileProposal.model_validate(value))
            except ValidationError:
                continue
    return proposals


def run_agent(
    agent: KitchenAgent,
    context: AgentContext,
    target_language: str = "zh",
) -> AgentResult:
    return agent.run(context, target_language)
