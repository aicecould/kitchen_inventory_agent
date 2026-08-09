"""DeepSeek-powered LangChain tool-calling Agent."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import Any, Literal

from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator

from app.actions import InventoryActionService
from app.allergens import AllergenIntolerance
from app.audit import (
    append_confidence_notice,
    canonical_tool_response,
    regex_audit,
    review_tool_calls,
)
from app.config import Settings
from app.context import AgentContext, AgentResult
from app.prompts import SYSTEM_PROMPT
from app.tools.inventory import InventoryRepository
from app.tools.recipe import RecipeRouter
from app.trace import TraceRecorder, elapsed_ms


class AllergenTranslation(BaseModel):
    original: str = Field(min_length=1, max_length=100)
    api_term: str = Field(min_length=1, max_length=100)

    @field_validator("original", "api_term")
    @classmethod
    def strip_value(cls, value: str) -> str:
        return value.strip()


_ENGLISH_API_TERM = re.compile(r"^[A-Za-z][A-Za-z0-9 '&()/.+-]*$")
_ALLERGEN_TOOL_ERRORS = frozenset(
    {
        "MISSING_ALLERGEN_FILTER",
        "UNEXPECTED_ALLERGEN_MAPPING",
        "INVALID_ENGLISH_API_TERM",
    }
)


@dataclass(slots=True)
class KitchenAgent:
    model: ChatOpenAI
    inventory: InventoryRepository
    recipes: RecipeRouter
    actions: InventoryActionService
    recursion_limit: int = 10
    max_input_chars: int = 12000

    def run(self, context: AgentContext, target_language: str = "zh") -> AgentResult:
        trace = TraceRecorder()
        user_message = self._user_message(context, target_language)
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
            response = agent.invoke(
                {"messages": [{"role": "user", "content": user_message}]},
                config={"recursion_limit": self.recursion_limit},
            )
        except Exception as exc:
            trace.record(
                "model",
                "deepseek_agent",
                "failed",
                type(exc).__name__,
                elapsed_ms(started),
            )
            raise
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
        canonical = canonical_tool_response(history)
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
        if not audit.passed:
            return AgentResult(
                content="输出未通过安全检查。",
                blocked=True,
                audit_reason=audit.reason,
                tool_history=history,
                execution_trace=trace.events,
            )
        tool_audit = review_tool_calls(
            content,
            history,
            ingredients=context.ingredients,
            vision_confirmation_result=context.vision_confirmation_result,
        )
        trace.record(
            "output",
            "tool_call_audit",
            "passed" if tool_audit.passed else "blocked",
            tool_audit.reason or "passed",
        )
        if not tool_audit.passed:
            return AgentResult(
                content="输出与工具执行记录不一致，本次结果已被拦截。",
                blocked=True,
                audit_reason=tool_audit.reason,
                tool_history=history,
                execution_trace=trace.events,
            )
        return AgentResult(
            content=content,
            tool_history=history,
            execution_trace=trace.events,
        )

    def _create_tools(
        self, context: AgentContext, trace: TraceRecorder
    ) -> list[Any]:
        repository = self.inventory
        recipe_router = self.recipes
        action_service = self.actions
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
            intolerances: list[AllergenIntolerance],
            exclude_ingredients: list[str],
            custom_allergen_mapping: list[AllergenTranslation],
            diet: str | None = None,
            cuisine: str | None = None,
            meal_type: str | None = None,
            max_ready_time: int | None = None,
            ranking: Literal[
                "max-used-ingredients", "min-missing-ingredients"
            ] = "max-used-ingredients",
        ) -> list[dict[str, object]]:
            """Search safe recipes using English API terms and explicit custom-allergen translations."""
            if not ingredients:
                raise ValueError("At least one ingredient is required")
            missing_broad = set(context.allergen_intolerances) - set(intolerances)
            configured_custom = {
                value.strip().casefold(): value for value in context.custom_allergens
            }
            translated_custom = {
                item.original.casefold(): item.api_term
                for item in custom_allergen_mapping
            }
            missing_custom = set(configured_custom) - set(translated_custom)
            if missing_broad or missing_custom:
                raise ValueError("MISSING_ALLERGEN_FILTER")
            if set(translated_custom) - set(configured_custom):
                raise ValueError("UNEXPECTED_ALLERGEN_MAPPING")
            english_terms = [
                *ingredients,
                *exclude_ingredients,
                *translated_custom.values(),
                *(value for value in (diet, cuisine, meal_type) if value is not None),
            ]
            if any(
                _ENGLISH_API_TERM.fullmatch(value) is None
                for value in english_terms
            ):
                raise ValueError("INVALID_ENGLISH_API_TERM")
            if max_ready_time is not None and max_ready_time <= 0:
                raise ValueError("max_ready_time must be positive")
            api_exclusions = list(
                dict.fromkeys([*exclude_ingredients, *translated_custom.values()])
            )
            return recipe_router.search(
                ingredients=ingredients,
                intolerances=intolerances,
                exclude_ingredients=api_exclusions,
                diet=diet,
                cuisine=cuisine,
                meal_type=meal_type,
                max_ready_time=max_ready_time,
                ranking=ranking,
                trace=trace.record,
            )

        return [
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
    def _user_message(context: AgentContext, target_language: str) -> str:
        payload = {
            "request": context.request_text,
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
    )
    return KitchenAgent(
        model=model,
        inventory=inventory,
        recipes=recipes,
        actions=actions,
        recursion_limit=settings.agent_recursion_limit,
        max_input_chars=settings.deepseek_max_input_chars,
    )


def run_agent(
    agent: KitchenAgent,
    context: AgentContext,
    target_language: str = "zh",
) -> AgentResult:
    return agent.run(context, target_language)
