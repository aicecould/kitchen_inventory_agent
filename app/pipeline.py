"""Top-level backend pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from classifier.model import FoodQueryClassifier
from app.actions import (
    InventoryActionService,
    PendingAction,
    PendingActionRepository,
)
from app.adapters.recipe_api import (
    DeepSeekWebSearchClient,
    SpoonacularClient,
    TheMealDbClient,
)
from app.adapters.vision_api import VisionApiClient
from app.agent import KitchenAgent, build_agent
from app.audit import (
    AuditCode,
    append_confidence_notice,
    canonical_tool_response,
    regex_audit,
    review_tool_calls,
)
from app.config import Settings, get_settings
from app.context import (
    AgentContext,
    AgentResult,
    CookingTaskContext,
    ConfidenceStatus,
    ExecutionTraceEvent,
    Ingredient,
    VisionConfirmationPrompt,
)
from app.expiry import DeepSeekExpiryInferer
from app.intent import (
    RouteDecision,
    match_intent,
    match_simple_inventory_query,
    route_deterministically,
)
from app.memory import (
    ConversationOutcome,
    ConversationRecord,
    ConversationHistoryRepository,
    UserProfile,
    profile_without_legacy_history,
    read_user_profile,
)
from app.profile import ProfileProposal, ProfileRepository
from app.profile_actions import ProfileActionService
from app.profile_summary import ProfileSummarizer
from app.proactive import ProactiveService, ReminderGenerator, VisitRepository
from app.supervisor import fixed_response
from app.tools.inventory import InventoryRepository
from app.tools.recipe import RecipeRouter
from app.tasks import CookingTask, CookingTaskRepository, CookingTaskService
from app.trace import TraceRecorder
from app.llm_log import active_store, log_store, logged_span, request_context
from app.vision import recognize_ingredients
from app.vision_confirmations import VisionConfirmationRepository
from recommender.models import RecommenderArtifactsError
from recommender.search import LocalRecipeRecommender, load_recommender


def build_runtime_recommender(settings: Settings) -> LocalRecipeRecommender | None:
    """Load read-only local artifacts once; never build, download, or repair them."""
    if not settings.recommender_enabled:
        return None
    try:
        return load_recommender(
            db_path=settings.recommender_db_path,
            faiss_path=settings.recommender_faiss_path,
            metadata_path=settings.recommender_metadata_path,
            model_path=settings.recommender_model_path,
            expected_model_revision=settings.recommender_model_revision,
        )
    except (RecommenderArtifactsError, OSError, RuntimeError, ImportError) as exc:
        logging.getLogger(__name__).warning(
            "local recipe recommender unavailable: %s",
            type(exc).__name__,
        )
        return None


@dataclass(slots=True)
class KitchenPipeline:
    settings: Settings
    profile: UserProfile
    inventory: InventoryRepository
    actions: InventoryActionService
    history: ConversationHistoryRepository
    tasks: CookingTaskService
    profile_actions: ProfileActionService
    profile_summarizer: ProfileSummarizer | None
    vision_confirmations: VisionConfirmationRepository
    agent: KitchenAgent | None
    vision: VisionApiClient | None
    food_classifier: FoodQueryClassifier | None
    food_classifier_load_error: str | None = None
    local_recipe_search_available: bool = False

    def process_request(
        self,
        *,
        user_id: str,
        text: str,
        image_bytes: bytes | None = None,
        target_language: str = "zh",
        initial_trace: list[ExecutionTraceEvent] | None = None,
        reconciliation_task_id: str | None = None,
    ) -> AgentResult:
        segment_id = self.history.reserve_segment_id()
        store = log_store(self.settings)
        image_reference = store.capture_image(image_bytes) if store else None
        with request_context(user_id, segment_id), logged_span(
            store, "request", "process_request", prompt={
                "text": text, "target_language": target_language,
                "reconciliation_task_id": reconciliation_task_id,
                "image": image_reference,
                "image_present": image_bytes is not None,
                "profile_markdown": self.profile.markdown,
            }
        ) as span:
            result = self._process_request(
                user_id=user_id,
                text=text,
                image_bytes=image_bytes,
                target_language=target_language,
                initial_trace=initial_trace,
                segment_id=segment_id,
                reconciliation_task_id=reconciliation_task_id,
            )
            result.segment_id = segment_id
            result.recipe_options = _recipe_options(result.tool_history)
            if (
                reconciliation_task_id is not None
                and not result.blocked
                and _has_sourced_inventory_proposal(
                    result.tool_history, reconciliation_task_id
                )
            ):
                reconciliation_task = self.tasks.get_reconciliation_task(
                    user_id, reconciliation_task_id
                )
                self.tasks.close_reconciliation(
                    user_id=user_id,
                    task_id=reconciliation_task_id,
                    expected_version=reconciliation_task.version,
                    request_id=f"reconcile-{segment_id}",
                )
            self._record_result(
                user_id=user_id,
                request_text=text,
                result=result,
                confirmation_result=None,
                segment_id=segment_id,
            )
            if self.profile_summarizer is not None:
                result.profile_proposals.extend(
                    self.profile_summarizer.run_if_due(user_id)
                )
                self.profile = self.profile_actions.repository.read()
            span["content"] = {"content": result.content, "blocked": result.blocked,
                               "audit_reason": result.audit_reason,
                               "execution_trace": result.execution_trace,
                               "tool_history": result.tool_history,
                               "profile_proposals": result.profile_proposals,
                               "recipe_options": result.recipe_options,
                               "public_response": result.public_payload()}
            return result

    def _process_request(
        self,
        *,
        user_id: str,
        text: str,
        image_bytes: bytes | None = None,
        target_language: str = "zh",
        initial_trace: list[ExecutionTraceEvent] | None = None,
        segment_id: str,
        reconciliation_task_id: str | None = None,
    ) -> AgentResult:
        trace = TraceRecorder()
        trace.events.extend(initial_trace or [])
        route_decision: RouteDecision | None = None
        force_agent = False
        if reconciliation_task_id is not None:
            self.tasks.get_reconciliation_task(user_id, reconciliation_task_id)
            force_agent = True
        if image_bytes is not None:
            trace.record(
                "routing",
                "food_query_classifier",
                "skipped",
                "image request bypasses text-only classifier",
            )
        else:
            with logged_span(active_store(), "component", "deterministic_route",
                             prompt={"text": text}) as routing_span:
                route_decision = route_deterministically(text)
                routing_span["content"] = route_decision
            route_detail = route_decision.kind
            if route_decision.operation is not None:
                route_detail += f"; {route_decision.operation}"
            trace.record(
                "routing",
                "deterministic_route",
                "undecided" if route_decision.kind == "undecided" else "matched",
                route_detail,
            )
            if route_decision.kind == "irrelevant":
                return AgentResult(
                    content="当前仅支持食谱和食材相关问题。",
                    blocked=True,
                    audit_reason="food_query_irrelevant",
                    execution_trace=trace.events,
                )
            force_agent = route_decision.kind == "contextual"

        if (
            image_bytes is None
            and route_decision is not None
            and route_decision.kind == "undecided"
        ):
            if self.food_classifier is None:
                trace.record(
                    "routing",
                    "food_query_classifier",
                    "unavailable",
                    (
                        f"fail-open; {self.food_classifier_load_error}"
                        if self.food_classifier_load_error
                        else "disabled; fail-open"
                    ),
                )
            else:
                try:
                    classification = trace.run(
                        "routing",
                        "food_query_classifier",
                        lambda: self.food_classifier.classify(text),
                        lambda value: (
                            f"related={str(value.is_related).lower()}; "
                            f"score={value.positive_score:.4f}; "
                            f"threshold={value.threshold:.4f}"
                        ),
                        inputs={"text": text},
                    )
                except Exception:
                    trace.record(
                        "routing",
                        "food_query_classifier_fallback",
                        "fail-open",
                        "prediction failed; continuing request",
                    )
                    classification = None

                if classification is not None and not classification.is_related:
                    return AgentResult(
                        content="当前仅支持食谱和食材相关问题。",
                        blocked=True,
                        audit_reason="food_query_irrelevant",
                        execution_trace=trace.events,
                    )

        with logged_span(active_store(), "component", "intent_match", prompt={"text": text}) as intent_span:
            intent = match_intent(text)
            intent_span["content"] = intent
        trace.record("routing", "intent_match", "success", intent)

        if intent == "order_unsupported":
            return self._finish_direct(
                user_id=user_id,
                request_text=text,
                target_language=target_language,
                content="当前原型暂不支持订单或购物车管理。",
                tool_history=[],
                trace=trace,
            )

        ingredients: list[Ingredient] = []
        prior_tool_history: list[dict[str, object]] = []
        if image_bytes is not None:
            if self.vision is None:
                raise ValueError("Baidu image API credentials are missing from .env")
            ingredients = trace.run(
                "external_api",
                "baidu_vision",
                lambda: recognize_ingredients(image_bytes, self.vision),
                lambda values: f"{len(values)} detection(s)",
            )
            prior_tool_history = [
                {
                    "tool": "recognize_ingredients",
                    "arguments": {"image": "validated-upload"},
                    "result": [
                        item.model_dump(mode="json") for item in ingredients
                    ],
                }
            ]
            low_confidence = [
                item
                for item in ingredients
                if item.status == ConfidenceStatus.REJECTED
            ]
            if low_confidence:
                pending = self.vision_confirmations.create(
                    user_id=user_id,
                    request_text=text,
                    target_language=target_language,
                    detections=ingredients,
                )
                trace.record(
                    "human_checkpoint",
                    "low_confidence_image",
                    "pending",
                    f"{len(low_confidence)} candidate(s) require confirmation",
                )
                return AgentResult(
                    content=(
                        "图片中有低置信度的食材候选，尚未交给 Agent。"
                        "请选择、修改或放弃这些候选；确认后将继续原请求。"
                    ),
                    execution_trace=trace.events,
                    awaiting_vision_confirmation=True,
                    vision_confirmation=VisionConfirmationPrompt(
                        confirmation_id=pending.confirmation_id,
                        candidates=low_confidence,
                    ),
                )

        return self._continue_request(
            user_id=user_id,
            text=text,
            target_language=target_language,
            ingredients=ingredients,
            confirmation_result=None,
            trace=trace,
            prior_tool_history=prior_tool_history,
            route_decision=route_decision,
            force_agent=force_agent,
            segment_id=segment_id,
            reconciliation_task_id=reconciliation_task_id,
        )

    def resolve_vision_confirmation(
        self,
        *,
        user_id: str,
        confirmation_id: str,
        selections: list[dict[str, object]],
    ) -> AgentResult:
        pending = self.vision_confirmations.get(confirmation_id, user_id)
        if pending.status == "resolved":
            if pending.result is None:
                raise RuntimeError("Resolved vision confirmation has no stored result")
            self._record_result(
                user_id=user_id,
                request_text=f"图片确认：{confirmation_id}",
                result=pending.result,
                confirmation_result=pending.resolution,
                outcome_type=ConversationOutcome.CONFIRMATION,
            )
            return pending.result
        if pending.status != "pending":
            raise ValueError(
                f"Vision confirmation cannot continue from status: {pending.status}"
            )

        low_candidates = pending.low_confidence_candidates
        selected_indices: set[int] = set()
        selected_records: list[dict[str, object]] = []
        confirmed_ingredients: list[Ingredient] = []
        for selection in selections:
            index_value = selection.get("candidate_index")
            name = str(selection.get("name", "")).strip()
            if not isinstance(index_value, int) or isinstance(index_value, bool):
                raise ValueError("candidate_index must be an integer")
            if index_value < 0 or index_value >= len(low_candidates):
                raise ValueError("Unknown low-confidence candidate")
            if index_value in selected_indices:
                raise ValueError("A low-confidence candidate was selected more than once")
            if not name or len(name) > 100 or "\n" in name or "\r" in name:
                raise ValueError("Confirmed ingredient name is invalid")
            selected_indices.add(index_value)
            source = low_candidates[index_value]
            confirmed_ingredients.append(
                Ingredient(
                    name=name,
                    confidence=source.confidence,
                    status=ConfidenceStatus.USER_CONFIRMED,
                )
            )
            selected_records.append(
                {
                    "candidate_index": index_value,
                    "original_name": source.name,
                    "name": name,
                    "confidence": source.confidence,
                }
            )

        dropped = [
            {
                "candidate_index": index,
                "name": item.name,
                "confidence": item.confidence,
            }
            for index, item in enumerate(low_candidates)
            if index not in selected_indices
        ]
        resolution: dict[str, object] = {
            "type": "vision",
            "confirmation_id": confirmation_id,
            "selected": selected_records,
            "dropped": dropped,
        }
        ingredients = [
            item
            for item in pending.detections
            if item.status != ConfidenceStatus.REJECTED
        ]
        ingredients.extend(confirmed_ingredients)

        trace = TraceRecorder()
        trace.record(
            "human_checkpoint",
            "low_confidence_image",
            "resolved",
            f"selected={len(selected_records)}; dropped={len(dropped)}",
        )
        segment_id = self.history.reserve_segment_id()
        result = self._continue_request(
            user_id=user_id,
            text=pending.request_text,
            target_language=pending.target_language,
            ingredients=ingredients,
            confirmation_result=resolution,
            trace=trace,
            prior_tool_history=[
                {
                    "tool": "recognize_ingredients",
                    "arguments": {"image": "validated-upload"},
                    "result": [
                        item.model_dump(mode="json") for item in pending.detections
                    ],
                },
                {
                    "tool": "resolve_vision_confirmation",
                    "arguments": {"confirmation_id": confirmation_id},
                    "result": resolution,
                },
            ],
            segment_id=segment_id,
        )
        completed = self.vision_confirmations.complete(
            confirmation_id,
            user_id,
            resolution=resolution,
            result=result,
        )
        assert completed.result is not None
        self._record_result(
            user_id=user_id,
            request_text=f"图片确认：{confirmation_id}",
            result=completed.result,
            confirmation_result=resolution,
            outcome_type=ConversationOutcome.CONFIRMATION,
            segment_id=segment_id,
        )
        return completed.result

    def resolve_profile_proposal(
        self,
        *,
        user_id: str,
        proposal_id: str,
        decision: str,
    ) -> ProfileProposal:
        if decision == "confirm":
            proposal = self.profile_actions.confirm(proposal_id, user_id)
            response = "用户画像变更已确认并写入。"
            status = "executed"
        elif decision == "reject":
            proposal = self.profile_actions.reject(proposal_id, user_id)
            response = "用户画像变更已拒绝，画像未改变。"
            status = "cancelled"
        else:
            raise ValueError("Unknown profile confirmation decision")
        self.profile = self.profile_actions.repository.read()
        self.history.append(
            user_id=user_id,
            request=f"{'确认' if decision == 'confirm' else '拒绝'}画像变更：{proposal_id}",
            response=response,
            tool_history=[
                {
                    "tool": f"{decision}_profile_change",
                    "arguments": {"proposal_id": proposal_id},
                    "result": proposal.model_dump(mode="json"),
                }
            ],
            confirmation_result={
                "type": "profile",
                "decision": decision,
                "proposal_id": proposal_id,
                "status": proposal.status.value,
            },
            outcome_type=ConversationOutcome.CONFIRMATION,
            execution_trace=[
                ExecutionTraceEvent(
                    stage="human_checkpoint",
                    name="profile_confirmation",
                    status=status,
                    detail=f"{decision}; {proposal.category.value}; {proposal.key}",
                )
            ],
        )
        return proposal

    def record_inventory_confirmation(
        self,
        *,
        action: PendingAction,
        decision: str,
    ) -> None:
        if decision not in {"confirm", "cancel"}:
            raise ValueError("Unknown inventory confirmation decision")
        executed = action.status == "executed"
        response = "库存操作已执行。" if executed else "库存操作已取消。"
        self.history.append(
            user_id=action.user_id,
            request=f"{'确认' if decision == 'confirm' else '取消'}：{action.summary}",
            response=response,
            tool_history=[
                {
                    "tool": f"{decision}_inventory_action",
                    "arguments": {"action_id": action.action_id},
                    "result": action.model_dump(mode="json"),
                }
            ],
            confirmation_result={
                "type": "inventory",
                "decision": decision,
                "action_id": action.action_id,
                "operation": action.operation,
                "status": action.status,
                "result": action.result,
            },
            outcome_type=ConversationOutcome.CONFIRMATION,
            execution_trace=[
                ExecutionTraceEvent(
                    stage="human_checkpoint",
                    name="inventory_confirmation",
                    status="executed" if executed else "cancelled",
                    detail=f"{decision}; {action.operation}",
                )
            ],
        )

    def _continue_request(
        self,
        *,
        user_id: str,
        text: str,
        target_language: str,
        ingredients: list[Ingredient],
        confirmation_result: dict[str, object] | None,
        trace: TraceRecorder,
        prior_tool_history: list[dict[str, object]] | None = None,
        route_decision: RouteDecision | None = None,
        force_agent: bool = False,
        segment_id: str | None = None,
        reconciliation_task_id: str | None = None,
    ) -> AgentResult:
        prior_tool_history = prior_tool_history or []
        operation = route_decision.operation if route_decision is not None else None
        arguments = route_decision.arguments if route_decision is not None else {}
        if operation == "request.cancel":
            trace.record("routing", "request_cancel", "matched", operation)
            return self._finish_direct(
                user_id=user_id,
                request_text=text,
                target_language=target_language,
                content=fixed_response(AuditCode.USER_CANCELLED, target_language),
                tool_history=prior_tool_history,
                trace=trace,
                ingredients=ingredients,
                confirmation_result=confirmation_result,
                audit_reason_override=AuditCode.USER_CANCELLED.value,
            )
        if not force_agent and (
            operation == "inventory.list"
            or (operation is None and match_simple_inventory_query(text))
        ):
            trace.record(
                "routing", "inventory_query_regex", "matched", "inventory.list"
            )
            items = trace.run(
                "database",
                "list_inventory",
                self.inventory.list_instances,
                lambda values: f"{len(values)} item(s)",
            )
            if items:
                lines = [
                    f"- {item['name']} [{item['instance_id']}]："
                    f"{item['quantity_text']}"
                    for item in items
                ]
                content = "当前库存：\n" + "\n".join(lines)
            else:
                content = "当前库存为空。"
            return self._finish_direct(
                user_id=user_id,
                request_text=text,
                target_language=target_language,
                content=content,
                tool_history=[
                    *prior_tool_history,
                    {
                        "tool": "list_inventory_instances",
                        "arguments": {},
                        "result": items,
                    }
                ],
                trace=trace,
                ingredients=ingredients,
                confirmation_result=confirmation_result,
            )

        if not force_agent and operation == "inventory.find":
            name = str(arguments["name"])
            trace.record("routing", "inventory_query_regex", "matched", operation)
            items = trace.run(
                "database",
                "find_inventory",
                lambda: self.inventory.find_by_name(name),
                lambda values: f"{len(values)} item(s)",
            )
            content = (
                "库存中没有找到该食材。"
                if not items
                else f"{name}库存：\n"
                + "\n".join(
                    f"- {item['name']} [{item['instance_id']}]："
                    f"{item['quantity_text']}"
                    for item in items
                )
            )
            return self._finish_direct(
                user_id=user_id,
                request_text=text,
                target_language=target_language,
                content=content,
                tool_history=[
                    *prior_tool_history,
                    {
                        "tool": "find_inventory_instances",
                        "arguments": {"name": name},
                        "result": items,
                    },
                ],
                trace=trace,
                ingredients=ingredients,
                confirmation_result=confirmation_result,
            )

        simple_operation = None
        if not force_agent and operation in {
            "inventory.add",
            "inventory.replace",
            "inventory.remove",
            "inventory.remove_all",
            "inventory.confirm",
        }:
            simple_operation = RouteDecision("supported", operation, arguments)
        if simple_operation is not None:
            trace.record(
                "routing",
                "inventory_write_regex",
                "matched",
                simple_operation.operation,
            )
            proposal_arguments = dict(simple_operation.arguments)
            if simple_operation.operation == "inventory.add":
                action = trace.run(
                    "database",
                    "create_pending_action",
                    lambda: self.actions.propose_add(
                        user_id,
                        str(simple_operation.arguments["name"]),
                        str(simple_operation.arguments["quantity_text"]),
                        source_task_id=reconciliation_task_id,
                    ),
                    lambda value: f"{value.operation}; status={value.status}",
                )
                if action.arguments.get("expiry_source") == "unknown":
                    trace.record(
                        "model",
                        "expiry_inference",
                        "degraded",
                        "unknown expiry; proposal remains pending",
                    )
            else:
                if simple_operation.operation in {
                    "inventory.replace",
                    "inventory.remove",
                    "inventory.confirm",
                }:
                    instance_id = str(proposal_arguments["instance_id"])
                    current = trace.run(
                        "database",
                        "get_inventory_instance",
                        lambda: self.inventory.get_instance(instance_id),
                        lambda value: "found" if value is not None else "not found",
                    )
                    if current is None:
                        raise KeyError(
                            f"inventory instance not found: {instance_id}"
                        )
                    if simple_operation.operation == "inventory.replace":
                        proposal_arguments = {
                            **proposal_arguments,
                            "expires_at": current["expires_at"],
                            "expiry_source": current["expiry_source"],
                        }
                action = trace.run(
                    "database",
                    "create_pending_action",
                    lambda: self.actions.propose(
                        user_id,
                        simple_operation.operation,
                        proposal_arguments,
                        source_task_id=reconciliation_task_id,
                    ),
                    lambda value: f"{value.operation}; status={value.status}",
                )
            tool_names = {
                "inventory.add": "propose_add_inventory_instance",
                "inventory.replace": "propose_replace_inventory_instance",
                "inventory.remove": "propose_remove_inventory_instance",
                "inventory.remove_all": "propose_remove_all_inventory_instances",
                "inventory.confirm": "propose_confirm_inventory_instance",
            }
            return self._finish_direct(
                user_id=user_id,
                request_text=text,
                target_language=target_language,
                content=f"已生成待确认操作：{action.summary}。确认前库存不会改变。",
                tool_history=[
                    *prior_tool_history,
                    {
                        "tool": tool_names[simple_operation.operation],
                        "arguments": action.arguments,
                        "result": action.model_dump(mode="json"),
                    }
                ],
                trace=trace,
                ingredients=ingredients,
                confirmation_result=confirmation_result,
            )

        if self.agent is None:
            raise ValueError("Missing model configuration in .env for the selected LLM_PROVIDER")

        task = (
            self.tasks.get_reconciliation_task(user_id, reconciliation_task_id)
            if reconciliation_task_id is not None
            else self.tasks.repository.get_active(user_id)
        )
        inventory_snapshot = None
        if match_intent(text) == "recipe" and "库存" in text:
            inventory_snapshot = trace.run(
                "database", "inventory_recipe_context", self.inventory.list_instances,
                lambda values: f"{len(values)} item(s)",
            )
            prior_tool_history.append({
                "tool": "list_inventory_instances", "arguments": {},
                "result": inventory_snapshot, "origin": "business_rule",
            })
        context = AgentContext(
            user_id=user_id,
            segment_id=segment_id or self.history.reserve_segment_id(),
            request_text=text,
            intent=match_intent(text),
            ingredients=ingredients,
            inventory_snapshot=inventory_snapshot,
            allergens=self.profile.allergens,
            allergen_intolerances=self.profile.allergen_intolerances,
            custom_allergens=self.profile.custom_allergens,
            profile_markdown=profile_without_legacy_history(self.profile.markdown),
            history_segments=self.history.list_context_window(user_id),
            cooking_task=_task_context(task),
            inventory_source_task_id=reconciliation_task_id,
            vision_confirmation_result=confirmation_result,
        )
        result = self.agent.run(context, target_language=target_language)
        result.tool_history = [*prior_tool_history, *result.tool_history]
        result.execution_trace = [*trace.events, *result.execution_trace]
        return result

    def _finish_direct(
        self,
        *,
        user_id: str,
        request_text: str,
        target_language: str,
        content: str,
        tool_history: list[dict[str, object]],
        trace: TraceRecorder,
        ingredients: list[Ingredient] | None = None,
        confirmation_result: dict[str, object] | None = None,
        audit_reason_override: str | None = None,
    ) -> AgentResult:
        ingredients = ingredients or []
        canonical = canonical_tool_response(
            tool_history, target_language=target_language
        )
        if canonical is not None:
            content = canonical
        content = append_confidence_notice(content, ingredients)
        regex_result = regex_audit(content)
        trace.record(
            "output",
            "regex_audit",
            "passed" if regex_result.passed else "blocked",
            regex_result.reason or "passed",
        )
        tool_result = review_tool_calls(
            content,
            tool_history,
            target_language=target_language,
            ingredients=ingredients,
            vision_confirmation_result=confirmation_result,
        )
        trace.record(
            "output",
            "tool_call_audit",
            "passed" if tool_result.passed else "blocked",
            tool_result.reason or "passed",
        )
        audit_reason = (
            audit_reason_override or regex_result.reason or tool_result.reason
        )
        blocked = not regex_result.passed or not tool_result.passed
        return AgentResult(
            content=(
                content
                if not blocked
                else "输出与安全规则或工具执行记录不一致，本次结果已被拦截。"
            ),
            blocked=blocked,
            audit_reason=audit_reason,
            tool_history=tool_history,
            execution_trace=trace.events,
        )

    def _record_result(
        self,
        *,
        user_id: str,
        request_text: str,
        result: AgentResult,
        confirmation_result: dict[str, object] | None,
        outcome_type: ConversationOutcome | None = None,
        segment_id: str | None = None,
    ) -> ConversationRecord:
        if outcome_type is None:
            if result.awaiting_vision_confirmation:
                outcome_type = ConversationOutcome.AWAITING_CONFIRMATION
            elif result.blocked:
                outcome_type = ConversationOutcome.BLOCKED
            else:
                outcome_type = ConversationOutcome.COMPLETED
        return self.history.append(
            user_id=user_id,
            request=request_text,
            response=result.content,
            tool_history=result.tool_history,
            confirmation_result=confirmation_result,
            audit_reason=result.audit_reason,
            is_relevant=result.audit_reason != "food_query_irrelevant",
            outcome_type=outcome_type,
            blocked=result.blocked,
            execution_trace=result.execution_trace,
            segment_id=segment_id,
        )

    def record_failure(
        self,
        *,
        user_id: str,
        request_text: str,
        response_text: str,
        execution_trace: list[ExecutionTraceEvent] | None = None,
    ) -> ConversationRecord:
        return self.history.append(
            user_id=user_id,
            request=request_text,
            response=response_text,
            outcome_type=ConversationOutcome.FAILED,
            execution_trace=execution_trace or [],
        )


def build_pipeline(settings: Settings | None = None) -> KitchenPipeline:
    settings = settings or get_settings()
    ZoneInfo(settings.application_timezone)
    profile_repository = ProfileRepository(settings.user_profile_path)
    profile = profile_repository.read()

    inventory = InventoryRepository(settings.inventory_db_path)
    inventory.initialize()
    action_repository = PendingActionRepository(settings.inventory_db_path)
    action_repository.initialize()
    expiry_inferer = (
        DeepSeekExpiryInferer(settings) if settings.llm_enabled else None
    )
    actions = InventoryActionService(
        action_repository,
        inventory,
        ttl_minutes=settings.pending_action_ttl_minutes,
        expiry_inferer=expiry_inferer,
    )
    history = ConversationHistoryRepository(settings.inventory_db_path)
    history.initialize()
    task_repository = CookingTaskRepository(settings.inventory_db_path)
    task_repository.initialize()
    tasks = CookingTaskService(task_repository, history)
    profile_actions = ProfileActionService(
        profile_repository,
        evidence_exists=lambda user_id, segment_id: bool(
            history.get_evidence(user_id, [segment_id])
        ),
    )
    vision_confirmations = VisionConfirmationRepository(
        settings.inventory_db_path,
        ttl_minutes=settings.pending_action_ttl_minutes,
    )
    vision_confirmations.initialize()

    spoonacular: SpoonacularClient | None = None
    if settings.spoonacular_api_key:
        spoonacular = SpoonacularClient(
            base_url=settings.spoonacular_base_url,
            api_key=settings.spoonacular_api_key,
            timeout=settings.http_timeout_seconds,
        )

    themealdb = TheMealDbClient(
        base_url=settings.themealdb_base_url,
        api_key=settings.themealdb_api_key,
        timeout=settings.http_timeout_seconds,
    )
    web_search = (
        DeepSeekWebSearchClient(
            base_url=settings.deepseek_base_url,
            api_key=settings.deepseek_api_key,
            model=settings.deepseek_model,
            timeout=settings.http_timeout_seconds,
            logging_settings=settings,
        )
        if settings.llm_provider == "deepseek" and settings.deepseek_api_key
        else None
    )
    local_recommender = build_runtime_recommender(settings)
    recipes = RecipeRouter(
        local_recommender,
        spoonacular,
        themealdb,
        web_search,
    )
    agent = (
        build_agent(settings, inventory, recipes, actions, history, profile_actions)
        if settings.llm_enabled
        else None
    )
    profile_summarizer = (
        ProfileSummarizer(agent.model, history, profile_actions, callbacks=agent.callbacks)
        if agent is not None
        else None
    )

    vision: VisionApiClient | None = None
    if settings.baidu_image_api_key and settings.baidu_image_secret_key:
        vision = VisionApiClient(
            endpoint=settings.baidu_image_endpoint,
            api_key=settings.baidu_image_api_key,
            secret_key=settings.baidu_image_secret_key,
            timeout=settings.http_timeout_seconds,
        )

    food_classifier: FoodQueryClassifier | None = None
    food_classifier_load_error: str | None = None
    if settings.food_classifier_enabled:
        try:
            food_classifier = FoodQueryClassifier.load(
                settings.food_classifier_model_path,
                settings.food_classifier_metadata_path,
            )
        except (ImportError, KeyError, OSError, TypeError, ValueError) as exc:
            food_classifier_load_error = type(exc).__name__

    return KitchenPipeline(
        settings=settings,
        profile=profile,
        inventory=inventory,
        actions=actions,
        history=history,
        tasks=tasks,
        profile_actions=profile_actions,
        profile_summarizer=profile_summarizer,
        vision_confirmations=vision_confirmations,
        agent=agent,
        vision=vision,
        food_classifier=food_classifier,
        food_classifier_load_error=food_classifier_load_error,
        local_recipe_search_available=local_recommender is not None,
    )


def _task_context(task: CookingTask | None) -> CookingTaskContext | None:
    if task is None:
        return None
    return CookingTaskContext(
        task_id=task.task_id,
        query_text=task.query_text,
        recipe_title=task.recipe.title,
        servings=task.servings,
        ingredients=task.recipe.ingredients,
    )


def _recipe_options(tool_history: list[dict[str, object]]) -> list[dict[str, object]]:
    fields = {
        "id",
        "title",
        "instructions",
        "ingredients",
        "source",
        "source_url",
        "image_url",
    }
    options: list[dict[str, object]] = []
    for entry in tool_history:
        if entry.get("tool") != "search_recipes" or not isinstance(
            entry.get("result"), list
        ):
            continue
        for item in entry["result"]:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                options.append({key: item.get(key) for key in fields})
    return options


def _has_sourced_inventory_proposal(
    tool_history: list[dict[str, object]], task_id: str
) -> bool:
    allowed_tools = {
        "propose_replace_inventory_instance",
        "propose_remove_inventory_instance",
    }
    allowed_operations = {"inventory.replace", "inventory.remove"}
    for entry in tool_history:
        result = entry.get("result")
        if (
            entry.get("tool") in allowed_tools
            and isinstance(result, dict)
            and result.get("status") == "pending"
            and result.get("operation") in allowed_operations
            and result.get("source_task_id") == task_id
        ):
            return True
    return False


def build_proactive_service(pipeline: KitchenPipeline) -> ProactiveService:
    visits = VisitRepository(pipeline.settings.inventory_db_path)
    visits.initialize()
    return ProactiveService(
        visits,
        ReminderGenerator(pipeline.settings),
        pipeline,
    )
