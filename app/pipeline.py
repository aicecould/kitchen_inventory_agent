"""Top-level backend pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from classifier.model import FoodQueryClassifier
from app.actions import (
    InventoryActionService,
    PendingAction,
    PendingActionRepository,
)
from app.adapters.recipe_api import SpoonacularClient, TheMealDbClient
from app.adapters.vision_api import VisionApiClient
from app.agent import KitchenAgent, build_agent
from app.audit import append_confidence_notice, regex_audit, review_tool_calls
from app.config import Settings, get_settings
from app.context import (
    AgentContext,
    AgentResult,
    ConfidenceStatus,
    Ingredient,
    VisionConfirmationPrompt,
)
from app.expiry import DeepSeekExpiryInferer
from app.intent import (
    RouteDecision,
    match_intent,
    match_simple_inventory_operation,
    match_simple_inventory_query,
    route_deterministically,
)
from app.memory import (
    ConversationHistoryRepository,
    UserProfile,
    profile_without_legacy_history,
    read_user_profile,
)
from app.tools.inventory import InventoryRepository
from app.tools.recipe import RecipeRouter
from app.trace import TraceRecorder
from app.vision import recognize_ingredients
from app.vision_confirmations import VisionConfirmationRepository


@dataclass(slots=True)
class KitchenPipeline:
    settings: Settings
    profile: UserProfile
    inventory: InventoryRepository
    actions: InventoryActionService
    history: ConversationHistoryRepository
    vision_confirmations: VisionConfirmationRepository
    agent: KitchenAgent | None
    vision: VisionApiClient | None
    food_classifier: FoodQueryClassifier | None
    food_classifier_load_error: str | None = None

    def process_request(
        self,
        *,
        user_id: str,
        text: str,
        image_bytes: bytes | None = None,
        target_language: str = "zh",
    ) -> AgentResult:
        trace = TraceRecorder()
        route_decision: RouteDecision | None = None
        force_agent = False
        if image_bytes is not None:
            trace.record(
                "routing",
                "food_query_classifier",
                "skipped",
                "image request bypasses text-only classifier",
            )
        else:
            route_decision = route_deterministically(text)
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

        intent = match_intent(text)
        trace.record("routing", "intent_match", "success", intent)

        if intent == "order_unsupported":
            return self._finish_direct(
                user_id=user_id,
                request_text=text,
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
        )
        completed = self.vision_confirmations.complete(
            confirmation_id,
            user_id,
            resolution=resolution,
            result=result,
        )
        assert completed.result is not None
        return completed.result

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
    ) -> AgentResult:
        prior_tool_history = prior_tool_history or []
        operation = route_decision.operation if route_decision is not None else None
        arguments = route_decision.arguments if route_decision is not None else {}
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
                    f"- {item['name']}：{item['quantity_text']}"
                    for item in items
                ]
                content = "当前库存：\n" + "\n".join(lines)
            else:
                content = "当前库存为空。"
            return self._finish_direct(
                user_id=user_id,
                request_text=text,
                content=content,
                tool_history=[
                    *prior_tool_history,
                    {
                        "tool": "list_inventory",
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
                    f"- {item['name']}：{item['quantity_text']}" for item in items
                )
            )
            return self._finish_direct(
                user_id=user_id,
                request_text=text,
                content=content,
                tool_history=[
                    *prior_tool_history,
                    {
                        "tool": "find_inventory",
                        "arguments": {"name": name},
                        "result": items,
                    },
                ],
                trace=trace,
                ingredients=ingredients,
                confirmation_result=confirmation_result,
            )

        simple_operation = (
            None if force_agent or operation is not None
            else match_simple_inventory_operation(text)
        )
        if not force_agent and operation in {"inventory.add", "inventory.remove_all"}:
            simple_operation = RouteDecision("supported", operation, arguments)
        if simple_operation is not None:
            trace.record(
                "routing",
                "inventory_write_regex",
                "matched",
                simple_operation.operation,
            )
            if simple_operation.operation == "inventory.add":
                action = trace.run(
                    "database",
                    "create_pending_action",
                    lambda: self.actions.propose_add(
                        user_id,
                        str(simple_operation.arguments["name"]),
                        str(simple_operation.arguments["quantity_text"]),
                    ),
                    lambda value: f"{value.operation}; status={value.status}",
                )
            else:
                action = trace.run(
                    "database",
                    "create_pending_action",
                    lambda: self.actions.propose(
                        user_id, simple_operation.operation, simple_operation.arguments
                    ),
                    lambda value: f"{value.operation}; status={value.status}",
                )
            return self._finish_direct(
                user_id=user_id,
                request_text=text,
                content=f"已生成待确认操作：{action.summary}。确认前库存不会改变。",
                tool_history=[
                    *prior_tool_history,
                    {
                        "tool": (
                            f"propose_{simple_operation.operation.replace('.', '_')}"
                        ),
                        "arguments": action.arguments,
                        "result": action.model_dump(mode="json"),
                    }
                ],
                trace=trace,
                ingredients=ingredients,
                confirmation_result=confirmation_result,
            )

        if self.agent is None:
            raise ValueError("Missing configuration in .env: deepseek_api_key")

        context = AgentContext(
            user_id=user_id,
            request_text=text,
            intent=match_intent(text),
            ingredients=ingredients,
            allergens=self.profile.allergens,
            allergen_intolerances=self.profile.allergen_intolerances,
            custom_allergens=self.profile.custom_allergens,
            profile_markdown=profile_without_legacy_history(self.profile.markdown),
            history_segments=self.history.list_recent(user_id),
            vision_confirmation_result=confirmation_result,
        )
        result = self.agent.run(context, target_language=target_language)
        result.tool_history = [*prior_tool_history, *result.tool_history]
        result.execution_trace = [*trace.events, *result.execution_trace]
        self._remember(
            user_id=user_id,
            request_text=text,
            result=result,
            confirmation_result=confirmation_result,
        )
        return result

    def _finish_direct(
        self,
        *,
        user_id: str,
        request_text: str,
        content: str,
        tool_history: list[dict[str, object]],
        trace: TraceRecorder,
        ingredients: list[Ingredient] | None = None,
        confirmation_result: dict[str, object] | None = None,
    ) -> AgentResult:
        ingredients = ingredients or []
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
            ingredients=ingredients,
            vision_confirmation_result=confirmation_result,
        )
        trace.record(
            "output",
            "tool_call_audit",
            "passed" if tool_result.passed else "blocked",
            tool_result.reason or "passed",
        )
        audit_reason = regex_result.reason or tool_result.reason
        blocked = not regex_result.passed or not tool_result.passed
        result = AgentResult(
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
        self._remember(
            user_id=user_id,
            request_text=request_text,
            result=result,
            confirmation_result=confirmation_result,
        )
        return result

    def _remember(
        self,
        *,
        user_id: str,
        request_text: str,
        result: AgentResult,
        confirmation_result: dict[str, object] | None,
    ) -> None:
        self.history.append(
            user_id=user_id,
            request=request_text,
            response=result.content,
            tool_history=result.tool_history,
            confirmation_result=confirmation_result,
        )


def build_pipeline(settings: Settings | None = None) -> KitchenPipeline:
    settings = settings or get_settings()
    profile = read_user_profile(settings.user_profile_path)

    inventory = InventoryRepository(settings.inventory_db_path)
    inventory.initialize()
    action_repository = PendingActionRepository(settings.inventory_db_path)
    action_repository.initialize()
    expiry_inferer = (
        DeepSeekExpiryInferer(settings) if settings.deepseek_api_key else None
    )
    actions = InventoryActionService(
        action_repository,
        inventory,
        ttl_minutes=settings.pending_action_ttl_minutes,
        expiry_inferer=expiry_inferer,
    )
    history = ConversationHistoryRepository(settings.inventory_db_path, max_segments=5)
    history.initialize()
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
    recipes = RecipeRouter(spoonacular, themealdb)
    agent = (
        build_agent(settings, inventory, recipes, actions)
        if settings.deepseek_api_key
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
        settings,
        profile,
        inventory,
        actions,
        history,
        vision_confirmations,
        agent,
        vision,
        food_classifier,
        food_classifier_load_error,
    )
