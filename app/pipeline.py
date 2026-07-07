"""Top-level backend pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from app.actions import InventoryActionService, PendingActionRepository
from app.adapters.recipe_api import SpoonacularClient, TheMealDbClient
from app.adapters.vision_api import VisionApiClient
from app.agent import KitchenAgent, build_agent
from app.audit import regex_audit
from app.config import Settings, get_settings
from app.context import AgentContext, AgentResult, Ingredient
from app.intent import (
    match_intent,
    match_simple_inventory_operation,
    match_simple_inventory_query,
)
from app.memory import UserProfile, read_user_profile
from app.tools.inventory import InventoryRepository
from app.tools.recipe import RecipeRouter
from app.trace import TraceRecorder
from app.vision import recognize_ingredients


@dataclass(slots=True)
class KitchenPipeline:
    settings: Settings
    profile: UserProfile
    inventory: InventoryRepository
    actions: InventoryActionService
    agent: KitchenAgent | None
    vision: VisionApiClient | None

    def process_request(
        self,
        *,
        user_id: str,
        text: str,
        image_bytes: bytes | None = None,
        target_language: str = "zh",
    ) -> AgentResult:
        trace = TraceRecorder()
        intent = match_intent(text)
        trace.record("routing", "intent_match", "success", intent)
        if intent == "order_unsupported":
            content = "当前原型暂不支持订单或购物车管理。"
            audit = regex_audit(content)
            trace.record(
                "output", "regex_audit", "passed" if audit.passed else "blocked", audit.reason or "passed"
            )
            return AgentResult(
                content=content if audit.passed else "输出未通过安全检查。",
                blocked=not audit.passed,
                audit_reason=audit.reason,
                execution_trace=trace.events,
            )

        if match_simple_inventory_query(text):
            trace.record("routing", "inventory_query_regex", "matched", "inventory.list")
            items = trace.run(
                "database",
                "list_inventory",
                self.inventory.list_items,
                lambda values: f"{len(values)} item(s)",
            )
            if items:
                lines = [
                    f"- {item['name']}：{float(item['quantity']):g} {item['unit']}"
                    for item in items
                ]
                content = "当前库存：\n" + "\n".join(lines)
            else:
                content = "当前库存为空。"
            audit = regex_audit(content)
            trace.record(
                "output", "regex_audit", "passed" if audit.passed else "blocked", audit.reason or "passed"
            )
            return AgentResult(
                content=content if audit.passed else "输出未通过安全检查。",
                blocked=not audit.passed,
                audit_reason=audit.reason,
                tool_history=[{"tool": "list_inventory", "result": items}],
                execution_trace=trace.events,
            )

        simple_operation = match_simple_inventory_operation(text)
        if simple_operation is not None:
            trace.record(
                "routing", "inventory_write_regex", "matched", simple_operation.operation
            )
            action = trace.run(
                "database",
                "create_pending_action",
                lambda: self.actions.propose(
                    user_id, simple_operation.operation, simple_operation.arguments
                ),
                lambda value: f"{value.operation}; status={value.status}",
            )
            return AgentResult(
                content=f"已生成待确认操作：{action.summary}。确认前库存不会改变。",
                tool_history=[
                    {
                        "tool": f"propose_{simple_operation.operation.replace('.', '_')}",
                        "result": action.model_dump(mode="json"),
                    }
                ],
                execution_trace=trace.events,
            )

        if self.agent is None:
            raise ValueError("Missing configuration in .env: deepseek_api_key")

        ingredients: list[Ingredient] = []
        if image_bytes is not None:
            if self.vision is None:
                raise ValueError("Baidu image API credentials are missing from .env")
            ingredients = trace.run(
                "external_api",
                "baidu_vision",
                lambda: recognize_ingredients(image_bytes, self.vision),
                lambda values: f"{len(values)} detection(s)",
            )

        context = AgentContext(
            user_id=user_id,
            request_text=text,
            intent=intent,
            ingredients=ingredients,
            allergens=self.profile.allergens,
            allergen_intolerances=self.profile.allergen_intolerances,
            custom_allergens=self.profile.custom_allergens,
            profile_markdown=self.profile.markdown,
            history_summary="；".join(self.profile.history),
        )
        result = self.agent.run(context, target_language=target_language)
        result.execution_trace = [*trace.events, *result.execution_trace]
        return result


def build_pipeline(settings: Settings | None = None) -> KitchenPipeline:
    settings = settings or get_settings()
    profile = read_user_profile(settings.user_profile_path)

    inventory = InventoryRepository(settings.inventory_db_path)
    inventory.initialize()
    action_repository = PendingActionRepository(settings.inventory_db_path)
    action_repository.initialize()
    actions = InventoryActionService(
        action_repository,
        inventory,
        ttl_minutes=settings.pending_action_ttl_minutes,
    )

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

    return KitchenPipeline(settings, profile, inventory, actions, agent, vision)
