"""Top-level backend pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from app.adapters.recipe_api import SpoonacularClient, TheMealDbClient
from app.adapters.translation_api import TranslationApiClient
from app.adapters.vision_api import VisionApiClient
from app.agent import KitchenAgent, build_agent
from app.audit import regex_audit
from app.config import Settings, get_settings
from app.context import AgentContext, AgentResult, Ingredient
from app.intent import match_intent
from app.memory import UserProfile, read_user_profile
from app.tools.inventory import InventoryRepository
from app.tools.recipe import RecipeRouter
from app.vision import recognize_ingredients


@dataclass(slots=True)
class KitchenPipeline:
    settings: Settings
    profile: UserProfile
    inventory: InventoryRepository
    agent: KitchenAgent
    vision: VisionApiClient | None

    def process_request(
        self,
        *,
        user_id: str,
        text: str,
        image_bytes: bytes | None = None,
        target_language: str = "zh",
    ) -> AgentResult:
        intent = match_intent(text)
        if intent == "order_unsupported":
            content = "当前原型暂不支持订单或购物车管理。"
            audit = regex_audit(content)
            return AgentResult(
                content=content if audit.passed else "输出未通过安全检查。",
                blocked=not audit.passed,
                audit_reason=audit.reason,
            )

        ingredients: list[Ingredient] = []
        if image_bytes is not None:
            if self.vision is None:
                raise ValueError("Baidu image API credentials are missing from .env")
            ingredients = recognize_ingredients(image_bytes, self.vision)

        context = AgentContext(
            user_id=user_id,
            request_text=text,
            intent=intent,
            ingredients=ingredients,
            allergens=self.profile.allergens,
            profile_markdown=self.profile.markdown,
            history_summary="；".join(self.profile.history),
        )
        return self.agent.run(context, target_language=target_language)


def build_pipeline(settings: Settings | None = None) -> KitchenPipeline:
    settings = settings or get_settings()
    profile = read_user_profile(settings.user_profile_path)

    inventory = InventoryRepository(settings.inventory_db_path)
    inventory.initialize()

    translator: TranslationApiClient | None = None
    if settings.baidu_translate_app_id and settings.baidu_translate_secret_key:
        translator = TranslationApiClient(
            endpoint=settings.baidu_translate_endpoint,
            app_id=settings.baidu_translate_app_id,
            secret_key=settings.baidu_translate_secret_key,
            timeout=settings.http_timeout_seconds,
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
    recipes = RecipeRouter(spoonacular, themealdb, translator)
    agent = build_agent(settings, inventory, recipes)

    vision: VisionApiClient | None = None
    if settings.baidu_image_api_key and settings.baidu_image_secret_key:
        vision = VisionApiClient(
            endpoint=settings.baidu_image_endpoint,
            api_key=settings.baidu_image_api_key,
            secret_key=settings.baidu_image_secret_key,
            timeout=settings.http_timeout_seconds,
        )

    return KitchenPipeline(settings, profile, inventory, agent, vision)
