"""Local-first recipe router with API and web-search fallbacks."""

from __future__ import annotations

import unicodedata
from collections.abc import Callable
from time import perf_counter

from app.adapters.recipe_api import (
    DeepSeekWebSearchClient,
    Recipe,
    SpoonacularClient,
    TheMealDbClient,
)
from recommender.ingredients import expand_allergen_categories, normalize_search_request
from recommender.models import IngredientTranslation, RecipeRanking
from recommender.search import LocalRecipeRecommender


TraceCallback = Callable[[str, str, str, str, int], None]


class RecipeRouter:
    def __init__(
        self,
        local: LocalRecipeRecommender | None,
        spoonacular: SpoonacularClient | None,
        themealdb: TheMealDbClient,
        web_search: DeepSeekWebSearchClient | None = None,
    ) -> None:
        self.local = local
        self.spoonacular = spoonacular
        self.themealdb = themealdb
        self.web_search = web_search

    def search(
        self,
        *,
        ingredients: list[str],
        ingredient_mapping: list[IngredientTranslation],
        allergen_categories: list[str],
        custom_allergens: list[str],
        exclude_ingredients: list[str],
        diet: str | None = None,
        cuisine: str | None = None,
        meal_type: str | None = None,
        max_ready_time: int | None = None,
        ranking: RecipeRanking = RecipeRanking.DEFAULT,
        limit: int = 10,
        minimum_results: int = 3,
        trace: TraceCallback | None = None,
    ) -> list[dict[str, object]]:
        started = perf_counter()
        request = normalize_search_request(
            ingredients=ingredients,
            ingredient_mapping=ingredient_mapping,
            allergen_categories=allergen_categories,
            custom_allergens=custom_allergens,
            exclude_ingredients=exclude_ingredients,
            diet=diet,
            cuisine=cuisine,
            meal_type=meal_type,
            max_ready_time=max_ready_time,
            ranking=ranking,
        )
        merged: list[Recipe] = []
        seen: set[tuple[object, ...]] = set()
        external_ingredients = list(request.ingredients)
        external_intolerances = list(request.allergen_categories)
        blocked_terms = sorted(request.blocked_ingredients)
        official_blocked = expand_allergen_categories(request.allergen_categories)
        external_exclusions = sorted(request.blocked_ingredients - official_blocked)

        if self.local is not None:
            local_started = perf_counter()
            try:
                local_payloads = self.local.search(request, trace=trace)
                local_recipes = [
                    self._recipe_from_local_result(payload)
                    for payload in local_payloads
                ]
                safe_local = self._safe_unique(
                    local_recipes,
                    blocked_terms,
                    seen,
                    limit=limit,
                )
                merged.extend(safe_local)
                self._trace(
                    trace,
                    "local_recipe_search",
                    "local",
                    "success" if safe_local else "empty",
                    f"{len(safe_local)} safe recipe(s)",
                    local_started,
                )
            except Exception as exc:
                self._trace(
                    trace,
                    "local_recipe_search",
                    "local",
                    "failed",
                    f"{type(exc).__name__}; using fallback",
                    local_started,
                )
        else:
            self._trace(
                trace,
                "local_recipe_search",
                "local",
                "skipped",
                "not configured; using fallback",
                started,
            )

        if self.spoonacular is not None and len(merged) < minimum_results:
            spoon_started = perf_counter()
            try:
                primary = self.spoonacular.search(
                    ingredients=external_ingredients,
                    intolerances=external_intolerances,
                    exclude_ingredients=external_exclusions,
                    diet=request.diet,
                    cuisine=request.cuisine,
                    meal_type=request.meal_type,
                    max_ready_time=request.max_ready_time,
                    ranking=ranking.value,
                    limit=limit,
                )
                safe_primary = self._safe_unique(
                    primary,
                    blocked_terms,
                    seen,
                    limit=limit,
                )
                merged.extend(safe_primary)
                self._trace(
                    trace,
                    "external_api",
                    "spoonacular",
                    "success" if safe_primary else "empty",
                    f"{len(safe_primary)} safe recipe(s)",
                    spoon_started,
                )
            except Exception as exc:
                self._trace(
                    trace,
                    "external_api",
                    "spoonacular",
                    "failed",
                    f"{type(exc).__name__}; using fallback",
                    spoon_started,
                )
        else:
            self._trace(
                trace,
                "external_api",
                "spoonacular",
                "skipped",
                "not configured or target already reached",
                started,
            )

        if len(merged) < minimum_results:
            meal_started = perf_counter()
            try:
                fallback = self.themealdb.search(
                    external_ingredients,
                    limit=limit,
                )
                safe_fallback = self._safe_unique(
                    fallback,
                    blocked_terms,
                    seen,
                    limit=limit,
                )
                merged.extend(safe_fallback)
                self._trace(
                    trace,
                    "external_api",
                    "themealdb",
                    "success" if safe_fallback else "empty",
                    f"{len(safe_fallback)} safe recipe(s)",
                    meal_started,
                )
            except Exception as exc:
                self._trace(
                    trace,
                    "external_api",
                    "themealdb",
                    "failed",
                    f"{type(exc).__name__}; using fallback",
                    meal_started,
                )
        else:
            self._trace(
                trace,
                "external_api",
                "themealdb",
                "skipped",
                "target already reached",
                started,
            )

        if self.web_search is not None and len(merged) < minimum_results:
            web_started = perf_counter()
            try:
                web_results = self.web_search.search(
                    ingredients=external_ingredients,
                    intolerances=external_intolerances,
                    exclude_ingredients=external_exclusions,
                    diet=request.diet,
                    cuisine=request.cuisine,
                    meal_type=request.meal_type,
                    max_ready_time=request.max_ready_time,
                    ranking=ranking.value,
                    limit=limit,
                )
                safe_web = self._safe_unique(
                    web_results,
                    blocked_terms,
                    seen,
                    limit=limit,
                )
                merged.extend(safe_web)
                self._trace(
                    trace,
                    "external_api",
                    "deepseek_web_search",
                    "success" if safe_web else "empty",
                    f"{len(safe_web)} safe recipe(s)",
                    web_started,
                )
            except Exception as exc:
                self._trace(
                    trace,
                    "external_api",
                    "deepseek_web_search",
                    "failed",
                    type(exc).__name__,
                    web_started,
                )

        if len(merged) < minimum_results:
            self._trace(
                trace,
                "router",
                "result",
                "degraded",
                f"{len(merged)} real recipe(s) below minimum",
                started,
            )
        return [recipe.as_dict() for recipe in merged[:limit]]

    @staticmethod
    def _trace(
        trace: TraceCallback | None,
        stage: str,
        name: str,
        status: str,
        detail: str,
        started: float,
    ) -> None:
        if trace is not None:
            trace(
                stage,
                name,
                status,
                detail,
                round((perf_counter() - started) * 1000),
            )

    @classmethod
    def _safe_unique(
        cls,
        recipes: list[Recipe],
        allergen_terms: list[str],
        seen: set[tuple[object, ...]],
        *,
        limit: int,
    ) -> list[Recipe]:
        unique: list[Recipe] = []
        for recipe in recipes:
            keys = cls._dedupe_keys(recipe)
            if any(key in seen for key in keys) or cls._contains_allergen(
                recipe, allergen_terms
            ):
                continue
            seen.update(keys)
            unique.append(recipe)
            if len(unique) >= limit:
                break
        return unique

    @staticmethod
    def _dedupe_keys(recipe: Recipe) -> tuple[tuple[object, ...], ...]:
        keys: list[tuple[object, ...]] = []
        if recipe.source_url:
            keys.append(("url", recipe.source_url))
        keys.append(("source_id", recipe.source, recipe.id))
        keys.append(("title", _normalized_title(recipe.title)))
        return tuple(keys)

    @staticmethod
    def _contains_allergen(recipe: Recipe, allergens: list[str]) -> bool:
        haystack = " ".join(
            [recipe.title, recipe.instructions, *recipe.ingredients]
        ).lower()
        return any(
            allergen.strip().lower() in haystack
            for allergen in allergens
            if allergen.strip()
        )

    @classmethod
    def _recipe_from_local_result(cls, payload: dict[str, object]) -> Recipe:
        try:
            recipe_id = str(payload["id"]).strip()
            title = str(payload["title"]).strip()
            instructions = str(payload["instructions"]).strip()
            raw_ingredients = payload["ingredients"]
            source = str(payload["source"]).strip()
            if not recipe_id or not title or not instructions or not source:
                raise ValueError
            if not isinstance(raw_ingredients, list) or not all(
                isinstance(item, str) and item for item in raw_ingredients
            ):
                raise ValueError
            source_url = payload.get("source_url")
            image_url = payload.get("image_url")
            if source_url is not None and not isinstance(source_url, str):
                raise ValueError
            if image_url is not None and not isinstance(image_url, str):
                raise ValueError
            used = _nullable_int(payload.get("used_ingredient_count"))
            missing = _nullable_int(payload.get("missing_ingredient_count"))
            rrf_score = payload.get("rrf_score")
            if rrf_score is not None and (
                not isinstance(rrf_score, (int, float)) or rrf_score <= 0
            ):
                raise ValueError
            fts_rank = _nullable_positive_int(payload.get("fts_rank"))
            semantic_rank = _nullable_positive_int(payload.get("semantic_rank"))
            return Recipe(
                id=recipe_id,
                title=title,
                instructions=instructions,
                ingredients=list(raw_ingredients),
                source=source,
                source_url=source_url,
                image_url=image_url,
                used_ingredient_count=used,
                missing_ingredient_count=missing,
                rrf_score=float(rrf_score) if rrf_score is not None else None,
                fts_rank=fts_rank,
                semantic_rank=semantic_rank,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid local recipe result") from exc


def _normalized_title(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _nullable_int(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or value < 0:
        raise ValueError
    return value


def _nullable_positive_int(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or value <= 0:
        raise ValueError
    return value
