"""Primary Spoonacular search with TheMealDB fallback."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

from app.adapters.recipe_api import (
    DeepSeekWebSearchClient,
    Recipe,
    SpoonacularClient,
    TheMealDbClient,
)

TraceCallback = Callable[[str, str, str, str, int], None]


class RecipeRouter:
    def __init__(
        self,
        spoonacular: SpoonacularClient | None,
        themealdb: TheMealDbClient,
        web_search: DeepSeekWebSearchClient | None = None,
    ) -> None:
        self.spoonacular = spoonacular
        self.themealdb = themealdb
        self.web_search = web_search

    def search(
        self,
        *,
        ingredients: list[str],
        intolerances: list[str],
        exclude_ingredients: list[str],
        diet: str | None = None,
        cuisine: str | None = None,
        meal_type: str | None = None,
        max_ready_time: int | None = None,
        ranking: str = "max-used-ingredients",
        limit: int = 3,
        trace: TraceCallback | None = None,
    ) -> list[dict[str, object]]:
        allergen_terms = [*intolerances, *exclude_ingredients]
        spoonacular_error: Exception | None = None

        if self.spoonacular is not None:
            started = perf_counter()
            try:
                primary = self.spoonacular.search(
                    ingredients=ingredients,
                    intolerances=intolerances,
                    exclude_ingredients=exclude_ingredients,
                    diet=diet,
                    cuisine=cuisine,
                    meal_type=meal_type,
                    max_ready_time=max_ready_time,
                    ranking=ranking,
                    limit=limit,
                )
                safe_primary = self._safe_unique(primary, allergen_terms, limit)
                if safe_primary:
                    self._trace(
                        trace,
                        "external_api",
                        "spoonacular",
                        "success",
                        f"{len(safe_primary)} safe recipe(s)",
                        started,
                    )
                    return [recipe.as_dict() for recipe in safe_primary]
                self._trace(
                    trace,
                    "external_api",
                    "spoonacular",
                    "empty",
                    "no safe recipes; using fallback",
                    started,
                )
            except Exception as exc:
                spoonacular_error = exc
                self._trace(
                    trace,
                    "external_api",
                    "spoonacular",
                    "failed",
                    f"{type(exc).__name__}; using fallback",
                    started,
                )
        else:
            if trace is not None:
                trace(
                    "external_api",
                    "spoonacular",
                    "skipped",
                    "not configured; using fallback",
                    0,
                )

        started = perf_counter()
        try:
            fallback = self.themealdb.search(ingredients, limit=limit)
            safe_fallback = self._safe_unique(fallback, allergen_terms, limit)
        except Exception as exc:
            self._trace(
                trace,
                "external_api",
                "themealdb",
                "failed",
                f"{type(exc).__name__}; using fallback",
                started,
            )
            themealdb_error: Exception | None = exc
            safe_fallback = []
        else:
            themealdb_error = None

        if safe_fallback:
            self._trace(
                trace,
                "external_api",
                "themealdb",
                "success",
                f"{len(safe_fallback)} safe recipe(s)",
                started,
            )
            return [recipe.as_dict() for recipe in safe_fallback]
        if themealdb_error is None:
            self._trace(
                trace,
                "external_api",
                "themealdb",
                "empty",
                (
                    "no safe recipes; using fallback"
                    if self.web_search
                    else "no safe recipes"
                ),
                started,
            )

        if self.web_search is not None:
            started = perf_counter()
            try:
                web_results = self.web_search.search(
                    ingredients=ingredients,
                    intolerances=intolerances,
                    exclude_ingredients=exclude_ingredients,
                    diet=diet,
                    cuisine=cuisine,
                    meal_type=meal_type,
                    max_ready_time=max_ready_time,
                    ranking=ranking,
                    limit=limit,
                )
                safe_web_results = self._safe_unique(
                    web_results, allergen_terms, limit
                )
            except Exception as exc:
                self._trace(
                    trace,
                    "external_api",
                    "deepseek_web_search",
                    "failed",
                    type(exc).__name__,
                    started,
                )
                raise RuntimeError("DeepSeek web search fallback failed") from exc
            if safe_web_results:
                self._trace(
                    trace,
                    "external_api",
                    "deepseek_web_search",
                    "success",
                    f"{len(safe_web_results)} safe recipe(s)",
                    started,
                )
                return [recipe.as_dict() for recipe in safe_web_results]
            self._trace(
                trace,
                "external_api",
                "deepseek_web_search",
                "empty",
                "no safe recipes",
                started,
            )

        if themealdb_error is not None:
            if spoonacular_error is not None:
                raise RuntimeError("Both recipe API routes failed") from themealdb_error
            raise RuntimeError("TheMealDB fallback failed") from themealdb_error
        if spoonacular_error is not None:
            raise RuntimeError(
                f"Spoonacular failed and TheMealDB returned no safe recipes: "
                f"{spoonacular_error}"
            )
        return []

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
        cls, recipes: list[Recipe], allergen_terms: list[str], limit: int
    ) -> list[Recipe]:
        unique: list[Recipe] = []
        seen: set[tuple[str, str]] = set()
        for recipe in recipes:
            key = (recipe.source, recipe.id)
            if key in seen or cls._contains_allergen(recipe, allergen_terms):
                continue
            seen.add(key)
            unique.append(recipe)
        return unique[:limit]

    @staticmethod
    def _contains_allergen(recipe: Recipe, allergens: list[str]) -> bool:
        haystack = " ".join([recipe.title, recipe.instructions, *recipe.ingredients]).lower()
        return any(allergen.strip().lower() in haystack for allergen in allergens if allergen.strip())
