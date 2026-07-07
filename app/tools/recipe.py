"""Primary Spoonacular search with TheMealDB fallback."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

from app.adapters.recipe_api import Recipe, SpoonacularClient, TheMealDbClient

TraceCallback = Callable[[str, str, str, str, int], None]


class RecipeRouter:
    def __init__(
        self,
        spoonacular: SpoonacularClient | None,
        themealdb: TheMealDbClient,
    ) -> None:
        self.spoonacular = spoonacular
        self.themealdb = themealdb

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
                type(exc).__name__,
                started,
            )
            if spoonacular_error is not None:
                raise RuntimeError(
                    f"Both recipe routes failed: spoonacular: {spoonacular_error} | "
                    f"themealdb: {exc}"
                ) from exc
            raise RuntimeError(f"TheMealDB fallback failed: {exc}") from exc

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
        self._trace(
            trace,
            "external_api",
            "themealdb",
            "empty",
            "no safe recipes",
            started,
        )
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
