"""Primary Spoonacular search with TheMealDB fallback."""

from __future__ import annotations

from app.adapters.recipe_api import Recipe, SpoonacularClient, TheMealDbClient


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
    ) -> list[dict[str, object]]:
        allergen_terms = [*intolerances, *exclude_ingredients]
        spoonacular_error: Exception | None = None

        if self.spoonacular is not None:
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
                    return [recipe.as_dict() for recipe in safe_primary]
            except Exception as exc:
                spoonacular_error = exc

        try:
            fallback = self.themealdb.search(ingredients, limit=limit)
            safe_fallback = self._safe_unique(fallback, allergen_terms, limit)
        except Exception as exc:
            if spoonacular_error is not None:
                raise RuntimeError(
                    f"Both recipe routes failed: spoonacular: {spoonacular_error} | "
                    f"themealdb: {exc}"
                ) from exc
            raise RuntimeError(f"TheMealDB fallback failed: {exc}") from exc

        if safe_fallback:
            return [recipe.as_dict() for recipe in safe_fallback]
        if spoonacular_error is not None:
            raise RuntimeError(
                f"Spoonacular failed and TheMealDB returned no safe recipes: "
                f"{spoonacular_error}"
            )
        return []

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
