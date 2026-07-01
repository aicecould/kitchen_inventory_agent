"""Dual-route recipe search and cooking guidance."""

from __future__ import annotations

from app.adapters.recipe_api import Recipe, SpoonacularClient, TheMealDbClient
from app.adapters.translation_api import TranslationApiClient


class RecipeRouter:
    def __init__(
        self,
        spoonacular: SpoonacularClient | None,
        themealdb: TheMealDbClient,
        translator: TranslationApiClient | None = None,
    ) -> None:
        self.spoonacular = spoonacular
        self.themealdb = themealdb
        self.translator = translator

    def search(
        self,
        ingredients: list[str],
        preferences: list[str],
        allergens: list[str],
        target_language: str = "zh",
        limit: int = 4,
    ) -> list[dict[str, object]]:
        del preferences  # Reserved for richer Spoonacular query options.
        english_ingredients = self._translate_ingredients(ingredients)
        effective_allergens = [*allergens, *self._translate_allergens(allergens)]
        recipes: list[Recipe] = []
        failures: list[str] = []

        if self.spoonacular is not None:
            try:
                recipes.extend(self.spoonacular.search(english_ingredients, limit=2))
            except Exception as exc:  # One route failing must not stop the other.
                failures.append(f"spoonacular: {exc}")

        try:
            recipes.extend(self.themealdb.search(english_ingredients, limit=2))
        except Exception as exc:
            failures.append(f"themealdb: {exc}")

        unique: list[Recipe] = []
        seen: set[tuple[str, str]] = set()
        for recipe in recipes:
            key = (recipe.source, recipe.id)
            if key not in seen and not self._contains_allergen(recipe, effective_allergens):
                seen.add(key)
                unique.append(recipe)

        output = [self._localize(recipe, target_language).as_dict() for recipe in unique[:limit]]
        if not output and failures:
            raise RuntimeError("Both recipe routes failed: " + " | ".join(failures))
        return output

    def _translate_ingredients(self, ingredients: list[str]) -> list[str]:
        if self.translator is None:
            return ingredients
        return [self.translator.translate(value, target="en") for value in ingredients]

    def _translate_allergens(self, allergens: list[str]) -> list[str]:
        if self.translator is None:
            return []
        return [self.translator.translate(value, target="en") for value in allergens]

    @staticmethod
    def _contains_allergen(recipe: Recipe, allergens: list[str]) -> bool:
        haystack = " ".join([recipe.title, recipe.instructions, *recipe.ingredients]).lower()
        return any(allergen.strip().lower() in haystack for allergen in allergens if allergen.strip())

    def _localize(self, recipe: Recipe, target_language: str) -> Recipe:
        if self.translator is None or target_language in {"en", "auto"}:
            return recipe
        return Recipe(
            id=recipe.id,
            title=self.translator.translate(recipe.title, target=target_language),
            instructions=self.translator.translate(
                recipe.instructions, target=target_language
            ),
            ingredients=[
                self.translator.translate(value, target=target_language)
                for value in recipe.ingredients
            ],
            source=recipe.source,
            source_url=recipe.source_url,
            image_url=recipe.image_url,
        )
