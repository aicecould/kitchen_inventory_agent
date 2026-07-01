"""Spoonacular and TheMealDB recipe API adapters."""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx


@dataclass(slots=True)
class Recipe:
    id: str
    title: str
    instructions: str
    ingredients: list[str] = field(default_factory=list)
    source: str = ""
    source_url: str | None = None
    image_url: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "instructions": self.instructions,
            "ingredients": self.ingredients,
            "source": self.source,
            "source_url": self.source_url,
            "image_url": self.image_url,
        }


class SpoonacularClient:
    def __init__(self, base_url: str, api_key: str, timeout: float = 20) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def search(self, ingredients: list[str], limit: int = 3) -> list[Recipe]:
        response = httpx.get(
            f"{self.base_url}/recipes/findByIngredients",
            params={
                "apiKey": self.api_key,
                "ingredients": ",".join(ingredients),
                "number": limit,
                "ranking": 1,
                "ignorePantry": True,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        recipes: list[Recipe] = []
        for item in response.json() or []:
            recipe_id = str(item["id"])
            detail = httpx.get(
                f"{self.base_url}/recipes/{recipe_id}/information",
                params={"apiKey": self.api_key, "includeNutrition": False},
                timeout=self.timeout,
            )
            detail.raise_for_status()
            payload = detail.json()
            recipes.append(
                Recipe(
                    id=recipe_id,
                    title=str(payload.get("title", item.get("title", ""))),
                    instructions=str(
                        payload.get("instructions")
                        or payload.get("summary")
                        or ""
                    ),
                    ingredients=[
                        str(value.get("original") or value.get("name", ""))
                        for value in payload.get("extendedIngredients", [])
                    ],
                    source="spoonacular",
                    source_url=payload.get("sourceUrl"),
                    image_url=payload.get("image") or item.get("image"),
                )
            )
        return recipes


class TheMealDbClient:
    def __init__(self, base_url: str, api_key: str = "1", timeout: float = 20) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    @property
    def api_root(self) -> str:
        return f"{self.base_url}/{self.api_key}"

    def search(self, ingredients: list[str], limit: int = 3) -> list[Recipe]:
        if not ingredients:
            return []
        response = httpx.get(
            f"{self.api_root}/filter.php",
            params={"i": ingredients[0].replace(" ", "_")},
            timeout=self.timeout,
        )
        response.raise_for_status()
        candidates = (response.json() or {}).get("meals") or []
        recipes: list[Recipe] = []
        for item in candidates[:limit]:
            detail = httpx.get(
                f"{self.api_root}/lookup.php",
                params={"i": item["idMeal"]},
                timeout=self.timeout,
            )
            detail.raise_for_status()
            meals = (detail.json() or {}).get("meals") or []
            if not meals:
                continue
            meal = meals[0]
            ingredient_lines = []
            for index in range(1, 21):
                name = str(meal.get(f"strIngredient{index}") or "").strip()
                measure = str(meal.get(f"strMeasure{index}") or "").strip()
                if name:
                    ingredient_lines.append(f"{measure} {name}".strip())
            recipes.append(
                Recipe(
                    id=str(meal["idMeal"]),
                    title=str(meal.get("strMeal", "")),
                    instructions=str(meal.get("strInstructions", "")),
                    ingredients=ingredient_lines,
                    source="themealdb",
                    source_url=meal.get("strSource"),
                    image_url=meal.get("strMealThumb"),
                )
            )
        return recipes
