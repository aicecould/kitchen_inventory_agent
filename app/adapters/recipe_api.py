"""Spoonacular and TheMealDB recipe API adapters."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from urllib.parse import urlparse

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


class DeepSeekWebSearchClient:
    """Recipe recall through DeepSeek's Anthropic-compatible server web search."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 20,
    ) -> None:
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        if not root.endswith("/anthropic"):
            root += "/anthropic"
        self.endpoint = f"{root}/v1/messages"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

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
    ) -> list[Recipe]:
        request = {
            "ingredients": ingredients,
            "intolerances": intolerances,
            "exclude_ingredients": exclude_ingredients,
            "diet": diet,
            "cuisine": cuisine,
            "meal_type": meal_type,
            "max_ready_time": max_ready_time,
            "ranking": ranking,
            "limit": limit,
        }
        payload: dict[str, object] = {
            "model": self.model,
            "max_tokens": 2000,
            "system": (
                "Use web search to find real recipe pages matching the request. "
                "Return only one JSON object with a recipes array. Each recipe must "
                "contain title, instructions, ingredients, source_url, and image_url. "
                "Never invent URLs. Exclude every listed intolerance and ingredient."
            ),
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(request, ensure_ascii=False),
                }
            ],
            "tools": [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 1,
                }
            ],
        }
        response_payload = self._post(payload)
        if response_payload.get("stop_reason") == "pause_turn":
            messages = payload["messages"]
            assert isinstance(messages, list)
            messages.append(
                {"role": "assistant", "content": response_payload.get("content", [])}
            )
            response_payload = self._post(payload)
        return self._recipes_from_response(response_payload, limit)

    def _post(self, payload: dict[str, object]) -> dict[str, object]:
        response = httpx.post(
            self.endpoint,
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict):
            raise ValueError("DeepSeek web search returned a non-object response")
        return result

    @classmethod
    def _recipes_from_response(
        cls, response: dict[str, object], limit: int
    ) -> list[Recipe]:
        content = response.get("content")
        if not isinstance(content, list):
            raise ValueError("DeepSeek web search response has no content")
        text = "\n".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
        if text.startswith("```") and text.endswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        decoded = json.loads(text)
        payloads = decoded.get("recipes") if isinstance(decoded, dict) else None
        if not isinstance(payloads, list):
            raise ValueError("DeepSeek web search response has no recipes array")

        recipes: list[Recipe] = []
        for payload in payloads:
            if not isinstance(payload, dict):
                continue
            title = str(payload.get("title") or "").strip()
            instructions = str(payload.get("instructions") or "").strip()
            source_url = cls._safe_url(payload.get("source_url"))
            raw_ingredients = payload.get("ingredients")
            if (
                not title
                or not instructions
                or source_url is None
                or not isinstance(raw_ingredients, list)
            ):
                continue
            ingredients = [
                value.strip()
                for item in raw_ingredients
                if (value := str(item).strip())
            ]
            if not ingredients:
                continue
            recipe_id = hashlib.sha256(
                f"{source_url}\n{title}".encode("utf-8")
            ).hexdigest()[:24]
            recipes.append(
                Recipe(
                    id=f"web_{recipe_id}",
                    title=title,
                    instructions=instructions,
                    ingredients=ingredients,
                    source="deepseek_web_search",
                    source_url=source_url,
                    image_url=cls._safe_url(payload.get("image_url")),
                )
            )
        return recipes[:limit]

    @staticmethod
    def _safe_url(value: object) -> str | None:
        if not isinstance(value, str):
            return None
        candidate = value.strip()
        parsed = urlparse(candidate)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return None
        return candidate


class SpoonacularClient:
    def __init__(self, base_url: str, api_key: str, timeout: float = 20) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

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
    ) -> list[Recipe]:
        if not ingredients:
            raise ValueError("complexSearch requires at least one ingredient")
        params: dict[str, object] = {
            "apiKey": self.api_key,
            "includeIngredients": ",".join(ingredients),
            "instructionsRequired": True,
            "ignorePantry": True,
            "sort": ranking,
            "number": limit,
            "addRecipeInformation": True,
            "addRecipeInstructions": True,
            "addRecipeNutrition": False,
            "fillIngredients": False,
        }
        if intolerances:
            params["intolerances"] = ",".join(intolerances)
        if exclude_ingredients:
            params["excludeIngredients"] = ",".join(exclude_ingredients)
        optional_params = {
            "diet": diet,
            "cuisine": cuisine,
            "type": meal_type,
            "maxReadyTime": max_ready_time,
        }
        params.update(
            {key: value for key, value in optional_params.items() if value is not None}
        )
        response = httpx.get(
            f"{self.base_url}/recipes/complexSearch",
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        recipes: list[Recipe] = []
        for payload in (response.json() or {}).get("results", []):
            recipe_id = str(payload["id"])
            recipes.append(
                Recipe(
                    id=recipe_id,
                    title=str(payload.get("title", "")),
                    instructions=self._instructions(payload),
                    ingredients=[
                        str(value.get("original") or value.get("name", ""))
                        for value in payload.get("extendedIngredients", [])
                    ],
                    source="spoonacular",
                    source_url=payload.get("sourceUrl"),
                    image_url=payload.get("image"),
                )
            )
        return recipes

    @staticmethod
    def _instructions(payload: dict[str, object]) -> str:
        plain = str(payload.get("instructions") or "").strip()
        if plain:
            return plain
        steps: list[str] = []
        for block in payload.get("analyzedInstructions", []) or []:  # type: ignore[union-attr]
            if not isinstance(block, dict):
                continue
            for step in block.get("steps", []) or []:
                if isinstance(step, dict) and step.get("step"):
                    steps.append(str(step["step"]))
        return "\n".join(steps)


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
