from typing import Any

import pytest

from app.adapters.recipe_api import SpoonacularClient


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {
            "results": [
                {
                    "id": 1,
                    "title": "Tomato Egg",
                    "instructions": "Cook it.",
                    "extendedIngredients": [
                        {"original": "2 tomatoes"},
                        {"original": "1 egg"},
                    ],
                }
            ]
        }


def test_complex_search_parameter_mapping(monkeypatch: Any) -> None:
    captured: dict[str, object] = {}

    def fake_get(url: str, **kwargs: object) -> FakeResponse:
        captured["url"] = url
        captured["params"] = kwargs["params"]
        return FakeResponse()

    monkeypatch.setattr("app.adapters.recipe_api.httpx.get", fake_get)
    client = SpoonacularClient("https://api.spoonacular.com", "secret")
    results = client.search(
        ingredients=["tomato", "egg"],
        intolerances=["Peanut"],
        exclude_ingredients=["cashew"],
        diet="vegetarian",
        cuisine="Chinese",
        meal_type="main course",
        max_ready_time=30,
        ranking="max-used-ingredients",
        limit=3,
    )

    params = captured["params"]
    assert isinstance(params, dict)
    assert captured["url"] == "https://api.spoonacular.com/recipes/complexSearch"
    assert params["includeIngredients"] == "tomato,egg"
    assert params["intolerances"] == "Peanut"
    assert params["excludeIngredients"] == "cashew"
    assert params["instructionsRequired"] is True
    assert params["addRecipeInformation"] is True
    assert params["addRecipeInstructions"] is True
    assert params["number"] == 3
    assert len(results) == 1


def test_complex_search_requires_ingredients() -> None:
    client = SpoonacularClient("https://api.spoonacular.com", "secret")
    with pytest.raises(ValueError):
        client.search(
            ingredients=[],
            intolerances=[],
            exclude_ingredients=[],
        )
