from pathlib import Path

import pytest

from recommender.ingredients import (
    ALLERGEN_INGREDIENTS,
    expand_allergen_categories,
    normalize_english_ingredient,
    normalize_search_request,
    parse_ingredient_lines,
)
from recommender.models import IngredientTranslation, RecipeRanking


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2 large tomatoes, chopped", "tomato"),
        ("½ cup unsalted butter, melted", "butter"),
        ("3 garlic cloves (about 1 oz.)", "garlic"),
        ("Potatoes", "potato"),
    ],
)
def test_normalize_english_ingredient(raw: str, expected: str) -> None:
    assert normalize_english_ingredient(raw) == expected


def test_literal_parser_never_executes_code(tmp_path: Path) -> None:
    marker = tmp_path / "recommender-pwned"
    raw = f"__import__('os').system('touch {marker}')"
    assert parse_ingredient_lines(raw) == []
    assert not marker.exists()


def test_mapping_covers_all_chinese_terms_once() -> None:
    request = normalize_search_request(
        ingredients=["西红柿", "egg"],
        ingredient_mapping=[
            IngredientTranslation(original="西红柿", api_term="tomato"),
            IngredientTranslation(original="腰果", api_term="cashew"),
        ],
        allergen_categories=["Peanut"],
        custom_allergens=["腰果"],
        exclude_ingredients=[],
    )
    assert request.ingredients == ("tomato", "egg")
    assert "cashew" in request.blocked_ingredients
    assert "peanut" in request.blocked_ingredients


@pytest.mark.parametrize(
    "mapping",
    [
        [],
        [IngredientTranslation(original="西红柿", api_term="番茄")],
        [
            IngredientTranslation(original="西红柿", api_term="tomato"),
            IngredientTranslation(original="西红柿", api_term="tomato fruit"),
        ],
        [
            IngredientTranslation(original="西红柿", api_term="tomato"),
            IngredientTranslation(original="不存在", api_term="unknown"),
        ],
    ],
)
def test_invalid_mapping_is_rejected(mapping: list[IngredientTranslation]) -> None:
    with pytest.raises(ValueError):
        normalize_search_request(
            ingredients=["西红柿"],
            ingredient_mapping=mapping,
            allergen_categories=[],
            custom_allergens=[],
            exclude_ingredients=[],
        )


def test_optional_fields_are_trimmed_and_empty_values_become_none() -> None:
    request = normalize_search_request(
        ingredients=["  Tomato  "],
        ingredient_mapping=[],
        allergen_categories=[],
        custom_allergens=[],
        exclude_ingredients=[""],
        diet=" Vegetarian ",
        cuisine="",
        meal_type=None,
        max_ready_time=30,
        ranking=RecipeRanking.MAX_USED_INGREDIENTS,
    )

    assert request.ingredients == ("tomato",)
    assert request.exclude_ingredients == frozenset()
    assert request.diet == "Vegetarian"
    assert request.cuisine is None
    assert request.meal_type is None
    assert request.max_ready_time == 30
    assert request.ranking == RecipeRanking.MAX_USED_INGREDIENTS


def test_exclude_and_custom_allergens_are_mapped_to_blocked_set() -> None:
    request = normalize_search_request(
        ingredients=["鸡蛋"],
        ingredient_mapping=[
            IngredientTranslation(original="鸡蛋", api_term="egg"),
            IngredientTranslation(original="腰果", api_term="cashew"),
            IngredientTranslation(original="香菜", api_term="cilantro"),
        ],
        allergen_categories=["Tree Nut"],
        custom_allergens=["腰果"],
        exclude_ingredients=["香菜"],
    )

    assert "cashew" in request.blocked_ingredients
    assert "cilantro" in request.blocked_ingredients
    assert "almond" in request.blocked_ingredients
    assert "egg" not in request.blocked_ingredients


def test_allergen_categories_are_complete_and_expand_representatives() -> None:
    assert set(ALLERGEN_INGREDIENTS) == {
        "Dairy",
        "Egg",
        "Gluten",
        "Grain",
        "Peanut",
        "Seafood",
        "Sesame",
        "Shellfish",
        "Soy",
        "Sulfite",
        "Tree Nut",
        "Wheat",
    }
    expanded = expand_allergen_categories(["Dairy", "Peanut", "Tree Nut"])
    assert {"milk", "butter", "peanut", "groundnut", "almond", "cashew"} <= expanded
