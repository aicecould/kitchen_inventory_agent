"""Deterministic ingredient extraction, normalization, and allergen expansion."""

from __future__ import annotations

import ast
import re
import unicodedata
from collections.abc import Sequence

from app.allergens import OFFICIAL_INTOLERANCE_SET
from recommender.models import (
    IngredientTranslation,
    NormalizedRecipeRequest,
    RecipeRanking,
)


_CJK = re.compile(r"[\u3400-\u9fff]")
_ENGLISH_API_TERM = re.compile(r"^[A-Za-z][A-Za-z0-9 '&()/.+-]*$")
_QUANTITY = re.compile(
    r"^\s*(?:\d+\s*[⁄/]\s*\d+|\d+\s*/\s*\d+|\d+(?:\.\d+)?|"
    r"[½⅓⅔¼¾⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞])"
    r"(?:\s*[-–]\s*(?:\d+\s*[⁄/]\s*\d+|\d+\s*/\s*\d+|\d+(?:\.\d+)?|"
    r"[½⅓⅔¼¾⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞]))?\s*"
)
_SIZES = re.compile(
    r"^(?:extra[- ]large|extra[- ]small|large|small|medium|whole|fresh|canned|"
    r"dried|ground|unsalted|salted|boneless|skinless|peeled|chopped|minced|"
    r"sliced|diced|raw|cooked|plain|seasoned|all[- ]purpose|organic)\s+"
)
_UNITS = re.compile(
    r"\b(?:cups?|tablespoons?|tbsp|teaspoons?|tsp|ounces?|oz|pounds?|lb|lbs|"
    r"grams?|kilograms?|kg|slices?|pieces?|cans?|bunches?|packages?|heads?|"
    r"stalks?|sprigs?|sticks?)\b"
)
_TRAILING_CLOVE = re.compile(r"\bcloves?\s*$")

IRREGULAR_PLURALS = {
    "tomatoes": "tomato",
    "potatoes": "potato",
    "leaves": "leaf",
    "cloves": "garlic",
    "eggs": "egg",
}

ALLERGEN_INGREDIENTS = {
    "Dairy": {
        "milk",
        "butter",
        "cheese",
        "cream",
        "yogurt",
        "whey",
        "casein",
        "ghee",
    },
    "Egg": {"egg", "mayonnaise", "meringue", "albumen"},
    "Gluten": {
        "wheat",
        "barley",
        "rye",
        "spelt",
        "semolina",
        "durum",
        "bulgur",
        "couscous",
    },
    "Grain": {
        "rice",
        "corn",
        "oat",
        "wheat",
        "barley",
        "rye",
        "quinoa",
        "millet",
        "sorghum",
    },
    "Peanut": {"peanut", "groundnut"},
    "Seafood": {
        "fish",
        "salmon",
        "tuna",
        "cod",
        "anchovy",
        "sardine",
        "squid",
        "octopus",
    },
    "Sesame": {"sesame", "tahini"},
    "Shellfish": {"shrimp", "prawn", "crab", "lobster", "crayfish"},
    "Soy": {"soy", "soybean", "tofu", "tempeh", "miso"},
    "Sulfite": {
        "sulfite",
        "sulphite",
        "sulfur dioxide",
        "wine",
        "wine vinegar",
        "dried fruit",
    },
    "Tree Nut": {
        "almond",
        "cashew",
        "walnut",
        "pecan",
        "pistachio",
        "hazelnut",
        "macadamia",
    },
    "Wheat": {
        "wheat",
        "wheat flour",
        "semolina",
        "durum",
        "bulgur",
        "couscous",
    },
}
assert set(ALLERGEN_INGREDIENTS) == OFFICIAL_INTOLERANCE_SET


def parse_ingredient_lines(raw: str) -> list[str]:
    """Safely parse a literal list/tuple into canonical ingredient names."""
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return []
    if not isinstance(parsed, (list, tuple)):
        return []
    cleaned: list[str] = []
    for item in parsed:
        if not isinstance(item, str):
            continue
        normalized = normalize_english_ingredient(item)
        if normalized is not None and normalized not in cleaned:
            cleaned.append(normalized)
    return cleaned


def normalize_english_ingredient(value: str) -> str | None:
    """Return a canonical English ingredient name or None when not usable."""
    normalized = unicodedata.normalize("NFKC", value).lower()
    normalized = " ".join(normalized.split())
    normalized = _QUANTITY.sub("", normalized, count=1)
    normalized = _SIZES.sub("", normalized, count=1)
    normalized = re.sub(r"\s*\([^)]*\)", " ", normalized)
    normalized = normalized.split(",", 1)[0]
    normalized = _UNITS.sub(" ", normalized)
    normalized = _TRAILING_CLOVE.sub("", normalized)
    normalized = " ".join(normalized.split())
    normalized = _SIZES.sub("", normalized, count=1)
    normalized = " ".join(normalized.split())
    if not normalized:
        return None
    normalized = IRREGULAR_PLURALS.get(normalized, normalized)
    if not re.search(r"[a-z]", normalized):
        return None
    return normalized


def expand_allergen_categories(
    categories: Sequence[str],
) -> frozenset[str]:
    expanded: set[str] = set()
    for category in categories:
        if category not in ALLERGEN_INGREDIENTS:
            raise ValueError(f"Unsupported allergen category: {category}")
        expanded.update(ALLERGEN_INGREDIENTS[category])
    return frozenset(expanded)


def normalize_search_request(
    *,
    ingredients: Sequence[str],
    ingredient_mapping: Sequence[IngredientTranslation],
    allergen_categories: Sequence[str],
    custom_allergens: Sequence[str],
    exclude_ingredients: Sequence[str],
    diet: str | None = None,
    cuisine: str | None = None,
    meal_type: str | None = None,
    max_ready_time: int | None = None,
    ranking: RecipeRanking = RecipeRanking.DEFAULT,
) -> NormalizedRecipeRequest:
    cleaned_diet = _optional_text(diet)
    cleaned_cuisine = _optional_text(cuisine)
    cleaned_meal_type = _optional_text(meal_type)
    if max_ready_time is not None and max_ready_time <= 0:
        raise ValueError("max_ready_time must be positive")

    cleaned_ingredients = _unique_terms(ingredients)
    cleaned_exclude = _unique_terms(exclude_ingredients)
    cleaned_custom = _unique_terms(custom_allergens)
    required = {
        _term_key(value)
        for value in [*cleaned_ingredients, *cleaned_exclude, *cleaned_custom]
        if _CJK.search(value)
    }
    mapped_by_key = _validate_mapping(ingredient_mapping, required)

    canonical_ingredients: list[str] = []
    for value in cleaned_ingredients:
        canonical = _canonical_term(value, mapped_by_key)
        if canonical is None:
            raise ValueError(f"Unusable ingredient: {value}")
        if canonical not in canonical_ingredients:
            canonical_ingredients.append(canonical)

    blocked = set(expand_allergen_categories(allergen_categories))
    exclude_set: set[str] = set()
    for value in cleaned_exclude:
        canonical = _canonical_term(value, mapped_by_key)
        if canonical is None:
            raise ValueError(f"Unusable excluded ingredient: {value}")
        exclude_set.add(canonical)
    for value in cleaned_custom:
        canonical = _canonical_term(value, mapped_by_key)
        if canonical is None:
            raise ValueError(f"Unusable custom allergen: {value}")
        blocked.add(canonical)

    return NormalizedRecipeRequest(
        ingredients=tuple(canonical_ingredients),
        exclude_ingredients=frozenset(exclude_set),
        blocked_ingredients=frozenset(blocked | exclude_set),
        allergen_categories=tuple(allergen_categories),
        diet=cleaned_diet,
        cuisine=cleaned_cuisine,
        meal_type=cleaned_meal_type,
        max_ready_time=max_ready_time,
        ranking=ranking,
    )


def _validate_mapping(
    mapping: Sequence[IngredientTranslation],
    required: set[str],
) -> dict[str, str]:
    mapped: dict[str, str] = {}
    for item in mapping:
        original = item.original.strip()
        api_term = item.api_term.strip()
        key = _term_key(original)
        if key in mapped:
            raise ValueError("DUPLICATE_INGREDIENT_MAPPING")
        if _ENGLISH_API_TERM.fullmatch(api_term) is None:
            raise ValueError("INVALID_ENGLISH_API_TERM")
        mapped[key] = api_term
    missing = required - set(mapped)
    if missing:
        raise ValueError("MISSING_INGREDIENT_MAPPING")
    extra = set(mapped) - required
    if extra:
        raise ValueError("UNEXPECTED_INGREDIENT_MAPPING")
    return mapped


def _canonical_term(
    value: str,
    mapping: dict[str, str],
) -> str | None:
    if _CJK.search(value):
        api_term = mapping[_term_key(value)]
        return normalize_english_ingredient(api_term)
    return normalize_english_ingredient(value)


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _unique_terms(values: Sequence[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = value.strip()
        if not cleaned:
            continue
        key = _term_key(cleaned)
        if key in seen:
            continue
        seen.add(key)
        unique.append(cleaned)
    return unique


def _term_key(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()
