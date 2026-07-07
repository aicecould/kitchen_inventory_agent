"""Allergen categories supported by Spoonacular complexSearch."""

from typing import Literal

AllergenIntolerance = Literal[
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
]

OFFICIAL_INTOLERANCES = (
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
)

OFFICIAL_INTOLERANCE_SET = frozenset(OFFICIAL_INTOLERANCES)
