"""Strict data contracts shared by recommender runtime components."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from app.allergens import AllergenIntolerance


class RecipeRanking(StrEnum):
    DEFAULT = "default"
    MAX_USED_INGREDIENTS = "max-used-ingredients"
    MIN_MISSING_INGREDIENTS = "min-missing-ingredients"


class IngredientTranslation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    original: str = Field(min_length=1, max_length=100)
    api_term: str = Field(min_length=1, max_length=100)

    @field_validator("original", "api_term")
    @classmethod
    def strip_value(cls, value: str) -> str:
        return value.strip()


class NormalizedRecipeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ingredients: tuple[str, ...]
    exclude_ingredients: frozenset[str] = frozenset()
    blocked_ingredients: frozenset[str] = frozenset()
    allergen_categories: tuple[AllergenIntolerance, ...] = ()
    diet: str | None = None
    cuisine: str | None = None
    meal_type: str | None = None
    max_ready_time: int | None = Field(default=None, gt=0, le=1440)
    ranking: RecipeRanking = RecipeRanking.DEFAULT


class StoredRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: int = Field(gt=0)
    title: str = Field(min_length=1)
    ingredient_lines: tuple[str, ...]
    normalized_ingredients: frozenset[str]
    ingredient_search_text: str = Field(min_length=1)
    instructions: str = Field(min_length=1)
    image_name: str | None = None


class RrfEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recipe_id: int = Field(gt=0)
    fts_rank: int | None = Field(default=None, gt=0)
    semantic_rank: int | None = Field(default=None, gt=0)
    score: float = Field(gt=0)


class BuildMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    source_commit: Literal["c5a1e8dab69d46841d914be1f20cca6fbe4b6260"]
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sqlite_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    license: Literal["CC BY-SA 3.0"]
    embedding_model: Literal["BAAI/bge-m3"]
    embedding_model_revision: Literal[
        "5617a9f61b028005a4858fdac845db406aefb181"
    ]
    embedding_max_seq_length: Literal[2048]
    embedding_dimension: int = Field(gt=0)
    recipe_count: int = Field(gt=0)
    built_at: AwareDatetime


class RecommenderArtifactsError(RuntimeError):
    """Runtime artifacts are missing, inconsistent, or unreadable."""
