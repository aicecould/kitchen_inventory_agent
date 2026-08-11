"""Query construction and metadata loading for local recipe search."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from recommender.models import (
    BuildMetadata,
    NormalizedRecipeRequest,
    RecommenderArtifactsError,
)


def build_fts_query(ingredients: Sequence[str]) -> str:
    if not ingredients:
        raise ValueError("FTS query requires at least one ingredient")
    phrases = ['"{}"'.format(value.replace('"', '""')) for value in ingredients]
    return " OR ".join(phrases)


def build_semantic_query(request: NormalizedRecipeRequest) -> str:
    if not request.ingredients:
        raise ValueError("Semantic query requires at least one ingredient")
    parts = ["Find a recipe using: " + ", ".join(request.ingredients)]
    if request.diet:
        parts.append(f"diet: {request.diet}")
    if request.cuisine:
        parts.append(f"cuisine: {request.cuisine}")
    if request.meal_type:
        parts.append(f"meal type: {request.meal_type}")
    if request.max_ready_time is not None:
        parts.append(f"ready within {request.max_ready_time} minutes")
    return "; ".join(parts) + "."


def load_metadata(
    path: Path,
    *,
    expected_embedding_model: str | None = None,
) -> BuildMetadata:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if (
            expected_embedding_model is not None
            and isinstance(raw, dict)
            and raw.get("embedding_model") != expected_embedding_model
        ):
            raise RecommenderArtifactsError("Unsupported embedding model in metadata")
        return BuildMetadata.model_validate(raw)
    except RecommenderArtifactsError:
        raise
    except (OSError, ValueError, ValidationError) as exc:
        raise RecommenderArtifactsError(
            f"Recipe metadata is missing or invalid: {type(exc).__name__}"
        ) from exc
