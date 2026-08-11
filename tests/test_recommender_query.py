import csv
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from recommender.build import build_sqlite
from recommender.models import BuildMetadata, NormalizedRecipeRequest
from recommender.query import (
    build_fts_query,
    build_semantic_query,
    load_metadata,
)
from recommender.repository import RecipeRepository


def write_recipe_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "",
                "Title",
                "Ingredients",
                "Instructions",
                "Image_Name",
                "Cleaned_Ingredients",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "": "0",
                "Title": "Tomato Egg",
                "Ingredients": "['2 tomatoes', '2 eggs']",
                "Instructions": "Cook tomato and egg.",
                "Image_Name": "tomato-egg",
                "Cleaned_Ingredients": "['2 tomatoes', '2 eggs']",
            }
        )
        writer.writerow(
            {
                "": "1",
                "Title": "Peanut Noodles",
                "Ingredients": "['noodles', 'peanuts']",
                "Instructions": "Mix noodles and peanuts.",
                "Image_Name": "peanut-noodles",
                "Cleaned_Ingredients": "['noodles', 'peanuts']",
            }
        )


def test_fts_query_quotes_each_phrase_and_escapes_quotes() -> None:
    assert build_fts_query(["tomato", 'egg" OR *']) == (
        '"tomato" OR "egg"" OR *"'
    )


def test_semantic_query_includes_only_positive_constraints() -> None:
    request = NormalizedRecipeRequest(
        ingredients=("tomato", "egg"),
        exclude_ingredients=frozenset({"cashew"}),
        blocked_ingredients=frozenset({"peanut", "cashew"}),
        diet="vegetarian",
        cuisine="Chinese",
        meal_type="dinner",
        max_ready_time=30,
    )
    text = build_semantic_query(request)
    assert text == (
        "Find a recipe using: tomato, egg; diet: vegetarian; "
        "cuisine: Chinese; meal type: dinner; ready within 30 minutes."
    )
    assert "peanut" not in text
    assert "cashew" not in text


def test_repository_reads_fts_and_bulk_records(tmp_path: Path) -> None:
    source = tmp_path / "recipes.csv"
    database = tmp_path / "recipes.db"
    write_recipe_csv(source)
    build_sqlite(source, database)

    repository = RecipeRepository.open_readonly(database)
    assert repository.fts_search('"tomato"') == [1]
    assert repository.fts_search('"peanut"') == [2]
    recipes = repository.get_many([1, 2])
    assert list(recipes) == [1, 2]
    assert recipes[1].title == "Tomato Egg"
    assert "tomato" in recipes[1].normalized_ingredients
    assert "egg" in recipes[1].normalized_ingredients
    assert recipes[1].image_name == "tomato-egg"


def test_repository_readonly_connection_rejects_delete(tmp_path: Path) -> None:
    source = tmp_path / "recipes.csv"
    database = tmp_path / "recipes.db"
    write_recipe_csv(source)
    build_sqlite(source, database)
    repository = RecipeRepository.open_readonly(database)

    with pytest.raises(sqlite3.OperationalError):
        repository._connection.execute("DELETE FROM recipes")


def _metadata_dict(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "schema_version": 1,
        "source_commit": "c5a1e8dab69d46841d914be1f20cca6fbe4b6260",
        "source_sha256": "a" * 64,
        "sqlite_sha256": "b" * 64,
        "license": "CC BY-SA 3.0",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_model_revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "embedding_max_seq_length": 512,
        "embedding_dimension": 384,
        "recipe_count": 2,
        "built_at": "2026-08-11T00:00:00Z",
    }
    values.update(overrides)
    return values


def test_load_metadata_accepts_valid_metadata(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(_metadata_dict()), encoding="utf-8")

    metadata = load_metadata(path)

    assert isinstance(metadata, BuildMetadata)
    assert metadata.embedding_dimension == 384
    assert metadata.recipe_count == 2
    assert metadata.built_at.tzinfo is not None


def test_load_metadata_rejects_malformed_metadata(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    path.write_text('{"schema_version": "invalid"}', encoding="utf-8")

    from recommender.models import RecommenderArtifactsError

    with pytest.raises(RecommenderArtifactsError):
        load_metadata(path)
