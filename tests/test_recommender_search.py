import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from recommender.models import (
    NormalizedRecipeRequest,
    RecipeRanking,
    RecommenderArtifactsError,
    RrfEntry,
    StoredRecipe,
)
from recommender.search import (
    LocalRecipeRecommender,
    load_recommender,
    reciprocal_rank_fusion,
)


def test_rrf_uses_one_based_ranks_and_k_60() -> None:
    fused = reciprocal_rank_fusion([1, 2, 3], [2, 3, 4])
    assert fused[1].score == pytest.approx(1 / 61)
    assert fused[2].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[3].score == pytest.approx(1 / 63 + 1 / 62)
    assert fused[4].score == pytest.approx(1 / 63)


class FakeRepository:
    def __init__(self, fts_ids: list[int], recipes: Sequence[StoredRecipe]) -> None:
        self.fts_ids = fts_ids
        self.recipes = {recipe.id: recipe for recipe in recipes}
        self.fts_limits: list[int] = []

    def fts_search(self, match_query: str, limit: int = 100) -> list[int]:
        self.fts_limits.append(limit)
        return list(self.fts_ids)

    def get_many(self, recipe_ids: Sequence[int]) -> dict[int, StoredRecipe]:
        return {
            recipe_id: self.recipes[recipe_id]
            for recipe_id in recipe_ids
            if recipe_id in self.recipes
        }


class FakeEncoder:
    def __init__(self) -> None:
        self.query_texts: list[str] = []

    def encode_query(self, text: str) -> np.ndarray:
        self.query_texts.append(text)
        return np.zeros((1, 8), dtype=np.float32)


class FakeFaissIndex:
    def __init__(self, semantic_ids: list[int]) -> None:
        self.semantic_ids = semantic_ids
        self.limits: list[int] = []

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        self.limits.append(k)
        ids = (self.semantic_ids + [-1] * k)[:k]
        return (
            np.zeros((1, k), dtype=np.float32),
            np.array([ids], dtype=np.int64),
        )


def stored_recipe(
    recipe_id: int,
    ingredients: set[str],
    *,
    title: str | None = None,
) -> StoredRecipe:
    name = title or f"Recipe {recipe_id}"
    return StoredRecipe(
        id=recipe_id,
        title=name,
        ingredient_lines=tuple(sorted(ingredients)),
        normalized_ingredients=frozenset(ingredients),
        ingredient_search_text=" ".join(sorted(ingredients)),
        instructions=f"Cook {name}.",
        image_name=None,
    )


def recommender(
    repository: FakeRepository,
    encoder: FakeEncoder,
    index: FakeFaissIndex,
    *,
    rrf_fuser: Any = None,
) -> LocalRecipeRecommender:
    return LocalRecipeRecommender(
        repository,
        index,
        encoder,
        rrf_fuser=rrf_fuser,
    )


def test_search_uses_top_100_and_applies_blocked_filter(tmp_path: Path) -> None:
    repository = FakeRepository(
        [1, 2, 3],
        [
            stored_recipe(1, {"tomato", "egg", "salt"}),
            stored_recipe(2, {"peanut", "noodles"}),
            stored_recipe(3, {"cashew", "rice"}),
            stored_recipe(4, {"tomato", "salt"}),
        ],
    )
    encoder = FakeEncoder()
    index = FakeFaissIndex([1, 2, 4])
    request = NormalizedRecipeRequest(
        ingredients=("tomato", "egg"),
        exclude_ingredients=frozenset({"peanut", "cashew"}),
        blocked_ingredients=frozenset({"peanut", "cashew"}),
        diet="vegan",
        max_ready_time=15,
    )

    results = recommender(repository, encoder, index).search(request)

    assert repository.fts_limits == [100]
    assert index.limits == [100]
    assert [item["id"] for item in results] == ["local-1", "local-4"]
    assert results[0]["used_ingredient_count"] == 2
    assert results[0]["missing_ingredient_count"] == 1
    assert results[0]["source"] == "local_recipe_dataset"
    assert results[0]["rrf_score"] > 0


def _sort_fixture() -> tuple[
    FakeRepository,
    FakeEncoder,
    FakeFaissIndex,
    Any,
]:
    repository = FakeRepository(
        [1, 2, 3, 4],
        [
            stored_recipe(1, {"a", "b", "c", "x", "y"}),
            stored_recipe(2, {"a", "b", "x"}),
            stored_recipe(3, {"a", "b", "c", "d", "e", "f", "g"}),
            stored_recipe(4, {"a"}),
        ],
    )
    encoder = FakeEncoder()
    index = FakeFaissIndex([1, 2, 3, 4])
    scores = {1: 0.30, 2: 0.40, 3: 0.10, 4: 0.20}

    def fuser(fts_ids: list[int], semantic_ids: list[int]) -> dict[int, RrfEntry]:
        return {
            recipe_id: RrfEntry(recipe_id=recipe_id, score=score)
            for recipe_id, score in scores.items()
        }

    return repository, encoder, index, fuser


@pytest.mark.parametrize(
    ("ranking", "expected"),
    [
        (RecipeRanking.DEFAULT, [2, 1, 4, 3]),
        (RecipeRanking.MAX_USED_INGREDIENTS, [1, 3, 2, 4]),
        (RecipeRanking.MIN_MISSING_INGREDIENTS, [4, 2, 1, 3]),
    ],
)
def test_search_uses_stable_ranking(ranking: RecipeRanking, expected: list[int]) -> None:
    repository, encoder, index, fuser = _sort_fixture()
    request = NormalizedRecipeRequest(
        ingredients=("a", "b", "c"),
        blocked_ingredients=frozenset(),
        ranking=ranking,
    )

    results = recommender(
        repository,
        encoder,
        index,
        rrf_fuser=fuser,
    ).search(request)

    assert [int(str(item["id"]).split("-")[1]) for item in results] == expected


def test_search_returns_at_most_ten_results() -> None:
    recipes = [
        stored_recipe(recipe_id, {f"ingredient-{recipe_id}"})
        for recipe_id in range(1, 13)
    ]
    repository = FakeRepository(list(range(1, 13)), recipes)
    encoder = FakeEncoder()
    index = FakeFaissIndex(list(range(1, 13)))
    request = NormalizedRecipeRequest(
        ingredients=tuple(f"ingredient-{recipe_id}" for recipe_id in range(1, 13)),
        blocked_ingredients=frozenset(),
    )

    results = recommender(repository, encoder, index).search(request)

    assert len(results) == 10


def test_load_recommender_rejects_model_revision_mismatch(tmp_path: Path) -> None:
    (tmp_path / "recipes.db").write_bytes(b"db")
    (tmp_path / "recipes.faiss").write_bytes(b"faiss")
    metadata = {
        "schema_version": 1,
        "source_commit": "c5a1e8dab69d46841d914be1f20cca6fbe4b6260",
        "source_sha256": "a" * 64,
        "sqlite_sha256": "b" * 64,
        "license": "CC BY-SA 3.0",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_model_revision": "wrong-revision",
        "embedding_max_seq_length": 512,
        "embedding_dimension": 384,
        "recipe_count": 1,
        "built_at": "2026-08-11T00:00:00Z",
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(RecommenderArtifactsError):
        load_recommender(
            db_path=tmp_path / "recipes.db",
            faiss_path=tmp_path / "recipes.faiss",
            metadata_path=tmp_path / "metadata.json",
            model_path=tmp_path / "missing-model",
            expected_model_revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        )


def test_load_recommender_rejects_embedding_model_mismatch(tmp_path: Path) -> None:
    (tmp_path / "recipes.db").write_bytes(b"db")
    (tmp_path / "recipes.faiss").write_bytes(b"faiss")
    model_path = tmp_path / "model"
    model_path.mkdir()
    metadata = {
        "schema_version": 1,
        "source_commit": "c5a1e8dab69d46841d914be1f20cca6fbe4b6260",
        "source_sha256": "a" * 64,
        "sqlite_sha256": "b" * 64,
        "license": "CC BY-SA 3.0",
        "embedding_model": "BAAI/bge-m3",
        "embedding_model_revision": "5617a9f61b028005a4858fdac845db406aefb181",
        "embedding_max_seq_length": 2048,
        "embedding_dimension": 1024,
        "recipe_count": 1,
        "built_at": "2026-08-11T00:00:00Z",
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(RecommenderArtifactsError, match="Unsupported embedding model"):
        load_recommender(
            db_path=tmp_path / "recipes.db",
            faiss_path=tmp_path / "recipes.faiss",
            metadata_path=tmp_path / "metadata.json",
            model_path=model_path,
            expected_model_revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        )
