"""Hybrid local recipe retrieval: FTS + FAISS + RRF."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter

import faiss
import numpy as np

from recommender.embeddings import LocalSentenceEncoder
from recommender.ingredients import contains_blocked_ingredient
from recommender.model_config import EMBEDDING_MODEL
from recommender.models import (
    NormalizedRecipeRequest,
    RecipeRanking,
    RecommenderArtifactsError,
    RrfEntry,
    StoredRecipe,
)
from recommender.query import build_fts_query, build_semantic_query, load_metadata
from recommender.repository import RecipeRepository


TraceCallback = Callable[[str, str, str, str, int], None]


class LocalRecipeSearchError(RuntimeError):
    """The local hybrid search could not produce a complete RRF result."""


def reciprocal_rank_fusion(
    fts_ids: Sequence[int],
    semantic_ids: Sequence[int],
    k: int = 60,
) -> dict[int, RrfEntry]:
    fused: dict[int, RrfEntry] = {}
    for rank, recipe_id in enumerate(fts_ids, start=1):
        contribution = 1.0 / (k + rank)
        if recipe_id in fused:
            fused[recipe_id].score += contribution
            fused[recipe_id].fts_rank = rank
        else:
            fused[recipe_id] = RrfEntry(
                recipe_id=recipe_id,
                fts_rank=rank,
                score=contribution,
            )
    for rank, recipe_id in enumerate(semantic_ids, start=1):
        contribution = 1.0 / (k + rank)
        if recipe_id in fused:
            fused[recipe_id].score += contribution
            fused[recipe_id].semantic_rank = rank
        else:
            fused[recipe_id] = RrfEntry(
                recipe_id=recipe_id,
                semantic_rank=rank,
                score=contribution,
            )
    return fused


class LocalRecipeRecommender:
    def __init__(
        self,
        repository: RecipeRepository,
        faiss_index: faiss.IndexIDMap2,
        encoder: LocalSentenceEncoder,
        *,
        rrf_fuser: Callable[
            [Sequence[int], Sequence[int]],
            dict[int, RrfEntry],
        ] = reciprocal_rank_fusion,
    ) -> None:
        self.repository = repository
        self.faiss_index = faiss_index
        self.encoder = encoder
        self._rrf_fuser = rrf_fuser or reciprocal_rank_fusion

    def search(
        self,
        request: NormalizedRecipeRequest,
        trace: TraceCallback | None = None,
    ) -> list[dict[str, object]]:
        started = perf_counter()
        try:
            fts_ids = self.repository.fts_search(
                build_fts_query(request.ingredients),
                limit=100,
            )
            semantic_text = build_semantic_query(request)
            query_vector = self.encoder.encode_query(semantic_text)
            _, semantic_vectors = self.faiss_index.search(query_vector, 100)
            semantic_ids = [
                int(recipe_id)
                for recipe_id in semantic_vectors[0]
                if int(recipe_id) >= 0
            ]
        except Exception as exc:
            self._trace(
                trace,
                "local_recipe_search",
                "hybrid",
                "failed",
                type(exc).__name__,
                started,
            )
            raise LocalRecipeSearchError(
                "Local hybrid recipe search failed"
            ) from exc

        if len(set(fts_ids)) != len(fts_ids) or len(set(semantic_ids)) != len(
            semantic_ids
        ):
            raise LocalRecipeSearchError("Local recall returned duplicate recipe IDs")

        self._trace(
            trace,
            "local_recipe_search",
            "fts5_recall",
            "success",
            f"{len(fts_ids)} candidate(s)",
            started,
        )
        self._trace(
            trace,
            "local_recipe_search",
            "faiss_recall",
            "success",
            f"{len(semantic_ids)} candidate(s)",
            started,
        )
        fused = self._rrf_fuser(fts_ids, semantic_ids)
        self._trace(
            trace,
            "local_recipe_search",
            "rrf_fusion",
            "success",
            f"{len(fused)} candidate(s)",
            started,
        )
        stored = self.repository.get_many(list(fused))
        missing_ids = set(fused) - set(stored)
        if missing_ids:
            raise LocalRecipeSearchError("Local recipes referenced by recall are missing")

        ranked: list[dict[str, object]] = []
        for recipe_id, entry in fused.items():
            recipe = stored[recipe_id]
            if contains_blocked_ingredient(
                recipe.normalized_ingredients,
                request.blocked_ingredients,
            ):
                continue
            available = set(request.ingredients)
            used = len(available & recipe.normalized_ingredients)
            missing = len(recipe.normalized_ingredients - available)
            ranked.append(
                {
                    "id": f"local-{recipe.id}",
                    "title": recipe.title,
                    "instructions": recipe.instructions,
                    "ingredients": list(recipe.ingredient_lines),
                    "source": "local_recipe_dataset",
                    "source_url": None,
                    "image_url": None,
                    "used_ingredient_count": used,
                    "missing_ingredient_count": missing,
                    "rrf_score": entry.score,
                    "fts_rank": entry.fts_rank,
                    "semantic_rank": entry.semantic_rank,
                }
            )
        self._trace(
            trace,
            "local_recipe_search",
            "recipe_filter",
            "success",
            f"{len(ranked)} safe candidate(s)",
            started,
        )
        ranked.sort(key=lambda item: _sort_key(item, request.ranking))
        self._trace(
            trace,
            "local_recipe_search",
            "recipe_ranking",
            "success",
            f"{len(ranked[:10])} recipe(s)",
            started,
        )
        return ranked[:10]

    @staticmethod
    def _trace(
        trace: TraceCallback | None,
        stage: str,
        name: str,
        status: str,
        detail: str,
        started: float,
    ) -> None:
        if trace is not None:
            trace(
                stage,
                name,
                status,
                detail,
                round((perf_counter() - started) * 1000),
            )


def load_recommender(
    db_path: Path,
    faiss_path: Path,
    metadata_path: Path,
    model_path: Path,
    expected_model_revision: str,
) -> LocalRecipeRecommender:
    try:
        return _load_recommender_checked(
            db_path=db_path,
            faiss_path=faiss_path,
            metadata_path=metadata_path,
            model_path=model_path,
            expected_model_revision=expected_model_revision,
        )
    except RecommenderArtifactsError:
        raise
    except Exception as exc:
        raise RecommenderArtifactsError(
            "Recommender startup validation failed"
        ) from exc


def _load_recommender_checked(
    *,
    db_path: Path,
    faiss_path: Path,
    metadata_path: Path,
    model_path: Path,
    expected_model_revision: str,
) -> LocalRecipeRecommender:
    if not db_path.is_file():
        raise RecommenderArtifactsError("Recommender SQLite artifact is missing")
    if not faiss_path.is_file():
        raise RecommenderArtifactsError("Recommender FAISS artifact is missing")
    if not metadata_path.is_file():
        raise RecommenderArtifactsError("Recommender metadata artifact is missing")
    if not model_path.is_dir():
        raise RecommenderArtifactsError("Recommender model directory is missing")
    metadata = load_metadata(metadata_path, expected_embedding_model=EMBEDDING_MODEL)
    if metadata.embedding_model != EMBEDDING_MODEL:
        raise RecommenderArtifactsError("Unsupported embedding model in metadata")
    if metadata.embedding_model_revision != expected_model_revision:
        raise RecommenderArtifactsError("Embedding model revision mismatch")
    if _sha256_file(db_path) != metadata.sqlite_sha256:
        raise RecommenderArtifactsError("SQLite SHA-256 does not match metadata")
    repository = RecipeRepository.open_readonly(db_path)
    if repository.count_recipes() != metadata.recipe_count:
        raise RecommenderArtifactsError("SQLite recipe count does not match metadata")
    index = faiss.read_index(str(faiss_path))
    if not isinstance(index, faiss.IndexIDMap2):
        raise RecommenderArtifactsError("FAISS index is not an IndexIDMap2")
    if index.d != metadata.embedding_dimension or index.ntotal != metadata.recipe_count:
        raise RecommenderArtifactsError("FAISS index does not match metadata")
    try:
        encoder = LocalSentenceEncoder(model_path, local_files_only=True)
    except Exception as exc:
        raise RecommenderArtifactsError(
            f"Local embedding model failed to load: {type(exc).__name__}"
        ) from exc
    if encoder.model_id != metadata.embedding_model:
        raise RecommenderArtifactsError("Embedding model ID does not match metadata")
    if encoder.model_revision != metadata.embedding_model_revision:
        raise RecommenderArtifactsError(
            "Embedding model revision does not match metadata"
        )
    probe = encoder.encode_query("dimension probe")
    if (
        not isinstance(probe, np.ndarray)
        or probe.ndim != 2
        or probe.shape[0] != 1
        or probe.shape[1] != index.d
    ):
        raise RecommenderArtifactsError(
            "Embedding query dimension does not match FAISS index"
        )
    if not np.isfinite(probe).all():
        raise RecommenderArtifactsError(
            "Embedding query probe contains NaN or Inf values"
        )
    return LocalRecipeRecommender(repository, index, encoder)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sort_key(
    item: dict[str, object],
    ranking: RecipeRanking,
) -> tuple[object, ...]:
    recipe_id = int(str(item["id"]).removeprefix("local-"))
    rrf_score = float(item["rrf_score"])
    used = int(item["used_ingredient_count"])
    missing = int(item["missing_ingredient_count"])
    if ranking == RecipeRanking.MAX_USED_INGREDIENTS:
        return (-used, missing, -rrf_score, recipe_id)
    if ranking == RecipeRanking.MIN_MISSING_INGREDIENTS:
        return (missing, -used, -rrf_score, recipe_id)
    return (-rrf_score, recipe_id)
