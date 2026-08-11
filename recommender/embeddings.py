"""Local-only sentence encoder used by the recommender runtime."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from pydantic import ValidationError

from recommender.model_config import (
    EMBEDDING_MAX_SEQ_LENGTH,
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_REVISION,
)
from recommender.models import ModelManifest, RecommenderArtifactsError


_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
_MANIFEST_FILES = {
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "modules.json",
    "1_Pooling/config.json",
}


class DocumentEncoder(Protocol):
    model_id: str
    model_revision: str
    encode_documents: Callable[[Sequence[str]], np.ndarray]


class LocalSentenceEncoder:
    model_id = EMBEDDING_MODEL

    def __init__(self, model_path: Path, *, local_files_only: bool) -> None:
        from sentence_transformers import SentenceTransformer

        manifest = load_model_manifest(model_path)
        self.model_id = manifest.model_id
        self.model_revision = manifest.revision
        self._model = SentenceTransformer(
            str(model_path),
            local_files_only=local_files_only,
        )
        self._model.max_seq_length = EMBEDDING_MAX_SEQ_LENGTH

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, batch_size=32)

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode(
            [f"{_BGE_QUERY_INSTRUCTION}{text}"],
            batch_size=1,
        )

    def _encode(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        if not isinstance(vectors, np.ndarray) or vectors.ndim != 2:
            raise ValueError("Embedding model must return a 2D NumPy array")
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if vectors.shape[0] == 0 or vectors.shape[1] == 0:
            raise ValueError("Embedding model returned an empty vector batch")
        if not np.isfinite(vectors).all():
            raise ValueError("Embedding model returned NaN or Inf values")
        return vectors


def load_model_manifest(model_path: Path) -> ModelManifest:
    """Load and verify the pinned local model manifest."""
    model_path = Path(model_path)
    manifest_path = model_path / "model-manifest.json"
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = ModelManifest.model_validate(raw)
    except (OSError, ValueError, ValidationError) as exc:
        raise RecommenderArtifactsError(
            f"Local embedding model manifest is missing or invalid: {type(exc).__name__}"
        ) from exc

    if (
        manifest.model_id != EMBEDDING_MODEL
        or manifest.revision != EMBEDDING_MODEL_REVISION
    ):
        raise RecommenderArtifactsError(
            "Local embedding model identity does not match pinned configuration"
        )

    missing = _MANIFEST_FILES - set(manifest.files)
    if missing:
        raise RecommenderArtifactsError(
            "Local embedding model manifest is missing required files"
        )
    for relative in manifest.files:
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise RecommenderArtifactsError(
                "Local embedding model manifest contains an unsafe file path"
            )
        expected = manifest.files[relative]
        try:
            actual = _sha256_file(model_path / relative)
        except OSError as exc:
            raise RecommenderArtifactsError(
                f"Local embedding model file is missing or unreadable: {relative}"
            ) from exc
        if actual != expected:
            raise RecommenderArtifactsError(
                f"Local embedding model file hash mismatch: {relative}"
            )
    return manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
