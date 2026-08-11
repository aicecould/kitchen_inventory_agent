"""Local-only bge-m3 encoding used by the recommender runtime."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np


class DocumentEncoder(Protocol):
    model_id: str
    model_revision: str
    encode_documents: Callable[[Sequence[str]], np.ndarray]


class BgeM3Encoder:
    model_id = "BAAI/bge-m3"

    def __init__(self, model_path: Path, *, local_files_only: bool) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            str(model_path),
            local_files_only=local_files_only,
        )
        self._model.max_seq_length = 2048
        self.model_revision = (
            self._model.model_card_data.base_model_revision
            if getattr(self._model, "model_card_data", None) is not None
            and getattr(self._model.model_card_data, "base_model_revision", None)
            else "5617a9f61b028005a4858fdac845db406aefb181"
        )

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, batch_size=32)

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode([text], batch_size=1)[0]

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
