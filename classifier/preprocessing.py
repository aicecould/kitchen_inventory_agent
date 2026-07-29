"""Shared text normalization and jieba tokenization."""

from __future__ import annotations

import re
import unicodedata

import jieba

PREPROCESSING_VERSION = 1
_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize user text without removing food quantities or punctuation."""
    normalized = unicodedata.normalize("NFKC", text).strip().lower()
    return _WHITESPACE.sub(" ", normalized)


def tokenize(text: str) -> list[str]:
    """Return non-empty jieba tokens for normalized text."""
    normalized = normalize_text(text)
    if not normalized:
        return []
    return [token.strip() for token in jieba.lcut(normalized) if token.strip()]


def preprocess_text(text: str) -> str:
    """Convert a query to the whitespace-delimited format FastText expects."""
    return " ".join(tokenize(text))
