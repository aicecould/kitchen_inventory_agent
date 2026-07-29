"""Runtime wrapper around a trained FastText binary model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from classifier.preprocessing import PREPROCESSING_VERSION, preprocess_text

POSITIVE_LABEL = "__label__1"


class PredictModel(Protocol):
    def predict(
        self, text: str | list[str], *, k: int = ...
    ) -> tuple[object, object]: ...


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    is_related: bool
    positive_score: float
    threshold: float


class FoodQueryClassifier:
    """Binary recipe/ingredient domain classifier loaded once at startup."""

    def __init__(self, model: PredictModel, *, threshold: float) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("Classifier threshold must be between 0 and 1")
        self.raw_model = model
        self.threshold = threshold

    @classmethod
    def load(cls, model_path: Path, metadata_path: Path) -> FoodQueryClassifier:
        if not model_path.is_file():
            raise FileNotFoundError(f"Classifier model not found: {model_path}")
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Classifier metadata not found: {metadata_path}"
            )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        version = int(metadata.get("preprocessing_version", -1))
        if version != PREPROCESSING_VERSION:
            raise ValueError(
                "Classifier preprocessing version does not match runtime"
            )

        import fasttext

        model = fasttext.load_model(str(model_path))
        return cls(model, threshold=float(metadata["threshold"]))

    def classify(self, text: str) -> ClassificationResult:
        score = positive_score(self.raw_model, text)
        return ClassificationResult(
            is_related=score >= self.threshold,
            positive_score=score,
            threshold=self.threshold,
        )


def positive_score(model: PredictModel, text: str) -> float:
    processed = preprocess_text(text)
    if not processed:
        return 0.0
    labels, probabilities = model.predict(processed, k=2)
    label_values = _flatten_prediction(labels)
    probability_values = _flatten_prediction(probabilities)
    for label, probability in zip(label_values, probability_values, strict=True):
        if str(label) == POSITIVE_LABEL:
            return float(probability)
    return 0.0


def positive_scores(model: PredictModel, texts: list[str]) -> list[float]:
    return [positive_score(model, text) for text in texts]


def _flatten_prediction(values: object) -> list[object]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, (list, tuple)):
        return [values]
    if len(values) == 1 and isinstance(values[0], (list, tuple)):
        return list(values[0])
    return list(values)
