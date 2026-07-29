"""Threshold selection and binary classification evaluation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from classifier.data import LabeledQuery, load_dataset, stratified_split
from classifier.model import FoodQueryClassifier, positive_scores


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    total: int
    true_positive: int
    false_negative: int
    false_positive: int
    true_negative: int
    recall: float
    false_positive_rate: float
    precision: float
    accuracy: float


def calculate_metrics(
    labels: list[int], scores: list[float], threshold: float
) -> BinaryMetrics:
    if len(labels) != len(scores) or not labels:
        raise ValueError("Labels and scores must be non-empty and have equal length")

    true_positive = false_negative = false_positive = true_negative = 0
    for label, score in zip(labels, scores, strict=True):
        predicted = int(score >= threshold)
        if label == 1 and predicted == 1:
            true_positive += 1
        elif label == 1:
            false_negative += 1
        elif predicted == 1:
            false_positive += 1
        else:
            true_negative += 1

    positive_count = true_positive + false_negative
    negative_count = false_positive + true_negative
    predicted_positive = true_positive + false_positive
    return BinaryMetrics(
        total=len(labels),
        true_positive=true_positive,
        false_negative=false_negative,
        false_positive=false_positive,
        true_negative=true_negative,
        recall=true_positive / positive_count if positive_count else 0.0,
        false_positive_rate=(
            false_positive / negative_count if negative_count else 0.0
        ),
        precision=(
            true_positive / predicted_positive if predicted_positive else 0.0
        ),
        accuracy=(true_positive + true_negative) / len(labels),
    )


def choose_threshold(
    labels: list[int],
    scores: list[float],
    *,
    minimum_recall: float,
) -> tuple[float, BinaryMetrics]:
    """Choose the highest observed threshold satisfying the recall constraint."""
    if not 0 < minimum_recall <= 1:
        raise ValueError("minimum_recall must be in (0, 1]")
    if 1 not in labels:
        raise ValueError("Threshold selection requires positive examples")

    candidates = sorted({0.0, *scores}, reverse=True)
    feasible: list[tuple[float, BinaryMetrics]] = []
    for threshold in candidates:
        metrics = calculate_metrics(labels, scores, threshold)
        if metrics.recall >= minimum_recall:
            feasible.append((threshold, metrics))
    if not feasible:
        raise ValueError("No threshold satisfies the requested recall")
    return max(feasible, key=lambda item: item[0])


def evaluate_records(
    model: object,
    records: list[LabeledQuery],
    threshold: float,
) -> BinaryMetrics:
    labels = [record.label for record in records]
    scores = positive_scores(model, [record.text for record in records])
    return calculate_metrics(labels, scores, threshold)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the food query classifier")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("classifier/artifacts/food_query.bin"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("classifier/artifacts/metadata.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classifier = FoodQueryClassifier.load(args.model, args.metadata)
    records = load_dataset(args.dataset)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    split = stratified_split(records, seed=int(metadata["seed"]))
    metrics = evaluate_records(
        classifier.raw_model,
        split.test,
        classifier.threshold,
    )
    print(json.dumps(asdict(metrics), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
