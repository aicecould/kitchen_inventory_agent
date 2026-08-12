"""Classifier and deterministic-router comparison benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Protocol

from app.config import Settings
from app.intent import route_deterministically
from classifier.data import LabeledQuery, load_dataset, stratified_split
from classifier.model import ClassificationResult, FoodQueryClassifier
from evaluation.benchmarks import load_json_records
from evaluation.metrics import confusion_counts, percentile
from evaluation.models import EvidenceKind, SampleResult, SuiteResult


class ClassifierLike(Protocol):
    def classify(self, text: str) -> ClassificationResult: ...


def select_classifier_test_records(
    dataset: Path, metadata: Path
) -> tuple[list[LabeledQuery], bool]:
    details = json.loads(metadata.read_text(encoding="utf-8"))
    records = load_dataset(dataset)
    test = stratified_split(records, seed=int(details["seed"])).test
    expected = int(details["split_counts"]["test"])
    return test, len(test) == expected


def evaluate_route_sample(
    sample: dict[str, object],
    classifier: ClassifierLike,
    *,
    repetitions: int = 5,
) -> SampleResult:
    text = str(sample["text"])
    route = route_deterministically(text)
    expected_route = str(sample["expected_route"])
    classifier_called = route.kind == "undecided"
    classification: ClassificationResult | None = None

    route_deterministically(text)
    latencies: list[float] = []
    for _ in range(repetitions):
        started = perf_counter()
        measured_route = route_deterministically(text)
        if measured_route.kind == "undecided":
            classification = classifier.classify(text)
        latencies.append((perf_counter() - started) * 1000)

    allowed = (
        route.kind in {"supported", "contextual"}
        or (
            route.kind == "undecided"
            and classification is not None
            and classification.is_related
        )
    )
    expected_allowed = bool(sample["food_related"])
    return SampleResult(
        id=str(sample["id"]),
        status="passed" if route.kind == expected_route else "route_mismatch",
        evidence=EvidenceKind.MEASURED,
        latency_ms=latencies,
        metrics={
            "dataset": "route",
            "text": text,
            "expected_route": expected_route,
            "actual_route": route.kind,
            "food_related": expected_allowed,
            "allowed": allowed,
            "classifier_called": classifier_called,
            "classifier_score": (
                classification.positive_score if classification is not None else None
            ),
        },
        call_counts={"classifier": repetitions if classifier_called else 0},
    )


def run_classifier_suite(
    settings: Settings, benchmark_path: Path
) -> SuiteResult:
    classifier = FoodQueryClassifier.load(
        settings.food_classifier_model_path,
        settings.food_classifier_metadata_path,
    )
    test_records, test_replayable = select_classifier_test_records(
        Path("classifier/dataset"), settings.food_classifier_metadata_path
    )
    classifier_samples: list[SampleResult] = []
    for index, record in enumerate(test_records):
        started = perf_counter()
        result = classifier.classify(record.text)
        duration = (perf_counter() - started) * 1000
        classifier_samples.append(
            SampleResult(
                id=f"classifier-test-{index:03d}",
                status="passed" if result.is_related == bool(record.label) else "wrong",
                evidence=EvidenceKind.MEASURED,
                latency_ms=[duration],
                metrics={
                    "dataset": "classifier_test",
                    "food_related": bool(record.label),
                    "allowed": result.is_related,
                    "score": result.positive_score,
                },
                call_counts={"classifier": 1},
            )
        )

    records = load_json_records(
        benchmark_path,
        required_keys={
            "id",
            "text",
            "food_related",
            "expected_route",
            "classifier_should_run",
        },
    )
    route_samples = [evaluate_route_sample(record, classifier) for record in records]
    test_expected = [bool(item.metrics["food_related"]) for item in classifier_samples]
    test_actual = [bool(item.metrics["allowed"]) for item in classifier_samples]
    test_counts = confusion_counts(test_expected, test_actual)
    observed_test_total = len(classifier_samples)
    observed_accuracy = (
        test_counts["tp"] + test_counts["tn"]
    ) / observed_test_total
    observed_recall = test_counts["tp"] / (test_counts["tp"] + test_counts["fn"])
    test_source = "current_dataset_replay"
    metadata = json.loads(
        settings.food_classifier_metadata_path.read_text(encoding="utf-8")
    )
    metadata_test_total = int(metadata["split_counts"]["test"])
    route_expected = [bool(item.metrics["food_related"]) for item in route_samples]
    route_actual = [bool(item.metrics["allowed"]) for item in route_samples]
    route_counts = confusion_counts(route_expected, route_actual)
    all_route_latencies = [
        latency for item in route_samples for latency in item.latency_ms
    ]
    supported = sum(
        item.metrics["actual_route"] in {"supported", "contextual"}
        for item in route_samples
    )
    return SuiteResult(
        name="classifier_router",
        evidence=EvidenceKind.MEASURED,
        samples=[*classifier_samples, *route_samples],
        summary={
            "classifier_test_total": observed_test_total,
            "classifier_test_metadata_declared_total": metadata_test_total,
            "classifier_test_metadata_count_delta": (
                observed_test_total - metadata_test_total
            ),
            "classifier_test_source": test_source,
            "classifier_test_replayable": test_replayable,
            "classifier_test_confusion": test_counts,
            "classifier_test_accuracy": observed_accuracy,
            "classifier_test_food_recall": observed_recall,
            "route_challenge_total": len(route_samples),
            "route_confusion": route_counts,
            "route_food_false_block_rate": route_counts["fn"]
            / (route_counts["tp"] + route_counts["fn"]),
            "route_irrelevant_block_rate": route_counts["tn"]
            / (route_counts["tn"] + route_counts["fp"]),
            "whitelist_hits": supported,
            "whitelist_hit_rate": supported / len(route_samples),
            "route_latency_p50_ms": percentile(all_route_latencies, 0.5),
            "route_latency_p95_ms": percentile(all_route_latencies, 0.95),
        },
    )
