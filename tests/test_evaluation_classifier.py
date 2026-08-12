from pathlib import Path

from classifier.model import ClassificationResult
from evaluation.classifier import (
    evaluate_route_sample,
    run_classifier_suite,
    select_classifier_test_records,
)
from app.config import get_settings


class CountingClassifier:
    threshold = 0.1

    def __init__(self, *, is_food_query: bool = True) -> None:
        self.calls = 0
        self.is_food_query = is_food_query

    def classify(self, text: str) -> ClassificationResult:
        self.calls += 1
        return ClassificationResult(
            is_related=self.is_food_query,
            positive_score=0.9 if self.is_food_query else 0.01,
            threshold=self.threshold,
        )


def test_classifier_split_detects_missing_training_source() -> None:
    records, provenance_matches = select_classifier_test_records(
        Path("classifier/dataset"), Path("classifier/artifacts/metadata.json")
    )
    assert len(records) == 243
    assert provenance_matches is False


def test_current_router_bypasses_classifier_for_supported_route() -> None:
    classifier = CountingClassifier()
    sample = {
        "id": "list-zh",
        "text": "查看库存",
        "food_related": True,
        "expected_route": "supported",
        "classifier_should_run": False,
    }
    result = evaluate_route_sample(sample, classifier, repetitions=1)
    assert result.metrics["classifier_called"] is False
    assert classifier.calls == 0


def test_challenge_benchmark_has_forty_unique_bilingual_records() -> None:
    result = run_classifier_suite(
        get_settings(), Path("evaluation/benchmarks/router_challenges.json")
    )
    challenge_samples = [
        sample for sample in result.samples if sample.metrics.get("dataset") == "route"
    ]
    assert len(challenge_samples) == 40
    texts = [str(sample.metrics["text"]) for sample in challenge_samples]
    assert len(set(texts)) == 40
    assert any(any("a" <= character.lower() <= "z" for character in text) for text in texts)
    assert any(any("\u4e00" <= character <= "\u9fff" for character in text) for text in texts)
    assert result.summary["classifier_test_total"] == 243
    assert result.summary["classifier_test_metadata_declared_total"] == 245
    assert result.summary["classifier_test_source"] == "current_dataset_replay"
    assert result.summary["classifier_test_replayable"] is False
