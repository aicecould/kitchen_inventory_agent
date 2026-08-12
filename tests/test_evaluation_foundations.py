import json
from pathlib import Path

import pytest

from evaluation.benchmarks import load_json_records
from evaluation.budget import CallBudget
from evaluation.metrics import (
    confusion_counts,
    mrr_at_k,
    percentile,
    recall_at_k,
    relative_reduction,
)
from evaluation.models import EvidenceKind, SampleResult, TokenUsage


def test_metrics_handle_ranks_percentiles_and_zero_baseline() -> None:
    assert recall_at_k([1, 4, None], 3) == pytest.approx(1 / 3)
    assert mrr_at_k([1, 2, None], 10) == pytest.approx(0.5)
    assert percentile([10, 20, 30, 40], 0.95) == 40
    assert relative_reduction(0, 0) is None


def test_confusion_counts_use_positive_true_semantics() -> None:
    assert confusion_counts(
        [True, True, False, False], [True, False, True, False]
    ) == {"tp": 1, "fn": 1, "fp": 1, "tn": 1}


def test_budget_never_allows_more_than_configured_limit() -> None:
    budget = CallBudget({"spoonacular": 2})
    assert budget.consume("spoonacular").allowed is True
    assert budget.consume("spoonacular").allowed is True
    exhausted = budget.consume("spoonacular")
    assert exhausted.allowed is False
    assert exhausted.reason == "budget_exhausted"
    assert budget.used("spoonacular") == 2


def test_benchmark_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "samples.json"
    path.write_text('[{"id":"same"},{"id":"same"}]', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate benchmark id"):
        load_json_records(path, required_keys={"id"})


def test_benchmark_loader_rejects_missing_keys(tmp_path: Path) -> None:
    path = tmp_path / "samples.json"
    path.write_text(json.dumps([{"id": "one"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required key: text"):
        load_json_records(path, required_keys={"id", "text"})


def test_result_models_reject_negative_usage_and_counts() -> None:
    with pytest.raises(ValueError):
        TokenUsage(input_tokens=-1, output_tokens=0, total_tokens=0)
    with pytest.raises(ValueError):
        SampleResult(
            id="sample",
            status="passed",
            evidence=EvidenceKind.MEASURED,
            call_counts={"deepseek": -1},
        )
