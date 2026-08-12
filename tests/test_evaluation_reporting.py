import json

import pytest

from evaluation.models import EvaluationRun, EvidenceKind, SampleResult, SuiteResult
from evaluation.reporting import render_markdown, write_reports


def sample_run(metadata=None):
    return EvaluationRun(
        metadata=metadata or {"run_id": "fixed", "git_commit": "abc"},
        suites=[SuiteResult(
            name="demo", evidence=EvidenceKind.MEASURED,
            samples=[SampleResult(id="one", status="passed", evidence=EvidenceKind.MEASURED)],
            summary={"success_rate": 0.75},
        )],
        limitations=["small sample"],
    )


def test_json_and_markdown_share_the_same_summary(tmp_path) -> None:
    json_path, md_path = write_reports(sample_run(), tmp_path, "fixed-run")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = md_path.read_text(encoding="utf-8")
    assert str(payload["suites"][0]["summary"]["success_rate"]) in markdown


def test_report_rejects_secret_like_metadata() -> None:
    with pytest.raises(ValueError, match="forbidden metadata key"):
        render_markdown(sample_run({"api_key": "secret"}))
