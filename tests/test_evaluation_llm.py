from dataclasses import replace

from app.config import get_settings
from evaluation import run as runner
from evaluation.models import EvidenceKind, SuiteResult


def test_llm_suite_does_not_run_other_model_or_business_suites(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("LLM suite must not run classifiers, embeddings or business fixtures")

    for name in ("run_classifier_suite", "run_retrieval_suite",
                 "run_cooking_session_suite", "run_end_to_end_suite"):
        monkeypatch.setattr(runner, name, forbidden)
    for name in ("run_auditor_suite", "run_gate_cost_probe", "run_supervisor_suite"):
        monkeypatch.setattr(runner, name, lambda *args, **kwargs: SuiteResult(
            name="checked", evidence=EvidenceKind.MEASURED))
    monkeypatch.setattr(runner, "get_settings", lambda: replace(
        get_settings(), llm_provider="qwen", deepseek_api_key=""))
    report = runner.run_quick(offline=False, suite="llm")
    assert len(report.suites) == 3
    assert report.metadata["llm_provider"] == "qwen"
    assert report.metadata["llm_model"] == "Qwen/Qwen3-1.7B"
    assert report.metadata["suite"] == "llm"
    assert any("9" in item for item in report.limitations)
