from pathlib import Path
import json

from app.config import get_settings
from evaluation.models import EvidenceKind
from evaluation.supervisor import run_supervisor_suite


CASES = Path("evaluation/benchmarks/supervisor_cases.json")


def test_offline_supervisor_suite_marks_online_cases_skipped() -> None:
    result = run_supervisor_suite(get_settings(), CASES, offline=True)
    assert len(result.samples) == 6
    assert all(sample.status == "skipped" for sample in result.samples)
    assert all(sample.evidence == EvidenceKind.SKIPPED for sample in result.samples)


def test_valid_retry_main_decision_is_not_counted_as_provider_failure() -> None:
    from app.audit import AuditCode
    from app.supervisor import RepairAction, RepairDecision
    from evaluation.supervisor import _decision_status

    status, final_passed = _decision_status(
        RepairDecision(
            action=RepairAction.RETRY_MAIN,
            feedback="重新调用菜谱工具。",
            reason_code=AuditCode.RECIPE_TOOL_MISSING,
        ),
        final_passed=False,
    )
    assert status == "retry_main_requested"
    assert final_passed is False


def test_provider_failure_never_inherits_previous_pass_state(
    tmp_path, monkeypatch
) -> None:
    from evaluation import supervisor as module

    class FailingModel:
        def with_structured_output(self, schema, *, method):
            return self

        def invoke(self, messages, config=None):
            raise TimeoutError("offline")

    monkeypatch.setattr(module, "ChatOpenAI", lambda **kwargs: FailingModel())
    path = tmp_path / "case.json"
    path.write_text(json.dumps([{
        "id": "failure", "content": "已完成付款", "intent": "unknown",
        "expected_recovery": "rewrite"
    }]), encoding="utf-8")

    result = module.run_supervisor_suite(get_settings(), path, offline=False)

    assert result.samples[0].status == "failed"
    assert result.samples[0].metrics["final_passed"] is False
    assert result.summary["repair_passes"] == 0


def test_online_supervisor_explicitly_disables_deepseek_thinking(
    tmp_path, monkeypatch
) -> None:
    from evaluation import supervisor as module

    captured: dict[str, object] = {}

    class FailingModel:
        def with_structured_output(self, schema, *, method):
            return self

        def invoke(self, messages, config=None):
            raise TimeoutError("offline")

    def build_model(**kwargs):
        captured.update(kwargs)
        return FailingModel()

    monkeypatch.setattr(module, "ChatOpenAI", build_model)
    path = tmp_path / "case.json"
    path.write_text(json.dumps([{
        "id": "thinking", "content": "已完成付款", "intent": "unknown",
        "expected_recovery": "rewrite"
    }]), encoding="utf-8")

    module.run_supervisor_suite(get_settings(), path, offline=False)

    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}
