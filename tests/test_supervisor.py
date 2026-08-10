import pytest
from pydantic import ValidationError

from app.audit import AuditCode, AuditResult
from app.supervisor import (
    RepairAction,
    RepairDecision,
    ResponseSupervisor,
    RetryMode,
    SupervisorCandidate,
    SupervisorRequest,
)


class SequenceAudit:
    def __init__(self, *results: AuditResult) -> None:
        self.results = list(results)
        self.calls = 0

    def __call__(
        self, content: str, tool_history: list[dict[str, object]]
    ) -> AuditResult:
        self.calls += 1
        return self.results.pop(0)


class RepairSpy:
    def __init__(self, decision: RepairDecision) -> None:
        self.decision = decision
        self.calls = 0
        self.seen_code: AuditCode | None = None

    def __call__(
        self, request: SupervisorRequest, audit: AuditResult
    ) -> RepairDecision:
        self.calls += 1
        self.seen_code = audit.code
        return self.decision


class RetrySpy:
    def __init__(self, candidate: SupervisorCandidate) -> None:
        self.candidate = candidate
        self.calls = 0
        self.mode: RetryMode | None = None

    def __call__(
        self,
        mode: RetryMode,
        feedback: str,
        tool_history: list[dict[str, object]],
    ) -> SupervisorCandidate:
        self.calls += 1
        self.mode = mode
        return self.candidate


def forbidden(*args: object, **kwargs: object) -> object:
    raise AssertionError("callback must not be invoked")


def request(content: str = "候选回答") -> SupervisorRequest:
    return SupervisorRequest(
        content=content,
        tool_history=[],
        intent="recipe",
        target_language="zh",
    )


def test_passing_answer_skips_both_recovery_callbacks() -> None:
    supervisor = ResponseSupervisor(
        audit=SequenceAudit(AuditResult(True)),
        repair=forbidden,  # type: ignore[arg-type]
        main_retry=forbidden,  # type: ignore[arg-type]
    )

    outcome = supervisor.review(request("合规回答"))

    assert outcome.content == "合规回答"
    assert outcome.audit_code == AuditCode.PASSED
    assert outcome.steps == [("response_audit", AuditCode.PASSED.value)]


def test_repairable_answer_is_rewritten_once_and_reaudited() -> None:
    audit = SequenceAudit(
        AuditResult(
            False,
            "format violation",
            AuditCode.REPAIRABLE_OUTPUT_VIOLATION,
        ),
        AuditResult(True),
    )
    repair = RepairSpy(
        RepairDecision(
            action=RepairAction.REWRITE,
            content="合规回答",
            reason_code=AuditCode.REPAIRABLE_OUTPUT_VIOLATION,
        )
    )
    supervisor = ResponseSupervisor(
        audit=audit,
        repair=repair,
        main_retry=forbidden,  # type: ignore[arg-type]
    )

    outcome = supervisor.review(request())

    assert outcome.content == "合规回答"
    assert outcome.blocked is False
    assert repair.calls == 1
    assert audit.calls == 2


def test_failed_repair_allows_only_one_main_retry_then_fixed_fallback() -> None:
    audit = SequenceAudit(
        AuditResult(False, "first", AuditCode.REPAIRABLE_OUTPUT_VIOLATION),
        AuditResult(False, "second", AuditCode.REPAIRABLE_OUTPUT_VIOLATION),
        AuditResult(False, "third", AuditCode.REPAIRABLE_OUTPUT_VIOLATION),
    )
    repair = RepairSpy(
        RepairDecision(
            action=RepairAction.REWRITE,
            content="仍不合规",
            reason_code=AuditCode.REPAIRABLE_OUTPUT_VIOLATION,
        )
    )
    retry = RetrySpy(SupervisorCandidate(content="依旧不合规", tool_history=[]))
    supervisor = ResponseSupervisor(audit=audit, repair=repair, main_retry=retry)

    outcome = supervisor.review(request())

    assert outcome.content == "回答未能通过安全检查，库存未发生变化。"
    assert outcome.blocked is True
    assert outcome.audit_code == AuditCode.RETRY_EXHAUSTED
    assert repair.calls == 1
    assert retry.calls == 1
    assert retry.mode == RetryMode.ANSWER_REWRITE
    assert audit.calls == 3


def test_recipe_failure_uses_subagent_feedback_then_recipe_only_retry() -> None:
    audit = SequenceAudit(
        AuditResult(False, "missing tool", AuditCode.RECIPE_TOOL_MISSING),
        AuditResult(True),
    )
    repair = RepairSpy(
        RepairDecision(
            action=RepairAction.RETRY_MAIN,
            feedback="必须先调用 search_recipes。",
            reason_code=AuditCode.RECIPE_TOOL_MISSING,
        )
    )
    recovered_history = [
        {"tool": "search_recipes", "arguments": {}, "result": []}
    ]
    retry = RetrySpy(
        SupervisorCandidate(content="没有找到菜谱。", tool_history=recovered_history)
    )
    supervisor = ResponseSupervisor(audit=audit, repair=repair, main_retry=retry)

    outcome = supervisor.review(request("凭空推荐一道菜谱。"))

    assert outcome.content == "没有找到菜谱。"
    assert outcome.tool_history == recovered_history
    assert repair.seen_code == AuditCode.RECIPE_TOOL_MISSING
    assert retry.mode == RetryMode.RECIPE_RECOVERY
    assert retry.calls == 1


def test_failed_recipe_retry_preserves_real_recovery_tool_history() -> None:
    audit = SequenceAudit(
        AuditResult(False, "missing tool", AuditCode.RECIPE_TOOL_MISSING),
        AuditResult(False, "wrong title", AuditCode.RECIPE_RESULT_MISMATCH),
    )
    repair = RepairSpy(
        RepairDecision(
            action=RepairAction.RETRY_MAIN,
            feedback="必须重新检索。",
            reason_code=AuditCode.RECIPE_TOOL_MISSING,
        )
    )
    recovered_history = [
        {
            "tool": "search_recipes",
            "arguments": {"ingredients": ["tomato"]},
            "result": [],
        }
    ]
    retry = RetrySpy(
        SupervisorCandidate(content="错误菜谱", tool_history=recovered_history)
    )
    supervisor = ResponseSupervisor(audit=audit, repair=repair, main_retry=retry)

    outcome = supervisor.review(request("凭空推荐一道菜谱。"))

    assert outcome.blocked is True
    assert outcome.audit_code == AuditCode.RETRY_EXHAUSTED
    assert outcome.tool_history == recovered_history


def test_user_cancellation_returns_fixed_text_without_model_callbacks() -> None:
    audit = SequenceAudit(
        AuditResult(False, "cancelled", AuditCode.USER_CANCELLED)
    )
    supervisor = ResponseSupervisor(
        audit=audit,
        repair=forbidden,  # type: ignore[arg-type]
        main_retry=forbidden,  # type: ignore[arg-type]
    )

    outcome = supervisor.review(request("取消"))

    assert outcome.content == "已停止处理；没有执行新的库存变更。"
    assert outcome.blocked is False
    assert outcome.audit_code == AuditCode.USER_CANCELLED


def test_impossible_task_returns_fixed_text_without_model_callbacks() -> None:
    audit = SequenceAudit(
        AuditResult(False, "provider unavailable", AuditCode.TASK_IMPOSSIBLE)
    )
    supervisor = ResponseSupervisor(
        audit=audit,
        repair=forbidden,  # type: ignore[arg-type]
        main_retry=forbidden,  # type: ignore[arg-type]
    )

    outcome = supervisor.review(request())

    assert outcome.content == "当前无法完成该请求，库存未发生变化。"
    assert outcome.blocked is False
    assert outcome.audit_code == AuditCode.TASK_IMPOSSIBLE


def test_repair_provider_failure_returns_invalid_output_fallback() -> None:
    def failed_repair(
        request: SupervisorRequest, audit: AuditResult
    ) -> RepairDecision:
        raise RuntimeError("provider unavailable")

    supervisor = ResponseSupervisor(
        audit=SequenceAudit(
            AuditResult(
                False,
                "format violation",
                AuditCode.REPAIRABLE_OUTPUT_VIOLATION,
            )
        ),
        repair=failed_repair,
        main_retry=forbidden,  # type: ignore[arg-type]
    )

    outcome = supervisor.review(request())

    assert outcome.content == "回答未能通过安全检查，库存未发生变化。"
    assert outcome.blocked is True
    assert outcome.audit_code == AuditCode.INVALID_SUPERVISOR_OUTPUT


def test_repair_decision_rejects_incompatible_fields() -> None:
    with pytest.raises(ValidationError):
        RepairDecision(
            action=RepairAction.REWRITE,
            content="重写",
            feedback="不能同时存在",
            reason_code=AuditCode.REPAIRABLE_OUTPUT_VIOLATION,
        )


def test_repair_decision_rejects_action_reason_mismatch() -> None:
    with pytest.raises(ValidationError):
        RepairDecision(
            action=RepairAction.RETRY_MAIN,
            feedback="重试",
            reason_code=AuditCode.PASSED,
        )
