"""Bounded response recovery after deterministic output audits."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.audit import AuditCode, AuditResult


class RepairAction(StrEnum):
    REWRITE = "rewrite"
    RETRY_MAIN = "retry_main"
    FALLBACK = "fallback"


class RetryMode(StrEnum):
    ANSWER_REWRITE = "answer_rewrite"
    RECIPE_RECOVERY = "recipe_recovery"


class RepairDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: RepairAction
    content: str | None = None
    feedback: str | None = None
    reason_code: AuditCode

    @model_validator(mode="after")
    def validate_fields(self) -> "RepairDecision":
        content = self.content.strip() if self.content is not None else ""
        feedback = self.feedback.strip() if self.feedback is not None else ""
        if self.action == RepairAction.REWRITE:
            if not content or feedback:
                raise ValueError("rewrite requires content and forbids feedback")
            if self.reason_code != AuditCode.REPAIRABLE_OUTPUT_VIOLATION:
                raise ValueError("rewrite requires a repairable output reason")
            self.content = content
        elif self.action == RepairAction.RETRY_MAIN:
            if not feedback or content:
                raise ValueError("retry_main requires feedback and forbids content")
            if self.reason_code not in {
                AuditCode.RECIPE_TOOL_MISSING,
                AuditCode.RECIPE_RESULT_MISMATCH,
            }:
                raise ValueError("retry_main requires a recipe evidence reason")
            self.feedback = feedback
        else:
            if content or feedback:
                raise ValueError("fallback forbids content and feedback")
            if self.reason_code not in {
                AuditCode.USER_CANCELLED,
                AuditCode.TASK_IMPOSSIBLE,
            }:
                raise ValueError("fallback requires cancellation or impossible reason")
        return self


class SupervisorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str
    tool_history: list[dict[str, object]] = Field(default_factory=list)
    intent: str = "unknown"
    target_language: str = "zh"


class SupervisorCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str
    tool_history: list[dict[str, object]] = Field(default_factory=list)


class SupervisorOutcome(BaseModel):
    content: str
    tool_history: list[dict[str, object]] = Field(default_factory=list)
    blocked: bool = False
    audit_code: AuditCode = AuditCode.PASSED
    audit_reason: str | None = None
    steps: list[tuple[str, str]] = Field(default_factory=list)


AuditCallback = Callable[[str, list[dict[str, object]]], AuditResult]
RepairCallback = Callable[[SupervisorRequest, AuditResult], RepairDecision]
MainRetryCallback = Callable[
    [RetryMode, str, list[dict[str, object]]], SupervisorCandidate
]


_FALLBACKS = {
    "zh": {
        AuditCode.USER_CANCELLED: "已停止处理；没有执行新的库存变更。",
        AuditCode.TASK_IMPOSSIBLE: "当前无法完成该请求，库存未发生变化。",
        AuditCode.INVALID_SUPERVISOR_OUTPUT: (
            "回答未能通过安全检查，库存未发生变化。"
        ),
        AuditCode.RETRY_EXHAUSTED: "回答未能通过安全检查，库存未发生变化。",
    },
    "en": {
        AuditCode.USER_CANCELLED: (
            "Processing stopped; no new inventory change was executed."
        ),
        AuditCode.TASK_IMPOSSIBLE: (
            "This request cannot be completed right now; inventory was not changed."
        ),
        AuditCode.INVALID_SUPERVISOR_OUTPUT: (
            "The response did not pass safety review; inventory was not changed."
        ),
        AuditCode.RETRY_EXHAUSTED: (
            "The response did not pass safety review; inventory was not changed."
        ),
    },
}


def fixed_response(code: AuditCode, target_language: str = "zh") -> str:
    language = "en" if target_language.lower().startswith("en") else "zh"
    return _FALLBACKS[language].get(
        code, _FALLBACKS[language][AuditCode.RETRY_EXHAUSTED]
    )


class ResponseSupervisor:
    def __init__(
        self,
        *,
        audit: AuditCallback,
        repair: RepairCallback,
        main_retry: MainRetryCallback,
    ) -> None:
        self.audit = audit
        self.repair = repair
        self.main_retry = main_retry

    def review(self, request: SupervisorRequest) -> SupervisorOutcome:
        history = deepcopy(request.tool_history)
        steps: list[tuple[str, str]] = []
        initial = self.audit(request.content, history)
        steps.append(("response_audit", initial.code.value))
        if initial.passed:
            return SupervisorOutcome(
                content=request.content,
                tool_history=history,
                steps=steps,
            )
        steps.append(("supervisor_classification", initial.code.value))

        if initial.code in {AuditCode.USER_CANCELLED, AuditCode.TASK_IMPOSSIBLE}:
            return self._fallback(request, initial.code, initial.reason, steps, False)

        try:
            decision = self.repair(request.model_copy(deep=True), initial)
        except Exception:
            steps.append(
                ("final_fallback", AuditCode.INVALID_SUPERVISOR_OUTPUT.value)
            )
            return self._fallback(
                request,
                AuditCode.INVALID_SUPERVISOR_OUTPUT,
                "Repair sub-agent failed",
                steps,
                True,
            )

        if initial.code in {
            AuditCode.RECIPE_TOOL_MISSING,
            AuditCode.RECIPE_RESULT_MISMATCH,
        }:
            if decision.action != RepairAction.RETRY_MAIN or decision.feedback is None:
                steps.append(
                    ("final_fallback", AuditCode.INVALID_SUPERVISOR_OUTPUT.value)
                )
                return self._fallback(
                    request,
                    AuditCode.INVALID_SUPERVISOR_OUTPUT,
                    "Recipe failure requires retry_main feedback",
                    steps,
                    True,
                )
            return self._run_main_retry(
                request,
                RetryMode.RECIPE_RECOVERY,
                decision.feedback,
                history,
                steps,
            )

        if decision.action == RepairAction.FALLBACK:
            code = (
                decision.reason_code
                if decision.reason_code
                in {AuditCode.USER_CANCELLED, AuditCode.TASK_IMPOSSIBLE}
                else AuditCode.INVALID_SUPERVISOR_OUTPUT
            )
            steps.append(("final_fallback", code.value))
            return self._fallback(
                request,
                code,
                initial.reason,
                steps,
                code != AuditCode.USER_CANCELLED,
            )

        if decision.action != RepairAction.REWRITE or decision.content is None:
            steps.append(
                ("final_fallback", AuditCode.INVALID_SUPERVISOR_OUTPUT.value)
            )
            return self._fallback(
                request,
                AuditCode.INVALID_SUPERVISOR_OUTPUT,
                "Repairable output requires rewrite content",
                steps,
                True,
            )

        steps.append(("subagent_rewrite", decision.reason_code.value))
        repaired = self.audit(decision.content, history)
        steps.append(("subagent_reaudit", repaired.code.value))
        if repaired.passed:
            return SupervisorOutcome(
                content=decision.content,
                tool_history=history,
                steps=steps,
            )
        return self._run_main_retry(
            request,
            RetryMode.ANSWER_REWRITE,
            repaired.reason or "The repaired response still failed review.",
            history,
            steps,
        )

    def _run_main_retry(
        self,
        request: SupervisorRequest,
        mode: RetryMode,
        feedback: str,
        history: list[dict[str, object]],
        steps: list[tuple[str, str]],
    ) -> SupervisorOutcome:
        steps.append(("main_agent_retry", mode.value))
        try:
            candidate = self.main_retry(mode, feedback, deepcopy(history))
        except Exception:
            steps.append(("final_fallback", AuditCode.RETRY_EXHAUSTED.value))
            return self._fallback(
                request,
                AuditCode.RETRY_EXHAUSTED,
                "Main Agent retry failed",
                steps,
                True,
            )
        candidate_history = (
            history
            if mode == RetryMode.ANSWER_REWRITE
            else deepcopy(candidate.tool_history)
        )
        final_audit = self.audit(candidate.content, candidate_history)
        steps.append(("response_audit", final_audit.code.value))
        if final_audit.passed:
            return SupervisorOutcome(
                content=candidate.content,
                tool_history=candidate_history,
                steps=steps,
            )
        steps.append(("final_fallback", AuditCode.RETRY_EXHAUSTED.value))
        return self._fallback(
            request,
            AuditCode.RETRY_EXHAUSTED,
            final_audit.reason,
            steps,
            True,
            tool_history=candidate_history,
        )

    @staticmethod
    def _fallback(
        request: SupervisorRequest,
        code: AuditCode,
        reason: str | None,
        steps: list[tuple[str, str]],
        blocked: bool,
        *,
        tool_history: list[dict[str, object]] | None = None,
    ) -> SupervisorOutcome:
        return SupervisorOutcome(
            content=fixed_response(code, request.target_language),
            tool_history=deepcopy(
                request.tool_history if tool_history is None else tool_history
            ),
            blocked=blocked,
            audit_code=code,
            audit_reason=reason,
            steps=steps,
        )
