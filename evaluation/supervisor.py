"""Bounded online measurements for the repair sub-agent path."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from langchain_openai import ChatOpenAI

from app.audit import AuditCode, review_candidate
from app.config import Settings
from app.prompts import REPAIR_SYSTEM_PROMPT
from app.supervisor import RepairAction, RepairDecision, SupervisorRequest
from app.structured_output import invoke_validated_json
from evaluation.benchmarks import load_json_records
from evaluation.models import EvidenceKind, SampleResult, SuiteResult
from evaluation.usage import UsageCollector


def _decision_status(
    decision: RepairDecision, *, final_passed: bool
) -> tuple[str, bool]:
    if decision.action == RepairAction.RETRY_MAIN:
        return "retry_main_requested", False
    if decision.action == RepairAction.FALLBACK:
        return "fallback_requested", False
    return ("passed" if final_passed else "not_repaired"), final_passed


def run_supervisor_suite(
    settings: Settings, path: Path, *, offline: bool
) -> SuiteResult:
    records = load_json_records(
        path,
        required_keys={"id", "content", "intent", "expected_recovery"},
    )
    if offline:
        return SuiteResult(
            name="repair_subagent",
            evidence=EvidenceKind.SKIPPED,
            samples=[
                SampleResult(
                    id=str(record["id"]), status="skipped",
                    evidence=EvidenceKind.SKIPPED,
                    metrics={"reason": "offline_mode"},
                )
                for record in records
            ],
            summary={"sample_count": len(records), "online_skipped": len(records)},
        )

    collector = UsageCollector()
    model = ChatOpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_model,
        temperature=0,
        timeout=settings.http_timeout_seconds,
        max_retries=1,
        max_tokens=settings.deepseek_max_output_tokens,
        extra_body={"thinking": {"type": "disabled"}},
    )
    samples: list[SampleResult] = []
    for record in records:
        target_language = str(record.get("target_language", "zh"))
        initial = review_candidate(
            str(record["content"]), [], intent=str(record["intent"]),
            target_language=target_language,
        )
        started = perf_counter()
        before_calls = collector.call_count
        final_passed = False
        try:
            decision = invoke_validated_json(
                model,
                RepairDecision,
                [
                    {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps({
                        "request": SupervisorRequest(
                            content=str(record["content"]), tool_history=[],
                            intent=str(record["intent"]), target_language=target_language,
                        ).model_dump(mode="json"),
                        "audit": {"code": initial.code.value, "reason": initial.reason},
                    }, ensure_ascii=False)},
                ],
                callbacks=[collector],
            )
            final = (
                review_candidate(
                    decision.content or "", [], intent=str(record["intent"]),
                    target_language=target_language,
                )
                if decision.content is not None
                else initial
            )
            status, final_passed = _decision_status(
                decision,
                final_passed=final.passed,
            )
            error = None
            action = decision.action.value
        except Exception as exc:
            status = "failed"
            error = type(exc).__name__
            action = None
        samples.append(
            SampleResult(
                id=str(record["id"]), status=status,
                evidence=EvidenceKind.MEASURED,
                latency_ms=[(perf_counter() - started) * 1000],
                metrics={
                    "initial_blocked": not initial.passed,
                    "initial_code": initial.code.value,
                    "expected_recovery": record["expected_recovery"],
                    "repair_action": action,
                    "final_passed": final_passed,
                },
                call_counts={"deepseek": collector.call_count - before_calls},
                error=error,
            )
        )
    usage = collector.snapshot()
    return SuiteResult(
        name="repair_subagent", evidence=EvidenceKind.MEASURED, samples=samples,
        summary={
            "sample_count": len(samples),
            "initial_blocks": sum(bool(s.metrics["initial_blocked"]) for s in samples),
            "repair_passes": sum(bool(s.metrics["final_passed"]) for s in samples),
            "retry_main_requests": sum(
                s.status == "retry_main_requested" for s in samples
            ),
            "valid_repair_decisions": sum(
                s.status in {"passed", "retry_main_requested", "fallback_requested"}
                for s in samples
            ),
            "failed_calls": sum(s.status == "failed" for s in samples),
            "deepseek_calls": collector.call_count,
            "input_tokens": usage.input_tokens if usage else None,
            "output_tokens": usage.output_tokens if usage else None,
            "total_tokens": usage.total_tokens if usage else None,
        },
    )
