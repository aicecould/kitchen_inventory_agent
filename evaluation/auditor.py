"""Deterministic output-auditor fault-injection benchmark."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.audit import review_candidate
from app.context import Ingredient
from evaluation.benchmarks import load_json_records
from evaluation.models import EvidenceKind, SampleResult, SuiteResult


class AuditCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    content: str
    intent: str = "unknown"
    target_language: str = "zh"
    tool_history: list[dict[str, object]] = Field(default_factory=list)
    ingredients: list[Ingredient] = Field(default_factory=list)
    vision_confirmation_result: dict[str, object] | None = None
    expected_block: bool
    category: str


def load_audit_cases(path: Path) -> list[AuditCase]:
    records = load_json_records(
        path,
        required_keys={"id", "content", "expected_block", "category"},
    )
    return [AuditCase.model_validate(record) for record in records]


def evaluate_audit_case(case: AuditCase) -> SampleResult:
    result = review_candidate(
        case.content,
        case.tool_history,
        intent=case.intent,
        target_language=case.target_language,
        ingredients=case.ingredients,
        vision_confirmation_result=case.vision_confirmation_result,
    )
    blocked = not result.passed
    return SampleResult(
        id=case.id,
        status="blocked" if blocked else "passed",
        evidence=(
            EvidenceKind.FAULT_INJECTION
            if case.expected_block
            else EvidenceKind.MEASURED
        ),
        metrics={
            "expected_block": case.expected_block,
            "actual_block": blocked,
            "category": case.category,
            "audit_code": result.code.value,
            "audit_reason": result.reason,
            "correct": blocked == case.expected_block,
        },
    )


def run_auditor_suite(path: Path) -> SuiteResult:
    samples = [evaluate_audit_case(case) for case in load_audit_cases(path)]
    unsafe = [sample for sample in samples if sample.metrics["expected_block"]]
    legal = [sample for sample in samples if not sample.metrics["expected_block"]]
    category_totals: Counter[str] = Counter()
    category_blocks: Counter[str] = Counter()
    code_hits: Counter[str] = Counter()
    for sample in samples:
        category = str(sample.metrics["category"])
        category_totals[category] += 1
        if sample.metrics["actual_block"]:
            category_blocks[category] += 1
            code_hits[str(sample.metrics["audit_code"])] += 1
    return SuiteResult(
        name="deterministic_auditor",
        evidence=EvidenceKind.FAULT_INJECTION,
        samples=samples,
        summary={
            "unsafe_total": len(unsafe),
            "legal_total": len(legal),
            "unsafe_detected": sum(bool(item.metrics["actual_block"]) for item in unsafe),
            "unsafe_detection_rate": sum(
                bool(item.metrics["actual_block"]) for item in unsafe
            ) / len(unsafe),
            "unsafe_acceptance_rate": sum(
                not bool(item.metrics["actual_block"]) for item in unsafe
            ) / len(unsafe),
            "legal_false_blocks": sum(
                bool(item.metrics["actual_block"]) for item in legal
            ),
            "legal_false_block_rate": sum(
                bool(item.metrics["actual_block"]) for item in legal
            ) / len(legal),
            "category_recall": {
                category: category_blocks[category] / total
                for category, total in sorted(category_totals.items())
                if any(
                    item.metrics["expected_block"]
                    and item.metrics["category"] == category
                    for item in samples
                )
            },
            "audit_code_hits": dict(sorted(code_hits.items())),
        },
    )
