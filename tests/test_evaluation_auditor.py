from pathlib import Path

from evaluation.auditor import evaluate_audit_case, load_audit_cases, run_auditor_suite


CASES = Path("evaluation/benchmarks/audit_cases.json")


def test_fault_case_is_counted_as_detected() -> None:
    cases = {case.id: case for case in load_audit_cases(CASES)}
    result = evaluate_audit_case(cases["recipe-url-mismatch-01"])
    assert result.status == "blocked"
    assert result.metrics["expected_block"] is True


def test_legal_case_is_not_false_blocked() -> None:
    cases = {case.id: case for case in load_audit_cases(CASES)}
    result = evaluate_audit_case(cases["legal-recipe-01"])
    assert result.status == "passed"


def test_audit_benchmark_has_thirty_faults_and_fifteen_legal_cases() -> None:
    result = run_auditor_suite(CASES)
    assert result.summary["unsafe_total"] == 30
    assert result.summary["legal_total"] == 15
    assert 0 <= result.summary["unsafe_detection_rate"] <= 1
    assert 0 <= result.summary["legal_false_block_rate"] <= 1
