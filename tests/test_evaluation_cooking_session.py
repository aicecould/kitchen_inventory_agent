from pathlib import Path

from app.config import get_settings
from evaluation.cooking_session import run_cooking_session_suite


CASES = Path("evaluation/benchmarks/cooking_sessions.json")


def test_structured_task_survives_more_than_ten_intervening_turns(tmp_path) -> None:
    result = run_cooking_session_suite(
        get_settings(), CASES, offline=True, temp_dir=tmp_path
    )
    assert result.summary["scenario_count"] == 6
    assert result.summary["structured_fact_retention"] == 1.0
    assert result.summary["history_only_fact_retention"] < 1.0
    assert result.summary["relative_error_reduction"] == 1.0
