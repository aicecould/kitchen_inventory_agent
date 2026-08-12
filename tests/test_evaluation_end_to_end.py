from pathlib import Path

from app.config import get_settings
from evaluation.end_to_end import run_end_to_end_suite


def test_offline_end_to_end_uses_isolated_state(tmp_path) -> None:
    result = run_end_to_end_suite(
        get_settings(),
        Path("evaluation/benchmarks/end_to_end_cases.json"),
        offline=True,
        temp_dir=tmp_path,
    )
    assert result.summary["sample_count"] == 12
    assert result.summary["network_calls"] == 0
    assert result.summary["temporary_db"] != str(get_settings().inventory_db_path)
