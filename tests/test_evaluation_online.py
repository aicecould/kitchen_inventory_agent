from app.config import get_settings
from evaluation.online import run_gate_cost_probe


def test_offline_gate_cost_probe_marks_samples_skipped() -> None:
    result = run_gate_cost_probe(
        get_settings(), ["今天有什么科技新闻？", "Explain binary search"], offline=True
    )
    assert len(result.samples) == 2
    assert result.summary["online_skipped"] == 2
    assert all(sample.status == "skipped" for sample in result.samples)
