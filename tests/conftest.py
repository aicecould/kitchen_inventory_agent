"""Keep test observations out of application logs."""
import pytest


@pytest.fixture(autouse=True)
def isolate_structured_logs(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_LOG_SOURCE", "test")
    monkeypatch.setenv("LLM_LOG_DB_PATH", str(tmp_path / "llm_logs.db"))
