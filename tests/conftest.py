"""Keep test observations out of application logs."""
import pytest


@pytest.fixture(autouse=True)
def isolate_structured_logs(monkeypatch, tmp_path):
    # Unit tests must never select a live local service from the developer's .env.
    # Provider-specific tests explicitly override this value themselves.
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("LLM_LOG_SOURCE", "test")
    monkeypatch.setenv("LLM_LOG_DB_PATH", str(tmp_path / "llm_logs.db"))
