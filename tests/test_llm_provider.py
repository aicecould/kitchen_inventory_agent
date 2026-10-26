from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.agent import build_agent
from app.config import get_settings
from app.llm import chat_model_kwargs
from app.llm_log import settings_metadata


def test_provider_env_and_invalid_value(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "qwen")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    settings = get_settings()
    assert settings.llm_enabled
    assert settings.llm_model == "Qwen/Qwen3-1.7B"
    monkeypatch.setenv("LLM_PROVIDER", "unknown")
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        get_settings()


def test_qwen_agent_does_not_need_deepseek_or_send_deepseek_options(monkeypatch):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr("app.agent.ChatOpenAI", create)
    settings = replace(get_settings(), llm_provider="qwen", deepseek_api_key="",
                       llm_logging_enabled=False)
    agent = build_agent(settings, None, None, None, None)
    assert captured["model"] == "Qwen/Qwen3-1.7B"
    assert captured["base_url"] == "http://127.0.0.1:8001/v1"
    assert captured["max_retries"] == 0
    assert "extra_body" not in captured
    assert agent.model_provider == "qwen"
    assert not captured["http_client"]._trust_env
    captured["http_client"].close()


def test_deepseek_remains_default_and_requires_credentials(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    settings = replace(get_settings(), deepseek_api_key="")
    assert settings.llm_provider == "deepseek"
    assert not settings.llm_enabled
    with pytest.raises(ValueError, match="deepseek_api_key"):
        chat_model_kwargs(settings)


def test_qwen_pipeline_enables_agent_without_deepseek_and_disables_paid_search(tmp_path, monkeypatch):
    from app.pipeline import build_pipeline

    def forbidden(**kwargs):
        raise AssertionError("Qwen mode must not enable DeepSeek web search")

    monkeypatch.setattr("app.pipeline.DeepSeekWebSearchClient", forbidden)
    settings = replace(
        get_settings(), llm_provider="qwen", deepseek_api_key="existing-paid-key",
        inventory_db_path=tmp_path / "inventory.db", user_profile_path=tmp_path / "profile.md",
        recommender_enabled=False, food_classifier_enabled=False, llm_logging_enabled=False,
    )
    pipeline = build_pipeline(settings)
    assert pipeline.agent is not None
    assert pipeline.agent.model.model_name == settings.qwen_model
    assert pipeline.agent.actions.expiry_inferer.model.model_name == settings.qwen_model


def test_log_metadata_names_the_selected_provider():
    settings = replace(get_settings(), llm_provider="qwen", qwen_api_key="private-local-key")
    metadata = settings_metadata(settings)
    assert metadata["config"]["llm_provider"] == "qwen"
    assert metadata["config"]["llm_model"] == settings.qwen_model
    assert metadata["provider_endpoint"]["host"] == "127.0.0.1"
    assert "private-local-key" not in str(metadata)


def test_web_status_uses_selected_llm_without_remote_key(monkeypatch):
    from app import web

    settings = replace(get_settings(), llm_provider="qwen", deepseek_api_key="")
    monkeypatch.setattr(web, "get_settings", lambda: settings)
    status = web.service_status()
    assert status["ready"]
    assert status["llm_provider"] == "qwen"
    assert status["services"]["qwen"]
    assert not status["services"]["deepseek"]
