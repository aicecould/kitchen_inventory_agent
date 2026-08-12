from types import SimpleNamespace

from app.agent import build_agent
from app.config import get_settings
from evaluation.usage import UsageCollector


def test_usage_collector_aggregates_provider_usage() -> None:
    collector = UsageCollector()
    collector.on_llm_end(
        SimpleNamespace(llm_output={"token_usage": {
            "prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14
        }}, generations=[])
    )
    collector.on_llm_end(
        SimpleNamespace(llm_output={"token_usage": {
            "prompt_tokens": 6, "completion_tokens": 2, "total_tokens": 8
        }}, generations=[])
    )
    assert collector.snapshot().model_dump() == {
        "input_tokens": 16, "output_tokens": 6, "total_tokens": 22
    }
    assert collector.call_count == 2


def test_usage_collector_returns_none_when_provider_omits_usage() -> None:
    collector = UsageCollector()
    collector.on_llm_end(SimpleNamespace(llm_output={}, generations=[]))
    assert collector.snapshot() is None
    assert collector.call_count == 1


def test_build_agent_accepts_optional_model_callbacks() -> None:
    collector = UsageCollector()
    agent = build_agent(
        get_settings(),
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        model_callbacks=[collector],
    )
    assert agent.model_callbacks == [collector]


def test_build_agent_explicitly_disables_deepseek_thinking(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeModel:
        pass

    def build_model(**kwargs):
        captured.update(kwargs)
        return FakeModel()

    monkeypatch.setattr("app.agent.ChatOpenAI", build_model)

    build_agent(
        get_settings(),
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
    )

    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}
