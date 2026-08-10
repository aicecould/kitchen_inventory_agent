import json
from dataclasses import replace
from datetime import date

import pytest

from app.agent import KitchenAgent
from app.config import get_settings
from app.context import AgentContext


class FixedDate(date):
    @classmethod
    def today(cls) -> "FixedDate":
        return cls(2026, 8, 10)


def test_deepseek_limits_have_conservative_defaults() -> None:
    settings = get_settings()

    assert settings.deepseek_max_output_tokens > 0
    assert settings.deepseek_max_input_chars > 0


def test_agent_rejects_oversized_input_before_model_call() -> None:
    agent = KitchenAgent(
        model=None,  # type: ignore[arg-type]
        inventory=None,  # type: ignore[arg-type]
        recipes=None,  # type: ignore[arg-type]
        actions=None,  # type: ignore[arg-type]
        max_input_chars=100,
    )
    context = AgentContext(user_id="test", request_text="番茄" * 100)

    with pytest.raises(ValueError, match="Agent input is too long"):
        agent.run(context)


def test_agent_context_defaults_current_date_to_today(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.agent.date", FixedDate)

    message = KitchenAgent._user_message(
        AgentContext(user_id="test", request_text="添加一盒牛奶"), "zh"
    )
    payload = json.loads(message.split("\n", 1)[1])

    assert payload["current_date"] == "2026-08-10"
