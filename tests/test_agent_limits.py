from dataclasses import replace

import pytest

from app.agent import KitchenAgent
from app.config import get_settings
from app.context import AgentContext


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
