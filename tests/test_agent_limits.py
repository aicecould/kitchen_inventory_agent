import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.agent import KitchenAgent
from app.config import get_settings
from app.context import AgentContext, HistoryExcerpt
from app.memory import ConversationHistoryRepository
from app.trace import TraceRecorder


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


def test_agent_context_derives_current_date_from_aware_time() -> None:
    message = KitchenAgent._user_message(
        AgentContext(user_id="test", request_text="添加一盒牛奶"),
        "zh",
        current_time=datetime(2026, 8, 10, tzinfo=UTC),
        application_timezone="UTC",
    )
    payload = json.loads(message.split("\n", 1)[1])

    assert payload["current_date"] == "2026-08-10"


def test_agent_prompt_includes_aware_current_time_and_compact_history() -> None:
    message = KitchenAgent._user_message(
        AgentContext(
            user_id="test",
            request_text="回顾昨天",
            history_segments=[HistoryExcerpt(request="问题", response="回答")],
        ),
        "zh",
        current_time=datetime(2026, 8, 10, 9, 30, tzinfo=UTC),
        application_timezone="Asia/Shanghai",
    )
    payload = json.loads(message.split("\n", 1)[1])

    assert payload["current_time"] == "2026-08-10T17:30:00+08:00"
    assert payload["current_date"] == "2026-08-10"
    assert payload["application_timezone"] == "Asia/Shanghai"
    assert payload["recent_history"] == [{"request": "问题", "response": "回答"}]


def test_history_tool_is_user_scoped_and_returns_compact_rows(
    tmp_path: Path,
) -> None:
    history = ConversationHistoryRepository(tmp_path / "history.db")
    history.initialize()
    history.append(
        user_id="u",
        request="昨天聊了什么",
        response="讨论了土豆。",
        created_at=datetime(2026, 8, 9, tzinfo=UTC),
    )
    history.append(
        user_id="other",
        request="秘密",
        response="秘密",
        created_at=datetime(2026, 8, 9, tzinfo=UTC),
    )
    agent = KitchenAgent(
        model=None,  # type: ignore[arg-type]
        inventory=None,  # type: ignore[arg-type]
        recipes=None,  # type: ignore[arg-type]
        actions=None,  # type: ignore[arg-type]
        history=history,
    )
    context = AgentContext(user_id="u", request_text="找昨天的对话")
    history_tool = next(
        item
        for item in agent._create_tools(context, TraceRecorder())
        if item.name == "search_conversation_history_by_time"
    )

    assert "user_id" not in history_tool.args
    result = history_tool.invoke(
        {
            "start_at": "2026-08-09T00:00:00+00:00",
            "end_at": "2026-08-10T00:00:00+00:00",
        }
    )

    assert result == [
        {
            "created_at": "2026-08-09T00:00:00Z",
            "request": "昨天聊了什么",
            "response": "讨论了土豆。",
        }
    ]


@pytest.mark.parametrize(
    "arguments",
    [
        {
            "start_at": "2026-08-09T00:00:00",
            "end_at": "2026-08-10T00:00:00+00:00",
        },
        {
            "start_at": "2026-08-10T00:00:00+00:00",
            "end_at": "2026-08-09T00:00:00+00:00",
        },
        {
            "start_at": "2026-08-09T00:00:00+00:00",
            "end_at": "2026-08-10T00:00:00+00:00",
            "limit": 101,
        },
    ],
)
def test_history_tool_rejects_invalid_time_arguments(
    tmp_path: Path, arguments: dict[str, object]
) -> None:
    history = ConversationHistoryRepository(tmp_path / "history.db")
    history.initialize()
    agent = KitchenAgent(
        model=None,  # type: ignore[arg-type]
        inventory=None,  # type: ignore[arg-type]
        recipes=None,  # type: ignore[arg-type]
        actions=None,  # type: ignore[arg-type]
        history=history,
    )
    context = AgentContext(user_id="u", request_text="找对话")
    history_tool = next(
        item
        for item in agent._create_tools(context, TraceRecorder())
        if item.name == "search_conversation_history_by_time"
    )

    assert history_tool.invoke(arguments) == {
        "error": {
            "code": "INVALID_HISTORY_TIME_RANGE",
            "message": "时间范围必须使用带时区的有效起止时间，且 limit 为 1 到 100。",
        }
    }
