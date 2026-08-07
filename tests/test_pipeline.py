from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from classifier.model import ClassificationResult
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from pydantic import Field

from app.adapters.recipe_api import TheMealDbClient
from app.agent import KitchenAgent
from app.config import get_settings
from app.context import AgentContext, AgentResult
from app.pipeline import build_pipeline
from app.tools.recipe import RecipeRouter


class ForbiddenClassifier:
    calls = 0

    def classify(self, text: str) -> ClassificationResult:
        self.calls += 1
        raise AssertionError(f"classifier must not receive deterministic route: {text}")


class IrrelevantClassifierSpy:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def classify(self, text: str) -> ClassificationResult:
        self.calls.append(text)
        return ClassificationResult(
            is_related=False, positive_score=0.01, threshold=0.5
        )


class RelatedClassifierSpy:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def classify(self, text: str) -> ClassificationResult:
        self.calls.append(text)
        return ClassificationResult(
            is_related=True, positive_score=0.99, threshold=0.5
        )


class CapturingAgent:
    def __init__(self) -> None:
        self.contexts: list[AgentContext] = []

    def run(self, context: AgentContext, target_language: str = "zh") -> AgentResult:
        self.contexts.append(context)
        return AgentResult(content="已根据上文继续处理。")


class FixedExpiryInferer:
    def infer(self, name: str, now: datetime) -> datetime:
        assert name == "番茄"
        return datetime(2099, 1, 2, tzinfo=UTC)


class ScriptedChatModel(FakeMessagesListChatModel):
    bound_tool_names: list[str] = Field(default_factory=list)
    received_messages: list[list[BaseMessage]] = Field(default_factory=list)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Any | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> "ScriptedChatModel":
        self.bound_tool_names = [
            str(getattr(tool, "name", tool.get("name") if isinstance(tool, dict) else ""))
            for tool in tools
        ]
        return self

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any):
        self.received_messages.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def build_test_pipeline(tmp_path: Path):
    profile_path = tmp_path / "profile.md"
    profile_path.write_text("# 用户画像\n", encoding="utf-8")
    settings = replace(
        get_settings(),
        deepseek_api_key="",
        inventory_db_path=tmp_path / "inventory.db",
        user_profile_path=profile_path,
    )
    return build_pipeline(settings)


def install_scripted_agent(
    pipeline: Any,
    responses: list[AIMessage],
) -> ScriptedChatModel:
    model = ScriptedChatModel(responses=responses)
    recipes = RecipeRouter(
        None,
        TheMealDbClient(
            base_url=pipeline.settings.themealdb_base_url,
            api_key=pipeline.settings.themealdb_api_key,
            timeout=pipeline.settings.http_timeout_seconds,
        ),
    )
    pipeline.agent = KitchenAgent(
        model=model,  # type: ignore[arg-type]
        inventory=pipeline.inventory,
        recipes=recipes,
        actions=pipeline.actions,
    )
    return model


def test_non_food_query_is_blocked_before_deepseek(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "profile.md"
    profile_path.write_text(
        "# 用户画像\n\n## 过敏原\n\n- 暂无\n",
        encoding="utf-8",
    )
    settings = replace(
        get_settings(),
        deepseek_api_key="",
        inventory_db_path=tmp_path / "inventory.db",
        user_profile_path=profile_path,
    )

    pipeline = build_pipeline(settings)
    result = pipeline.process_request(user_id="test", text="Python怎么安装")

    assert result.blocked is True
    assert result.audit_reason == "food_query_irrelevant"
    assert "仅支持食谱和食材" in result.content


def test_simple_write_preserves_quantity_text_without_deepseek(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "profile.md"
    profile_path.write_text("# 用户画像\n", encoding="utf-8")
    settings = replace(
        get_settings(),
        deepseek_api_key="",
        inventory_db_path=tmp_path / "inventory.db",
        user_profile_path=profile_path,
    )
    pipeline = build_pipeline(settings)
    pipeline.actions.expiry_inferer = FixedExpiryInferer()

    result = pipeline.process_request(user_id="test", text="添加 02个 番茄到库存")

    assert "待确认" in result.content
    assert pipeline.inventory.find_by_name("番茄") == []
    pending = pipeline.actions.repository.list_pending("test")
    assert len(pending) == 1
    assert pending[0].arguments == {
        "name": "番茄",
        "quantity_text": "02个",
        "expires_at": "2099-01-02T00:00:00Z",
        "expiry_source": "llm",
    }
    assert result.tool_history[0]["tool"] == "propose_add_inventory_instance"


def test_missing_classifier_artifact_fails_open(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.md"
    profile_path.write_text("# 用户画像\n", encoding="utf-8")
    settings = replace(
        get_settings(),
        deepseek_api_key="",
        inventory_db_path=tmp_path / "inventory.db",
        user_profile_path=profile_path,
        food_classifier_model_path=tmp_path / "missing.bin",
        food_classifier_metadata_path=tmp_path / "missing.json",
    )
    pipeline = build_pipeline(settings)

    result = pipeline.process_request(user_id="test", text="帮我加入购物车")

    assert result.blocked is False
    classifier_event = result.execution_trace[1]
    assert classifier_event.name == "food_query_classifier"
    assert classifier_event.status == "unavailable"
    assert "fail-open" in classifier_event.detail


def test_inventory_query_returns_exact_instance_snapshots_without_deepseek(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "profile.md"
    profile_path.write_text("# 用户画像\n", encoding="utf-8")
    settings = replace(
        get_settings(),
        deepseek_api_key="",
        inventory_db_path=tmp_path / "inventory.db",
        user_profile_path=profile_path,
    )
    pipeline = build_pipeline(settings)
    item = pipeline.inventory.add_instance(
        "土豆", "2 个", None, "unknown", now=datetime(2026, 8, 7, tzinfo=UTC)
    )

    result = pipeline.process_request(user_id="test", text="查询一下当前库存")

    assert result.content == (
        f"当前库存：\n- 土豆 [{str(item['instance_id'])[:8]}]：2 个"
    )
    assert result.blocked is False
    assert result.tool_history == [
        {
            "tool": "list_inventory_instances",
            "arguments": {},
            "result": [item],
        }
    ]
    assert [event.name for event in result.execution_trace] == [
        "deterministic_route",
        "intent_match",
        "inventory_query_regex",
        "list_inventory",
        "regex_audit",
        "tool_call_audit",
    ]
    assert all(event.duration_ms >= 0 for event in result.execution_trace)


def test_irrelevant_greeting_bypasses_classifier_with_boundary_result(
    tmp_path: Path,
) -> None:
    pipeline = build_test_pipeline(tmp_path)
    classifier = ForbiddenClassifier()
    pipeline.food_classifier = classifier  # type: ignore[assignment]

    result = pipeline.process_request(user_id="test", text="你好")

    assert result.blocked is True
    assert result.audit_reason == "food_query_irrelevant"
    assert result.content == "当前仅支持食谱和食材相关问题。"
    assert classifier.calls == 0
    assert [event.name for event in result.execution_trace] == [
        "deterministic_route"
    ]
    assert result.execution_trace[0].detail == "irrelevant"


def test_supported_list_bypasses_classifier_and_returns_real_inventory(
    tmp_path: Path,
) -> None:
    pipeline = build_test_pipeline(tmp_path)
    classifier = ForbiddenClassifier()
    pipeline.food_classifier = classifier  # type: ignore[assignment]
    item = pipeline.inventory.add_instance("土豆", "大约100个", None, "unknown")

    result = pipeline.process_request(user_id="test", text="查询库存")

    assert result.content == (
        f"当前库存：\n- 土豆 [{str(item['instance_id'])[:8]}]：大约100个"
    )
    assert classifier.calls == 0
    assert result.execution_trace[0].name == "deterministic_route"
    assert result.execution_trace[0].detail == "supported; inventory.list"


def test_supported_named_read_bypasses_classifier_and_returns_exact_instances(
    tmp_path: Path,
) -> None:
    pipeline = build_test_pipeline(tmp_path)
    classifier = ForbiddenClassifier()
    pipeline.food_classifier = classifier  # type: ignore[assignment]
    first = pipeline.inventory.add_instance("土豆", "2个", None, "unknown")
    second = pipeline.inventory.add_instance("土豆", "半袋", None, "unknown")

    result = pipeline.process_request(user_id="test", text="查看土豆还有多少")

    assert result.content == (
        "土豆库存：\n"
        f"- 土豆 [{str(first['instance_id'])[:8]}]：2个\n"
        f"- 土豆 [{str(second['instance_id'])[:8]}]：半袋"
    )
    assert result.tool_history == [
        {
            "tool": "find_inventory_instances",
            "arguments": {"name": "土豆"},
            "result": [first, second],
        }
    ]
    assert classifier.calls == 0
    assert [event.name for event in result.execution_trace[:3]] == [
        "deterministic_route",
        "intent_match",
        "inventory_query_regex",
    ]


def test_supported_remove_all_bypasses_classifier_and_stays_pending(
    tmp_path: Path,
) -> None:
    pipeline = build_test_pipeline(tmp_path)
    classifier = ForbiddenClassifier()
    pipeline.food_classifier = classifier  # type: ignore[assignment]
    pipeline.inventory.add_instance("土豆", "2个", None, "unknown")
    pipeline.inventory.add_instance("土豆", "1袋", None, "unknown")

    result = pipeline.process_request(user_id="test", text="删除所有土豆")

    assert "待确认" in result.content
    assert len(pipeline.inventory.find_by_name("土豆")) == 2
    pending = pipeline.actions.repository.list_pending("test")
    assert len(pending) == 1
    assert pending[0].operation == "inventory.remove_all"
    assert pending[0].arguments == {"name": "土豆"}
    assert result.tool_history[0]["tool"] == "propose_remove_all_inventory_instances"
    assert classifier.calls == 0
    assert result.execution_trace[0].detail == "supported; inventory.remove_all"


def test_contextual_route_bypasses_classifier_and_supplies_recent_history(
    tmp_path: Path,
) -> None:
    pipeline = build_test_pipeline(tmp_path)
    classifier = ForbiddenClassifier()
    agent = CapturingAgent()
    pipeline.food_classifier = classifier  # type: ignore[assignment]
    pipeline.agent = agent  # type: ignore[assignment]
    pipeline.history.append(
        user_id="test",
        request="土豆怎么处理",
        response="可以先清洗土豆。",
    )

    result = pipeline.process_request(user_id="test", text="下一步呢")

    assert result.content == "已根据上文继续处理。"
    assert classifier.calls == 0
    assert len(agent.contexts) == 1
    assert [segment.request for segment in agent.contexts[0].history_segments] == [
        "土豆怎么处理"
    ]
    assert result.execution_trace[0].name == "deterministic_route"
    assert result.execution_trace[0].detail == "contextual"


def test_relative_use_runs_real_agent_and_keeps_replacement_pending(
    tmp_path: Path,
) -> None:
    pipeline = build_test_pipeline(tmp_path)
    classifier = ForbiddenClassifier()
    original = pipeline.inventory.add_instance("番茄", "2个", None, "unknown")
    instance_id = str(original["instance_id"])
    model = install_scripted_agent(
        pipeline,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "find_inventory_instances",
                        "args": {"name": "番茄"},
                        "id": "call-find",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "propose_replace_inventory_instance",
                        "args": {
                            "instance_id": instance_id,
                            "quantity_text": "1个",
                            "expires_at": None,
                            "expiry_source": "unknown",
                        },
                        "id": "call-replace",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="已生成替换库存实例的待确认操作，确认前库存不会改变。"),
        ],
    )
    pipeline.food_classifier = classifier  # type: ignore[assignment]

    result = pipeline.process_request(user_id="test", text="用了一半")

    assert result.blocked is False
    assert classifier.calls == 0
    assert pipeline.inventory.get_instance(instance_id) == original
    pending = pipeline.actions.repository.list_pending("test")
    assert len(pending) == 1
    assert pending[0].operation == "inventory.replace"
    assert pending[0].arguments == {
        "instance_id": instance_id,
        "quantity_text": "1个",
        "expires_at": None,
        "expiry_source": "unknown",
    }
    assert [entry["tool"] for entry in result.tool_history] == [
        "find_inventory_instances",
        "propose_replace_inventory_instance",
    ]
    assert "find_inventory_instances" in model.bound_tool_names
    assert "propose_replace_inventory_instance" in model.bound_tool_names


def test_relative_replacement_accepts_equivalent_expiry_serializations(
    tmp_path: Path,
) -> None:
    pipeline = build_test_pipeline(tmp_path)
    original = pipeline.inventory.add_instance(
        "番茄",
        "2个",
        datetime(2026, 8, 10, 9, tzinfo=UTC),
        "user",
    )
    instance_id = str(original["instance_id"])
    install_scripted_agent(
        pipeline,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "find_inventory_instances",
                        "args": {"name": "番茄"},
                        "id": "call-find",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "propose_replace_inventory_instance",
                        "args": {
                            "instance_id": instance_id,
                            "quantity_text": "1个",
                            "expires_at": "2026-08-10T09:00:00+00:00",
                            "expiry_source": "user",
                        },
                        "id": "call-replace",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="已生成待确认操作，确认前库存不会改变。"),
        ],
    )

    result = pipeline.process_request(user_id="test", text="用了一半")

    assert result.blocked is False
    assert pipeline.inventory.get_instance(instance_id) == original
    pending = pipeline.actions.repository.list_pending("test")
    assert len(pending) == 1
    assert pending[0].arguments == {
        "instance_id": instance_id,
        "quantity_text": "1个",
        "expires_at": "2026-08-10T09:00:00Z",
        "expiry_source": "user",
    }


def test_named_replacement_reaches_agent_lookup_instead_of_direct_write(
    tmp_path: Path,
) -> None:
    pipeline = build_test_pipeline(tmp_path)
    classifier = RelatedClassifierSpy()
    original = pipeline.inventory.add_instance("番茄", "2个", None, "unknown")
    instance_id = str(original["instance_id"])
    install_scripted_agent(
        pipeline,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "find_inventory_instances",
                        "args": {"name": "番茄"},
                        "id": "call-find",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "propose_replace_inventory_instance",
                        "args": {
                            "instance_id": instance_id,
                            "quantity_text": "1个",
                            "expires_at": None,
                            "expiry_source": "unknown",
                        },
                        "id": "call-replace",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="已生成待确认操作，确认前库存不会改变。"),
        ],
    )
    pipeline.food_classifier = classifier  # type: ignore[assignment]

    result = pipeline.process_request(user_id="test", text="将番茄更新为1个")

    assert result.blocked is False
    assert classifier.calls == ["将番茄更新为1个"]
    assert pipeline.inventory.get_instance(instance_id) == original
    pending = pipeline.actions.repository.list_pending("test")
    assert len(pending) == 1
    assert pending[0].operation == "inventory.replace"
    assert [entry["tool"] for entry in result.tool_history] == [
        "find_inventory_instances",
        "propose_replace_inventory_instance",
    ]


def test_undecided_request_invokes_classifier_once(tmp_path: Path) -> None:
    pipeline = build_test_pipeline(tmp_path)
    classifier = IrrelevantClassifierSpy()
    pipeline.food_classifier = classifier  # type: ignore[assignment]

    result = pipeline.process_request(user_id="test", text="Python怎么安装")

    assert classifier.calls == ["Python怎么安装"]
    assert result.blocked is True
    assert [event.name for event in result.execution_trace] == [
        "deterministic_route",
        "food_query_classifier",
    ]
