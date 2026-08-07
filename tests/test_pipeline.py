from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from classifier.model import ClassificationResult
from app.config import get_settings
from app.context import AgentContext, AgentResult
from app.pipeline import build_pipeline


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


class CapturingAgent:
    def __init__(self) -> None:
        self.contexts: list[AgentContext] = []

    def run(self, context: AgentContext, target_language: str = "zh") -> AgentResult:
        self.contexts.append(context)
        return AgentResult(content="已根据上文继续处理。")


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

    result = pipeline.process_request(user_id="test", text="添加 02个 番茄到库存")

    assert "待确认" in result.content
    assert pipeline.inventory.find_by_name("番茄") == []
    pending = pipeline.actions.repository.list_pending("test")
    assert len(pending) == 1
    assert pending[0].arguments == {
        "name": "番茄",
        "quantity_text": "02个",
        "expires_at": None,
        "expiry_source": "unknown",
    }


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

    assert result.content == "当前库存：\n- 土豆：2 个"
    assert result.blocked is False
    assert result.tool_history == [
        {
            "tool": "list_inventory",
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
    pipeline.inventory.add_instance("土豆", "2个", None, "unknown")

    result = pipeline.process_request(user_id="test", text="查询库存")

    assert result.content == "当前库存：\n- 土豆：2个"
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

    assert result.content == "土豆库存：\n- 土豆：2个\n- 土豆：半袋"
    assert result.tool_history == [
        {
            "tool": "find_inventory",
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
