from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from app.config import get_settings
from app.pipeline import build_pipeline


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


def test_simple_write_creates_confirmation_without_deepseek(
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

    result = pipeline.process_request(user_id="test", text="添加 2 个 番茄到库存")

    assert "待确认" in result.content
    assert pipeline.inventory.find_by_name("番茄") == []
    pending = pipeline.actions.repository.list_pending("test")
    assert len(pending) == 1
    assert pending[0].arguments == {
        "name": "番茄",
        "quantity_text": "2 个",
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

    result = pipeline.process_request(user_id="test", text="查询一下当前库存")

    assert result.blocked is False
    classifier_event = result.execution_trace[0]
    assert classifier_event.name == "food_query_classifier"
    assert classifier_event.status == "unavailable"
    assert "fail-open" in classifier_event.detail


def test_inventory_query_returns_prefilled_database_result_without_deepseek(
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
    pipeline.inventory.add_instance(
        "土豆", "2 个", None, "unknown", now=datetime(2026, 8, 7, tzinfo=UTC)
    )

    result = pipeline.process_request(user_id="test", text="查询一下当前库存")

    assert result.content == "当前库存：\n- 土豆：2 个"
    assert result.blocked is False
    assert result.tool_history == [
        {
            "tool": "list_inventory",
            "arguments": {},
            "result": [{"name": "土豆", "quantity": 2.0, "unit": "个"}],
        }
    ]
    assert [event.name for event in result.execution_trace] == [
        "food_query_classifier",
        "intent_match",
        "inventory_query_regex",
        "list_inventory",
        "regex_audit",
        "tool_call_audit",
    ]
    assert all(event.duration_ms >= 0 for event in result.execution_trace)
