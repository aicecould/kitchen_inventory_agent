from dataclasses import replace
from pathlib import Path

from app.config import get_settings
from app.pipeline import build_pipeline


def test_order_fast_path_does_not_require_deepseek(
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
    result = pipeline.process_request(user_id="test", text="帮我下单")

    assert result.blocked is False
    assert "暂不支持订单" in result.content


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
    assert pipeline.inventory.get_item("番茄") is None
    assert len(pipeline.actions.repository.list_pending("test")) == 1
