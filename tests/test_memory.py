from pathlib import Path

from app.memory import parse_user_profile, write_user_allergens


def test_parse_markdown_profile() -> None:
    profile = parse_user_profile(
        "# 用户画像\n\n## 过敏原\n\n- 花生\n\n## 饮食偏好\n\n- 少辣\n"
    )
    assert profile.allergens == ["花生"]
    assert profile.preferences == ["少辣"]


def test_write_user_allergens_separates_api_categories_and_custom_foods(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profile.md"
    path.write_text(
        "# 用户画像\n\n## 过敏原\n\n- 花生\n\n## 饮食偏好\n\n- 少辣\n",
        encoding="utf-8",
    )

    profile = write_user_allergens(path, ["Dairy", "Peanut"], ["腰果", "虾皮"])

    assert profile.allergen_intolerances == ["Dairy", "Peanut"]
    assert profile.custom_allergens == ["腰果", "虾皮"]
    assert profile.preferences == ["少辣"]
    saved = path.read_text(encoding="utf-8")
    assert "## 广义过敏原" in saved
    assert "## 自定义过敏食材" in saved
    assert "## 过敏原" not in saved
