from app.memory import parse_user_profile


def test_parse_markdown_profile() -> None:
    profile = parse_user_profile(
        "# 用户画像\n\n## 过敏原\n\n- 花生\n\n## 饮食偏好\n\n- 少辣\n"
    )
    assert profile.allergens == ["花生"]
    assert profile.preferences == ["少辣"]
