from pathlib import Path

from app.memory import (
    ConversationHistoryRepository,
    DEFAULT_USER_PROFILE_TEMPLATE,
    parse_user_profile,
    read_user_profile,
    write_user_allergens,
)


def test_conversation_history_round_trips_final_audit_reason(tmp_path: Path) -> None:
    repository = ConversationHistoryRepository(tmp_path / "history.db")
    repository.initialize()

    repository.append(
        user_id="u",
        request="推荐一道菜",
        response="暂时无法完成",
        audit_reason="retry_exhausted",
    )

    assert repository.list_recent("u")[0].audit_reason == "retry_exhausted"


def test_conversation_history_adds_audit_column_to_existing_database(
    tmp_path: Path,
) -> None:
    import sqlite3

    path = tmp_path / "history.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE conversation_history (
                segment_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                request_text TEXT NOT NULL,
                response_text TEXT NOT NULL,
                tool_history_json TEXT NOT NULL,
                confirmation_result_json TEXT,
                created_at TEXT NOT NULL
            )
            """
        )

    repository = ConversationHistoryRepository(path)
    repository.initialize()
    repository.append(
        user_id="u",
        request="取消",
        response="已取消",
        audit_reason="user_cancelled",
    )

    assert repository.list_recent("u")[0].audit_reason == "user_cancelled"


def test_read_user_profile_initializes_missing_file_from_template(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nested" / "profile.md"

    profile = read_user_profile(path)

    expected = DEFAULT_USER_PROFILE_TEMPLATE.read_text(encoding="utf-8")
    assert path.read_text(encoding="utf-8") == expected
    assert profile.markdown == expected
    assert profile.allergens == []
    assert profile.preferences == []


def test_read_user_profile_does_not_replace_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "profile.md"
    existing = "# 用户画像\n\n## 饮食偏好\n\n- 少辣\n"
    path.write_text(existing, encoding="utf-8")

    profile = read_user_profile(path)

    assert path.read_text(encoding="utf-8") == existing
    assert profile.preferences == ["少辣"]


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
