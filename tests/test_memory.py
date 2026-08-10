import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from app.context import ExecutionTraceEvent
from app.memory import (
    ConversationOutcome,
    ConversationHistoryRepository,
    DEFAULT_USER_PROFILE_TEMPLATE,
    parse_user_profile,
    read_user_profile,
    write_user_allergens,
)


def test_history_migrates_old_rows_and_never_prunes(tmp_path: Path) -> None:
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
                audit_reason TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO conversation_history VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "old",
                "u",
                "旧问题",
                "旧回答",
                "[]",
                None,
                None,
                "2026-08-01T00:00:00+00:00",
            ),
        )

    repository = ConversationHistoryRepository(path)
    repository.initialize()
    for index in range(12):
        repository.append(user_id="u", request=f"q{index}", response=f"a{index}")

    records = repository.list_records("u")
    assert len(records) == 13
    assert records[0].segment_id == "old"
    assert records[0].is_relevant is True
    assert records[0].outcome_type == ConversationOutcome.COMPLETED
    assert records[0].blocked is False
    assert records[0].execution_trace == []


def test_history_full_record_round_trips(tmp_path: Path) -> None:
    repository = ConversationHistoryRepository(tmp_path / "history.db")
    repository.initialize()
    trace = [
        ExecutionTraceEvent(
            stage="routing",
            name="deterministic_route",
            status="matched",
            detail="irrelevant",
            duration_ms=3,
        )
    ]

    saved = repository.append(
        user_id="u",
        request="写代码",
        response="当前仅支持食谱和食材相关问题。",
        tool_history=[{"tool": "classifier", "result": "irrelevant"}],
        confirmation_result={"decision": "cancel"},
        audit_reason="food_query_irrelevant",
        is_relevant=False,
        outcome_type=ConversationOutcome.BLOCKED,
        blocked=True,
        execution_trace=trace,
        created_at=datetime(2026, 8, 10, 12, 0, tzinfo=UTC),
    )

    assert repository.list_records("u") == [saved]
    assert saved.tool_history == [{"tool": "classifier", "result": "irrelevant"}]
    assert saved.confirmation_result == {"decision": "cancel"}
    assert saved.audit_reason == "food_query_irrelevant"
    assert saved.is_relevant is False
    assert saved.outcome_type == ConversationOutcome.BLOCKED
    assert saved.blocked is True
    assert saved.execution_trace == trace
    assert saved.created_at == datetime(2026, 8, 10, 12, 0, tzinfo=UTC)


def test_context_window_skips_irrelevant_rows_before_taking_ten(
    tmp_path: Path,
) -> None:
    repository = ConversationHistoryRepository(tmp_path / "history.db")
    repository.initialize()
    base = datetime(2026, 8, 1, tzinfo=UTC)
    for index in range(14):
        repository.append(
            user_id="u",
            request=f"q{index}",
            response=f"a{index}",
            is_relevant=index not in {11, 13},
            tool_history=[{"secret": index}],
            audit_reason=(
                "food_query_irrelevant" if index in {11, 13} else None
            ),
            created_at=base + timedelta(minutes=index),
        )

    window = repository.list_context_window("u")

    assert [item.request for item in window] == [
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q7",
        "q8",
        "q9",
        "q10",
        "q12",
    ]
    assert [set(item.model_dump()) for item in window] == [
        {"request", "response"}
    ] * 10
    assert len(repository.list_records("u")) == 14


def test_context_window_rejects_non_positive_limit(tmp_path: Path) -> None:
    repository = ConversationHistoryRepository(tmp_path / "history.db")
    repository.initialize()

    with pytest.raises(ValueError, match="positive"):
        repository.list_context_window("u", limit=0)


def test_search_by_time_uses_aware_half_open_utc_range(tmp_path: Path) -> None:
    repository = ConversationHistoryRepository(tmp_path / "history.db")
    repository.initialize()
    for minute in range(4):
        repository.append(
            user_id="u",
            request=f"q{minute}",
            response=f"a{minute}",
            is_relevant=minute != 2,
            created_at=datetime(2026, 8, 10, 0, minute, tzinfo=UTC),
        )
    repository.append(
        user_id="other",
        request="private",
        response="private",
        created_at=datetime(2026, 8, 10, 0, 1, tzinfo=UTC),
    )

    results = repository.search_by_time(
        "u",
        datetime(2026, 8, 10, 8, 1, tzinfo=ZoneInfo("Asia/Shanghai")),
        datetime(2026, 8, 10, 8, 3, tzinfo=ZoneInfo("Asia/Shanghai")),
    )

    assert [item.request for item in results] == ["q2", "q1"]
    assert all(
        set(item.model_dump()) == {"created_at", "request", "response"}
        for item in results
    )


def test_search_by_time_defaults_to_twenty_and_allows_one_hundred(
    tmp_path: Path,
) -> None:
    repository = ConversationHistoryRepository(tmp_path / "history.db")
    repository.initialize()
    base = datetime(2026, 8, 1, tzinfo=UTC)
    for index in range(105):
        repository.append(
            user_id="u",
            request=f"q{index}",
            response=f"a{index}",
            created_at=base + timedelta(minutes=index),
        )

    default_results = repository.search_by_time(
        "u", base, base + timedelta(days=1)
    )
    capped_results = repository.search_by_time(
        "u", base, base + timedelta(days=1), limit=100
    )

    assert len(default_results) == 20
    assert default_results[0].request == "q104"
    assert len(capped_results) == 100
    assert capped_results[-1].request == "q5"


@pytest.mark.parametrize(
    ("start_at", "end_at", "limit", "message"),
    [
        (
            datetime(2026, 8, 1),
            datetime(2026, 8, 2, tzinfo=UTC),
            20,
            "timezone-aware",
        ),
        (
            datetime(2026, 8, 1, tzinfo=UTC),
            datetime(2026, 8, 2),
            20,
            "timezone-aware",
        ),
        (
            datetime(2026, 8, 2, tzinfo=UTC),
            datetime(2026, 8, 2, tzinfo=UTC),
            20,
            "earlier",
        ),
        (
            datetime(2026, 8, 3, tzinfo=UTC),
            datetime(2026, 8, 2, tzinfo=UTC),
            20,
            "earlier",
        ),
        (
            datetime(2026, 8, 1, tzinfo=UTC),
            datetime(2026, 8, 2, tzinfo=UTC),
            0,
            "between 1 and 100",
        ),
        (
            datetime(2026, 8, 1, tzinfo=UTC),
            datetime(2026, 8, 2, tzinfo=UTC),
            101,
            "between 1 and 100",
        ),
    ],
)
def test_search_by_time_rejects_invalid_bounds_and_limits(
    tmp_path: Path,
    start_at: datetime,
    end_at: datetime,
    limit: int,
    message: str,
) -> None:
    repository = ConversationHistoryRepository(tmp_path / "history.db")
    repository.initialize()

    with pytest.raises(ValueError, match=message):
        repository.search_by_time("u", start_at, end_at, limit=limit)


def test_conversation_history_round_trips_final_audit_reason(tmp_path: Path) -> None:
    repository = ConversationHistoryRepository(tmp_path / "history.db")
    repository.initialize()

    repository.append(
        user_id="u",
        request="推荐一道菜",
        response="暂时无法完成",
        audit_reason="retry_exhausted",
    )

    assert repository.list_records("u")[0].audit_reason == "retry_exhausted"


def test_conversation_history_adds_audit_column_to_existing_database(
    tmp_path: Path,
) -> None:
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

    assert repository.list_records("u")[0].audit_reason == "user_cancelled"


def test_history_accepts_one_reserved_segment_id(tmp_path: Path) -> None:
    repository = ConversationHistoryRepository(tmp_path / "history.db")
    repository.initialize()
    reserved = repository.reserve_segment_id()

    saved = repository.append(
        user_id="u", request="问题", response="回答", segment_id=reserved
    )

    assert saved.segment_id == reserved
    with pytest.raises(sqlite3.IntegrityError):
        repository.append(
            user_id="u", request="重复", response="重复", segment_id=reserved
        )


def test_history_evidence_is_ordered_and_user_scoped(tmp_path: Path) -> None:
    repository = ConversationHistoryRepository(tmp_path / "history.db")
    repository.initialize()
    first = repository.append(user_id="u", request="q1", response="a1")
    second = repository.append(user_id="u", request="q2", response="a2")
    foreign = repository.append(user_id="other", request="private", response="private")

    evidence = repository.get_evidence(
        "u", [second.segment_id, foreign.segment_id, first.segment_id]
    )

    assert [record.segment_id for record in evidence] == [
        second.segment_id,
        first.segment_id,
    ]


def test_completed_cursor_query_excludes_ineligible_records(tmp_path: Path) -> None:
    repository = ConversationHistoryRepository(tmp_path / "history.db")
    repository.initialize()
    base = datetime(2026, 8, 10, tzinfo=UTC)
    cursor = repository.append(
        user_id="u", request="old", response="old", created_at=base
    )
    expected: list[str] = []
    for index, values in enumerate(
        [
            (True, ConversationOutcome.COMPLETED, False),
            (False, ConversationOutcome.COMPLETED, False),
            (True, ConversationOutcome.BLOCKED, True),
            (True, ConversationOutcome.FAILED, False),
            (True, ConversationOutcome.CONFIRMATION, False),
            (True, ConversationOutcome.COMPLETED, False),
        ],
        start=1,
    ):
        relevant, outcome, blocked = values
        record = repository.append(
            user_id="u",
            request=f"q{index}",
            response=f"a{index}",
            is_relevant=relevant,
            outcome_type=outcome,
            blocked=blocked,
            created_at=base + timedelta(minutes=index),
        )
        if relevant and outcome == ConversationOutcome.COMPLETED and not blocked:
            expected.append(record.segment_id)

    records = repository.list_completed_after("u", cursor.segment_id, limit=10)

    assert [record.segment_id for record in records] == expected


def test_read_user_profile_returns_virtual_template_without_silent_write(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nested" / "profile.md"

    profile = read_user_profile(path)

    expected = DEFAULT_USER_PROFILE_TEMPLATE.read_text(encoding="utf-8")
    assert path.exists() is False
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


def test_legacy_allergen_writer_cannot_bypass_profile_audit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profile.md"
    path.write_text(
        "# 用户画像\n\n## 过敏原\n\n- 花生\n\n## 饮食偏好\n\n- 少辣\n",
        encoding="utf-8",
    )

    before = path.read_text(encoding="utf-8")

    with pytest.raises(RuntimeError, match="audited profile action"):
        write_user_allergens(path, ["Dairy", "Peanut"], ["腰果", "虾皮"])

    assert path.read_text(encoding="utf-8") == before
