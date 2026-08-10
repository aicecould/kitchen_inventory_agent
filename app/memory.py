"""User profile and bounded conversation-history persistence."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field

from app.context import ExecutionTraceEvent, HistoryExcerpt

DEFAULT_USER_PROFILE_TEMPLATE = (
    Path(__file__).resolve().parent / "templates" / "user_profile.md"
)


class UserProfile(BaseModel):
    markdown: str
    allergens: list[str] = Field(default_factory=list)
    allergen_intolerances: list[str] = Field(default_factory=list)
    custom_allergens: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)
    history: list[str] = Field(default_factory=list)


class ConversationOutcome(StrEnum):
    COMPLETED = "completed"
    BLOCKED = "blocked"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    FAILED = "failed"
    CONFIRMATION = "confirmation"


class ConversationRecord(BaseModel):
    segment_id: str
    user_id: str
    request: str
    response: str
    tool_history: list[dict[str, object]] = Field(default_factory=list)
    confirmation_result: dict[str, object] | None = None
    audit_reason: str | None = None
    is_relevant: bool = True
    outcome_type: ConversationOutcome = ConversationOutcome.COMPLETED
    blocked: bool = False
    execution_trace: list[ExecutionTraceEvent] = Field(default_factory=list)
    created_at: datetime


class HistoryRecall(BaseModel):
    created_at: datetime
    request: str
    response: str


class ConversationHistoryRepository:
    """Persist complete flat turns without coupling storage to context limits.

    A row never contains the history that was supplied to its Agent call. This
    prevents old windows from being recursively copied into newer history rows.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_history (
                    segment_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    request_text TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    tool_history_json TEXT NOT NULL,
                    confirmation_result_json TEXT,
                    audit_reason TEXT,
                    is_relevant INTEGER NOT NULL DEFAULT 1,
                    outcome_type TEXT NOT NULL DEFAULT 'completed',
                    blocked INTEGER NOT NULL DEFAULT 0,
                    execution_trace_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                )
                """
            )
            columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(conversation_history)"
                ).fetchall()
            }
            if "audit_reason" not in columns:
                connection.execute(
                    "ALTER TABLE conversation_history ADD COLUMN audit_reason TEXT"
                )
            migrations = {
                "is_relevant": (
                    "ALTER TABLE conversation_history "
                    "ADD COLUMN is_relevant INTEGER NOT NULL DEFAULT 1"
                ),
                "outcome_type": (
                    "ALTER TABLE conversation_history "
                    "ADD COLUMN outcome_type TEXT NOT NULL DEFAULT 'completed'"
                ),
                "blocked": (
                    "ALTER TABLE conversation_history "
                    "ADD COLUMN blocked INTEGER NOT NULL DEFAULT 0"
                ),
                "execution_trace_json": (
                    "ALTER TABLE conversation_history "
                    "ADD COLUMN execution_trace_json TEXT NOT NULL DEFAULT '[]'"
                ),
            }
            for column, statement in migrations.items():
                if column not in columns:
                    connection.execute(statement)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_history_user_created "
                "ON conversation_history(user_id, created_at DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_history_user_relevant_created "
                "ON conversation_history(user_id, is_relevant, created_at DESC)"
            )

    def append(
        self,
        *,
        user_id: str,
        request: str,
        response: str,
        tool_history: list[dict[str, object]] | None = None,
        confirmation_result: dict[str, object] | None = None,
        audit_reason: str | None = None,
        is_relevant: bool = True,
        outcome_type: ConversationOutcome = ConversationOutcome.COMPLETED,
        blocked: bool = False,
        execution_trace: list[ExecutionTraceEvent] | None = None,
        created_at: datetime | None = None,
    ) -> ConversationRecord:
        timestamp = _as_utc(created_at or datetime.now(UTC))
        segment_id = f"hist_{uuid4().hex}"
        trace = execution_trace or []
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO conversation_history(
                    segment_id, user_id, request_text, response_text,
                    tool_history_json, confirmation_result_json, audit_reason,
                    is_relevant, outcome_type, blocked, execution_trace_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    segment_id,
                    user_id,
                    request.strip(),
                    response.strip(),
                    json.dumps(tool_history or [], ensure_ascii=False),
                    (
                        json.dumps(confirmation_result, ensure_ascii=False)
                        if confirmation_result is not None
                        else None
                    ),
                    audit_reason,
                    int(is_relevant),
                    outcome_type.value,
                    int(blocked),
                    json.dumps(
                        [event.model_dump(mode="json") for event in trace],
                        ensure_ascii=False,
                    ),
                    timestamp.isoformat(),
                ),
            )
        return ConversationRecord(
            segment_id=segment_id,
            user_id=user_id,
            request=request.strip(),
            response=response.strip(),
            tool_history=tool_history or [],
            confirmation_result=confirmation_result,
            audit_reason=audit_reason,
            is_relevant=is_relevant,
            outcome_type=outcome_type,
            blocked=blocked,
            execution_trace=trace,
            created_at=timestamp,
        )

    def list_records(self, user_id: str) -> list[ConversationRecord]:
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT segment_id, user_id, request_text, response_text,
                       tool_history_json, confirmation_result_json, audit_reason,
                       is_relevant, outcome_type, blocked, execution_trace_json,
                       created_at
                FROM conversation_history
                WHERE user_id = ?
                ORDER BY created_at ASC, rowid ASC
                """,
                (user_id,),
            ).fetchall()
        return [_conversation_record(row) for row in rows]

    def list_context_window(
        self, user_id: str, limit: int = 10
    ) -> list[HistoryExcerpt]:
        if limit <= 0:
            raise ValueError("context window limit must be positive")
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT request_text, response_text
                FROM conversation_history
                WHERE user_id = ? AND is_relevant = 1
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        rows.reverse()
        return [
            HistoryExcerpt(
                request=str(row[0]),
                response=str(row[1]),
            )
            for row in rows
        ]

    def search_by_time(
        self,
        user_id: str,
        start_at: datetime,
        end_at: datetime,
        *,
        limit: int = 20,
    ) -> list[HistoryRecall]:
        if not 1 <= limit <= 100:
            raise ValueError("history recall limit must be between 1 and 100")
        start_utc = _as_utc_bound(start_at)
        end_utc = _as_utc_bound(end_at)
        if start_utc >= end_utc:
            raise ValueError("start_at must be earlier than end_at")
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT created_at, request_text, response_text
                FROM conversation_history
                WHERE user_id = ? AND created_at >= ? AND created_at < ?
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?
                """,
                (
                    user_id,
                    start_utc.isoformat(),
                    end_utc.isoformat(),
                    limit,
                ),
            ).fetchall()
        return [
            HistoryRecall(
                created_at=datetime.fromisoformat(str(row[0])),
                request=str(row[1]),
                response=str(row[2]),
            )
            for row in rows
        ]


def _conversation_record(row: tuple[object, ...]) -> ConversationRecord:
    return ConversationRecord(
        segment_id=str(row[0]),
        user_id=str(row[1]),
        request=str(row[2]),
        response=str(row[3]),
        tool_history=json.loads(str(row[4])),
        confirmation_result=(
            json.loads(str(row[5])) if row[5] is not None else None
        ),
        audit_reason=str(row[6]) if row[6] is not None else None,
        is_relevant=bool(row[7]),
        outcome_type=ConversationOutcome(str(row[8])),
        blocked=bool(row[9]),
        execution_trace=json.loads(str(row[10])),
        created_at=datetime.fromisoformat(str(row[11])),
    )


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("conversation timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _as_utc_bound(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("history time bounds must be timezone-aware")
    return value.astimezone(UTC)


def load_user_profile(path: Path) -> str:
    if not path.exists():
        template = DEFAULT_USER_PROFILE_TEMPLATE.read_text(encoding="utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(template)
        except FileExistsError:
            pass
    return path.read_text(encoding="utf-8")


def parse_user_profile(markdown: str) -> UserProfile:
    sections: dict[str, list[str]] = {}
    current = ""
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            current = line[3:].strip()
            sections.setdefault(current, [])
        elif current and line.startswith("- "):
            value = line[2:].strip()
            if value and value not in {"待填写", "暂无"}:
                sections[current].append(value)

    broad = sections.get("广义过敏原", [])
    custom = [*sections.get("自定义过敏食材", []), *sections.get("过敏原", [])]
    return UserProfile(
        markdown=markdown,
        allergens=list(dict.fromkeys([*broad, *custom])),
        allergen_intolerances=broad,
        custom_allergens=list(dict.fromkeys(custom)),
        preferences=sections.get("饮食偏好", []),
        history=sections.get("历史摘要", []),
    )


def read_user_profile(path: Path) -> UserProfile:
    return parse_user_profile(load_user_profile(path))


def profile_without_legacy_history(markdown: str) -> str:
    """Remove the old Markdown history section before building Agent context."""
    retained: list[str] = []
    skipping = False
    for line in markdown.splitlines():
        if line.startswith("## "):
            skipping = line[3:].strip() == "历史摘要"
        if not skipping:
            retained.append(line)
    return "\n".join(retained).rstrip()


def write_user_allergens(
    path: Path,
    allergen_intolerances: list[str],
    custom_allergens: list[str],
) -> UserProfile:
    markdown = load_user_profile(path)
    target_sections = {"过敏原", "广义过敏原", "自定义过敏食材"}
    retained: list[str] = []
    skipping = False
    for line in markdown.splitlines():
        if line.startswith("## "):
            skipping = line[3:].strip() in target_sections
        if not skipping:
            retained.append(line)

    cleaned = "\n".join(retained).rstrip()
    broad_lines = [f"- {value}" for value in allergen_intolerances] or ["- 暂无"]
    custom_lines = [f"- {value}" for value in custom_allergens] or ["- 暂无"]
    updated = (
        f"{cleaned}\n\n## 广义过敏原\n\n"
        + "\n".join(broad_lines)
        + "\n\n## 自定义过敏食材\n\n"
        + "\n".join(custom_lines)
        + "\n"
    )
    path.write_text(updated, encoding="utf-8")
    return parse_user_profile(updated)
