"""User profile and bounded conversation-history persistence."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field

from app.context import HistorySegment

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


class ConversationHistoryRepository:
    """Persist flat turns and expose only the latest bounded window.

    A row never contains the history that was supplied to its Agent call. This
    prevents old windows from being recursively copied into newer history rows.
    """

    def __init__(self, db_path: Path, max_segments: int = 5) -> None:
        if max_segments <= 0:
            raise ValueError("max_segments must be positive")
        self.db_path = db_path
        self.max_segments = max_segments

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
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_history_user_created "
                "ON conversation_history(user_id, created_at DESC)"
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
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO conversation_history(
                    segment_id, user_id, request_text, response_text,
                    tool_history_json, confirmation_result_json, audit_reason,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"hist_{uuid4().hex}",
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
                    now,
                ),
            )
            connection.execute(
                """
                DELETE FROM conversation_history
                WHERE user_id = ? AND segment_id NOT IN (
                    SELECT segment_id
                    FROM conversation_history
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                )
                """,
                (user_id, user_id, self.max_segments),
            )

    def list_recent(self, user_id: str) -> list[HistorySegment]:
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT request_text, response_text, tool_history_json,
                       confirmation_result_json, audit_reason
                FROM conversation_history
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, self.max_segments),
            ).fetchall()
        rows.reverse()
        return [
            HistorySegment(
                request=str(row[0]),
                response=str(row[1]),
                tool_history=json.loads(str(row[2])),
                confirmation_result=(
                    json.loads(str(row[3])) if row[3] is not None else None
                ),
                audit_reason=str(row[4]) if row[4] is not None else None,
            )
            for row in rows
        ]


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
