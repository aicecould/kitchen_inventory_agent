"""Single active cooking-task attachment and lifecycle service."""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from app.memory import ConversationHistoryRepository, ConversationRecord


_TASK_ID = re.compile(r"task_[0-9a-f]{32}")
_TASK_COLUMNS = """
task_id, user_id, status, query_text, recipe_json, servings,
servings_source, completion_reason, reconciliation_open, version,
last_request_id, created_at, updated_at, completed_at
"""


class TaskStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"


class CompletionReason(StrEnum):
    COOKED = "cooked"
    CONTEXT_CLEARED = "context_cleared"


class ServingsSource(StrEnum):
    USER = "user"
    PROFILE = "profile"
    FALLBACK = "fallback"


class RecipeSnapshot(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=500)
    instructions: str = Field(min_length=1, max_length=20_000)
    ingredients: list[str] = Field(min_length=1, max_length=100)
    source: str = Field(default="", max_length=200)
    source_url: str | None = Field(default=None, max_length=2_000)
    image_url: str | None = Field(default=None, max_length=2_000)
    history_segment_id: str
    query_text: str = Field(min_length=1, max_length=2_000)

    @field_validator("history_segment_id")
    @classmethod
    def validate_history_segment_id(cls, value: str) -> str:
        if re.fullmatch(r"hist_[0-9a-f]{32}", value) is None:
            raise ValueError("recipe history segment ID is invalid")
        return value

    @field_validator("ingredients")
    @classmethod
    def validate_ingredients(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value or "\n" in value or "\r" in value for value in cleaned):
            raise ValueError("recipe ingredients must be non-empty single lines")
        return cleaned


class CookingTask(BaseModel):
    task_id: str
    user_id: str
    status: TaskStatus
    query_text: str
    recipe: RecipeSnapshot
    servings: int = Field(gt=0, le=100)
    servings_source: ServingsSource
    completion_reason: CompletionReason | None = None
    reconciliation_open: bool = False
    version: int = Field(gt=0)
    last_request_id: str | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, value: str) -> str:
        if _TASK_ID.fullmatch(value) is None:
            raise ValueError("cooking task ID is invalid")
        return value


class CookingTaskRepository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cooking_tasks (
                    task_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('active', 'paused', 'completed')
                    ),
                    query_text TEXT NOT NULL,
                    recipe_json TEXT NOT NULL,
                    servings INTEGER NOT NULL CHECK (servings > 0),
                    servings_source TEXT NOT NULL CHECK (
                        servings_source IN ('user', 'profile', 'fallback')
                    ),
                    completion_reason TEXT CHECK (
                        completion_reason IS NULL OR completion_reason IN (
                            'cooked', 'context_cleared'
                        )
                    ),
                    reconciliation_open INTEGER NOT NULL DEFAULT 0 CHECK (
                        reconciliation_open IN (0, 1)
                    ),
                    version INTEGER NOT NULL DEFAULT 1 CHECK (version > 0),
                    last_request_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT
                )
                """
            )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_one_active_task_per_user "
                "ON cooking_tasks(user_id) WHERE status = 'active'"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_user_status_updated "
                "ON cooking_tasks(user_id, status, updated_at DESC)"
            )

    def create_active(
        self,
        *,
        user_id: str,
        query_text: str,
        recipe: RecipeSnapshot,
        servings: int,
        servings_source: ServingsSource,
        request_id: str,
    ) -> CookingTask:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            task = self._insert_active(
                connection,
                user_id=user_id,
                query_text=query_text,
                recipe=recipe,
                servings=servings,
                servings_source=servings_source,
                request_id=request_id,
            )
        return task

    def _insert_active(
        self,
        connection: sqlite3.Connection,
        *,
        user_id: str,
        query_text: str,
        recipe: RecipeSnapshot,
        servings: int,
        servings_source: ServingsSource,
        request_id: str,
    ) -> CookingTask:
        now = datetime.now(UTC)
        task_id = f"task_{uuid4().hex}"
        connection.execute(
            """
            INSERT INTO cooking_tasks(
                task_id, user_id, status, query_text, recipe_json, servings,
                servings_source, completion_reason, reconciliation_open,
                version, last_request_id, created_at, updated_at, completed_at
            ) VALUES (?, ?, 'active', ?, ?, ?, ?, NULL, 0, 1, ?, ?, ?, NULL)
            """,
            (
                task_id,
                user_id,
                query_text.strip(),
                json.dumps(recipe.model_dump(mode="json"), ensure_ascii=False),
                servings,
                servings_source.value,
                _request_id(request_id),
                now.isoformat(),
                now.isoformat(),
            ),
        )
        task = self._get_in(connection, task_id)
        assert task is not None
        return task

    def get(self, task_id: str) -> CookingTask | None:
        _valid_task_id(task_id)
        with sqlite3.connect(self.db_path) as connection:
            return self._get_in(connection, task_id)

    def get_active(self, user_id: str) -> CookingTask | None:
        with sqlite3.connect(self.db_path) as connection:
            return self._get_active_in(connection, user_id)

    def list_available(self, user_id: str) -> list[CookingTask]:
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                f"SELECT {_TASK_COLUMNS} FROM cooking_tasks "
                "WHERE user_id = ? AND (status IN ('active', 'paused') OR "
                "reconciliation_open = 1) ORDER BY updated_at DESC, task_id",
                (user_id,),
            ).fetchall()
        return [_task_from_row(row) for row in rows]

    @staticmethod
    def _get_in(
        connection: sqlite3.Connection, task_id: str
    ) -> CookingTask | None:
        row = connection.execute(
            f"SELECT {_TASK_COLUMNS} FROM cooking_tasks WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        return _task_from_row(row) if row is not None else None

    @staticmethod
    def _get_active_in(
        connection: sqlite3.Connection, user_id: str
    ) -> CookingTask | None:
        row = connection.execute(
            f"SELECT {_TASK_COLUMNS} FROM cooking_tasks "
            "WHERE user_id = ? AND status = 'active'",
            (user_id,),
        ).fetchone()
        return _task_from_row(row) if row is not None else None


class CookingTaskService:
    def __init__(
        self,
        repository: CookingTaskRepository,
        history: ConversationHistoryRepository,
    ) -> None:
        if repository.db_path != history.db_path:
            raise ValueError("task and history repositories must share a database")
        self.repository = repository
        self.history = history

    def confirm_recipe(
        self,
        *,
        user_id: str,
        segment_id: str,
        recipe_id: str,
        servings: int | None,
        default_context: list[str],
        as_new: bool,
        expected_version: int | None,
        request_id: str,
    ) -> CookingTask:
        request_id = _request_id(request_id)
        evidence = self.history.get_evidence(user_id, [segment_id])
        if not evidence:
            raise KeyError("recipe evidence not found")
        snapshot = _recipe_snapshot(evidence[0], recipe_id)
        resolved_servings, source = _resolve_servings(servings, default_context)
        with sqlite3.connect(self.repository.db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            active = self.repository._get_active_in(connection, user_id)
            if active is None:
                return self.repository._insert_active(
                    connection,
                    user_id=user_id,
                    query_text=evidence[0].request,
                    recipe=snapshot,
                    servings=resolved_servings,
                    servings_source=source,
                    request_id=request_id,
                )
            if active.last_request_id == request_id:
                return active
            _expected_version(active, expected_version)
            if as_new:
                now = datetime.now(UTC).isoformat()
                connection.execute(
                    "UPDATE cooking_tasks SET status = 'paused', version = version + 1, "
                    "last_request_id = ?, updated_at = ? WHERE task_id = ?",
                    (request_id, now, active.task_id),
                )
                return self.repository._insert_active(
                    connection,
                    user_id=user_id,
                    query_text=evidence[0].request,
                    recipe=snapshot,
                    servings=resolved_servings,
                    servings_source=source,
                    request_id=request_id,
                )
            now = datetime.now(UTC).isoformat()
            connection.execute(
                """
                UPDATE cooking_tasks
                SET query_text = ?, recipe_json = ?, servings = ?,
                    servings_source = ?, version = version + 1,
                    last_request_id = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (
                    evidence[0].request,
                    json.dumps(snapshot.model_dump(mode="json"), ensure_ascii=False),
                    resolved_servings,
                    source.value,
                    request_id,
                    now,
                    active.task_id,
                ),
            )
            updated = self.repository._get_in(connection, active.task_id)
            assert updated is not None
            return updated

    def update_servings(
        self,
        *,
        user_id: str,
        task_id: str,
        servings: int,
        expected_version: int,
        request_id: str,
    ) -> CookingTask:
        if not 1 <= servings <= 100:
            raise ValueError("servings must be between 1 and 100")
        return self._update_active(
            user_id,
            task_id,
            expected_version,
            request_id,
            "servings = ?, servings_source = 'user'",
            (servings,),
        )

    def activate(
        self,
        *,
        user_id: str,
        task_id: str,
        expected_version: int,
        request_id: str,
    ) -> CookingTask:
        _valid_task_id(task_id)
        request_id = _request_id(request_id)
        with sqlite3.connect(self.repository.db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            target = self.repository._get_in(connection, task_id)
            _owned(target, user_id)
            assert target is not None
            if target.status == TaskStatus.ACTIVE and target.last_request_id == request_id:
                return target
            if target.status != TaskStatus.PAUSED:
                raise ValueError("only a paused task can be activated")
            _expected_version(target, expected_version)
            now = datetime.now(UTC).isoformat()
            connection.execute(
                "UPDATE cooking_tasks SET status = 'paused', version = version + 1, "
                "updated_at = ? WHERE user_id = ? AND status = 'active'",
                (now, user_id),
            )
            connection.execute(
                "UPDATE cooking_tasks SET status = 'active', version = version + 1, "
                "last_request_id = ?, updated_at = ? WHERE task_id = ?",
                (request_id, now, task_id),
            )
            activated = self.repository._get_in(connection, task_id)
            assert activated is not None
            return activated

    def complete_cooked(
        self,
        *,
        user_id: str,
        task_id: str,
        expected_version: int,
        request_id: str,
    ) -> tuple[CookingTask, bool]:
        _valid_task_id(task_id)
        request_id = _request_id(request_id)
        with sqlite3.connect(self.repository.db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            task = self.repository._get_in(connection, task_id)
            _owned(task, user_id)
            assert task is not None
            if (
                task.status == TaskStatus.COMPLETED
                and task.completion_reason == CompletionReason.COOKED
                and task.last_request_id == request_id
            ):
                return task, False
            if task.status != TaskStatus.ACTIVE:
                raise ValueError("only an active task can be completed as cooked")
            _expected_version(task, expected_version)
            now = datetime.now(UTC).isoformat()
            connection.execute(
                """
                UPDATE cooking_tasks
                SET status = 'completed', completion_reason = 'cooked',
                    reconciliation_open = 1, version = version + 1,
                    last_request_id = ?, updated_at = ?, completed_at = ?
                WHERE task_id = ?
                """,
                (request_id, now, now, task_id),
            )
            completed = self.repository._get_in(connection, task_id)
            assert completed is not None
            return completed, True

    def close_reconciliation(
        self,
        *,
        user_id: str,
        task_id: str,
        expected_version: int,
        request_id: str,
    ) -> CookingTask:
        _valid_task_id(task_id)
        request_id = _request_id(request_id)
        with sqlite3.connect(self.repository.db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            task = self.repository._get_in(connection, task_id)
            _owned(task, user_id)
            assert task is not None
            if not task.reconciliation_open and task.last_request_id == request_id:
                return task
            if (
                task.status != TaskStatus.COMPLETED
                or task.completion_reason != CompletionReason.COOKED
                or not task.reconciliation_open
            ):
                raise ValueError("task has no open inventory reconciliation")
            _expected_version(task, expected_version)
            now = datetime.now(UTC).isoformat()
            connection.execute(
                "UPDATE cooking_tasks SET reconciliation_open = 0, "
                "version = version + 1, last_request_id = ?, updated_at = ? "
                "WHERE task_id = ?",
                (request_id, now, task_id),
            )
            closed = self.repository._get_in(connection, task_id)
            assert closed is not None
            return closed

    def get_reconciliation_task(self, user_id: str, task_id: str) -> CookingTask:
        task = self.repository.get(task_id)
        _owned(task, user_id)
        assert task is not None
        if (
            task.status != TaskStatus.COMPLETED
            or task.completion_reason != CompletionReason.COOKED
            or not task.reconciliation_open
        ):
            raise ValueError("task has no open inventory reconciliation")
        return task

    def _update_active(
        self,
        user_id: str,
        task_id: str,
        expected_version: int,
        request_id: str,
        assignment: str,
        parameters: tuple[object, ...],
    ) -> CookingTask:
        _valid_task_id(task_id)
        request_id = _request_id(request_id)
        with sqlite3.connect(self.repository.db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            task = self.repository._get_in(connection, task_id)
            _owned(task, user_id)
            assert task is not None
            if task.last_request_id == request_id:
                return task
            if task.status != TaskStatus.ACTIVE:
                raise ValueError("only an active task can be updated")
            _expected_version(task, expected_version)
            now = datetime.now(UTC).isoformat()
            connection.execute(
                f"UPDATE cooking_tasks SET {assignment}, version = version + 1, "
                "last_request_id = ?, updated_at = ? WHERE task_id = ?",
                (*parameters, request_id, now, task_id),
            )
            updated = self.repository._get_in(connection, task_id)
            assert updated is not None
            return updated


def _task_from_row(row: tuple[object, ...]) -> CookingTask:
    return CookingTask(
        task_id=str(row[0]),
        user_id=str(row[1]),
        status=TaskStatus(str(row[2])),
        query_text=str(row[3]),
        recipe=RecipeSnapshot.model_validate(json.loads(str(row[4]))),
        servings=int(row[5]),
        servings_source=ServingsSource(str(row[6])),
        completion_reason=(CompletionReason(str(row[7])) if row[7] else None),
        reconciliation_open=bool(row[8]),
        version=int(row[9]),
        last_request_id=str(row[10]) if row[10] else None,
        created_at=datetime.fromisoformat(str(row[11])),
        updated_at=datetime.fromisoformat(str(row[12])),
        completed_at=datetime.fromisoformat(str(row[13])) if row[13] else None,
    )


def _recipe_snapshot(record: ConversationRecord, recipe_id: str) -> RecipeSnapshot:
    for entry in record.tool_history:
        if entry.get("tool") != "search_recipes":
            continue
        result = entry.get("result")
        if not isinstance(result, list):
            continue
        for item in result:
            if not isinstance(item, dict) or item.get("id") != recipe_id:
                continue
            return RecipeSnapshot(
                id=str(item.get("id", "")),
                title=str(item.get("title", "")),
                instructions=str(item.get("instructions", "")),
                ingredients=item.get("ingredients", []),
                source=str(item.get("source", "")),
                source_url=(
                    str(item["source_url"]) if item.get("source_url") else None
                ),
                image_url=(str(item["image_url"]) if item.get("image_url") else None),
                history_segment_id=record.segment_id,
                query_text=record.request,
            )
    raise KeyError("recipe not found in persisted search result")


_CHINESE_SERVINGS = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def _resolve_servings(
    explicit: int | None, default_context: list[str]
) -> tuple[int, ServingsSource]:
    if explicit is not None:
        if not 1 <= explicit <= 100:
            raise ValueError("servings must be between 1 and 100")
        return explicit, ServingsSource.USER
    for value in default_context:
        match = re.search(r"(?:默认\s*)?(\d{1,3})\s*(?:人份|份)\b", value, re.I)
        if match is None:
            match = re.search(r"servings\s*[:：]\s*(\d{1,3})", value, re.I)
        if match is not None:
            servings = int(match.group(1))
            if 1 <= servings <= 100:
                return servings, ServingsSource.PROFILE
        chinese = re.search(r"(?:默认\s*)?([一二两三四五六七八九十])\s*人份", value)
        if chinese is not None:
            return _CHINESE_SERVINGS[chinese.group(1)], ServingsSource.PROFILE
    return 1, ServingsSource.FALLBACK


def _valid_task_id(task_id: str) -> None:
    if _TASK_ID.fullmatch(task_id) is None:
        raise ValueError("cooking task ID is invalid")


def _request_id(value: str) -> str:
    cleaned = value.strip()
    if not cleaned or len(cleaned) > 100 or "\n" in cleaned or "\r" in cleaned:
        raise ValueError("request ID is invalid")
    return cleaned


def _owned(task: CookingTask | None, user_id: str) -> None:
    if task is None:
        raise KeyError("cooking task not found")
    if task.user_id != user_id:
        raise PermissionError("cooking task belongs to another user")


def _expected_version(task: CookingTask, expected: int | None) -> None:
    if expected is None or expected != task.version:
        raise ValueError("cooking task version conflict")
