"""Two-phase, allowlisted execution for inventory mutations."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.tools.inventory import InventoryRepository

ActionStatus = Literal[
    "pending", "executing", "executed", "cancelled", "expired", "failed"
]


class AddInventoryArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=100)
    quantity: float = Field(gt=0)
    unit: str = Field(min_length=1, max_length=20)


class UpdateInventoryArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=100)
    quantity: float = Field(ge=0)
    unit: str = Field(min_length=1, max_length=20)


class RemoveInventoryArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=100)
    quantity: float | None = Field(default=None, gt=0)


class PendingAction(BaseModel):
    action_id: str
    user_id: str
    operation: str
    arguments: dict[str, object]
    summary: str
    status: ActionStatus
    created_at: datetime
    expires_at: datetime
    result: dict[str, object] | None = None
    error: str | None = None


ARGUMENT_MODELS = {
    "inventory.add": AddInventoryArguments,
    "inventory.update": UpdateInventoryArguments,
    "inventory.remove": RemoveInventoryArguments,
}

WRITE_OPERATION_ALLOWLIST = frozenset(ARGUMENT_MODELS)


def validate_operation(operation: str, arguments: dict[str, object]) -> BaseModel:
    model = ARGUMENT_MODELS.get(operation)
    if model is None:
        raise ValueError(f"Operation is not allowlisted: {operation}")
    return model.model_validate(arguments)


def action_summary(operation: str, arguments: BaseModel) -> str:
    values = arguments.model_dump()
    if operation == "inventory.add":
        return f"向库存添加 {values['quantity']:g} {values['unit']} {values['name']}"
    if operation == "inventory.update":
        return f"将 {values['name']} 的库存设置为 {values['quantity']:g} {values['unit']}"
    quantity = values.get("quantity")
    if quantity is None:
        return f"从库存删除 {values['name']}"
    return f"从库存减少 {quantity:g} {values['name']}"


class PendingActionRepository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS pending_actions (
                    action_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_pending_actions_user_status "
                "ON pending_actions(user_id, status)"
            )

    def create(
        self,
        *,
        user_id: str,
        operation: str,
        arguments: dict[str, object],
        summary: str,
        ttl_minutes: int,
    ) -> PendingAction:
        now = datetime.now(timezone.utc)
        action = PendingAction(
            action_id=f"act_{uuid4().hex}",
            user_id=user_id,
            operation=operation,
            arguments=arguments,
            summary=summary,
            status="pending",
            created_at=now,
            expires_at=now + timedelta(minutes=ttl_minutes),
        )
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO pending_actions(
                    action_id, user_id, operation, arguments_json, summary,
                    status, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    action.action_id,
                    action.user_id,
                    action.operation,
                    json.dumps(action.arguments, ensure_ascii=False),
                    action.summary,
                    action.status,
                    action.created_at.isoformat(),
                    action.expires_at.isoformat(),
                ),
            )
        return action

    def get(self, action_id: str) -> PendingAction | None:
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT * FROM pending_actions WHERE action_id = ?", (action_id,)
            ).fetchone()
        return self._from_row(row) if row else None

    def list_pending(self, user_id: str) -> list[PendingAction]:
        self.expire_old_actions()
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                "SELECT * FROM pending_actions WHERE user_id = ? AND status = 'pending' "
                "ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def claim(self, action_id: str, user_id: str) -> tuple[PendingAction, bool]:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM pending_actions WHERE action_id = ?", (action_id,)
            ).fetchone()
            if row is None:
                raise KeyError("Pending action not found")
            action = self._from_row(row)
            if action.user_id != user_id:
                raise PermissionError("Pending action belongs to another user")
            if action.status == "executed":
                return action, False
            if action.status != "pending":
                raise ValueError(f"Action cannot be confirmed from status: {action.status}")
            if action.expires_at <= datetime.now(timezone.utc):
                connection.execute(
                    "UPDATE pending_actions SET status = 'expired' WHERE action_id = ?",
                    (action_id,),
                )
                raise ValueError("Pending action has expired")
            connection.execute(
                "UPDATE pending_actions SET status = 'executing' "
                "WHERE action_id = ? AND status = 'pending'",
                (action_id,),
            )
        action.status = "executing"
        return action, True

    def complete(self, action_id: str, result: dict[str, object]) -> PendingAction:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "UPDATE pending_actions SET status = 'executed', result_json = ? "
                "WHERE action_id = ? AND status = 'executing'",
                (json.dumps(result, ensure_ascii=False), action_id),
            )
        action = self.get(action_id)
        assert action is not None
        return action

    def fail(self, action_id: str, error: str) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "UPDATE pending_actions SET status = 'failed', error = ? "
                "WHERE action_id = ? AND status = 'executing'",
                (error[:500], action_id),
            )

    def cancel(self, action_id: str, user_id: str) -> PendingAction:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM pending_actions WHERE action_id = ?", (action_id,)
            ).fetchone()
            if row is None:
                raise KeyError("Pending action not found")
            action = self._from_row(row)
            if action.user_id != user_id:
                raise PermissionError("Pending action belongs to another user")
            if action.status != "pending":
                raise ValueError(f"Action cannot be cancelled from status: {action.status}")
            connection.execute(
                "UPDATE pending_actions SET status = 'cancelled' WHERE action_id = ?",
                (action_id,),
            )
        action.status = "cancelled"
        return action

    def expire_old_actions(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "UPDATE pending_actions SET status = 'expired' "
                "WHERE status = 'pending' AND expires_at <= ?",
                (now,),
            )

    @staticmethod
    def _from_row(row: tuple[object, ...]) -> PendingAction:
        return PendingAction(
            action_id=str(row[0]),
            user_id=str(row[1]),
            operation=str(row[2]),
            arguments=json.loads(str(row[3])),
            summary=str(row[4]),
            status=str(row[5]),  # type: ignore[arg-type]
            created_at=datetime.fromisoformat(str(row[6])),
            expires_at=datetime.fromisoformat(str(row[7])),
            result=json.loads(str(row[8])) if row[8] else None,
            error=str(row[9]) if row[9] else None,
        )


class InventoryActionService:
    def __init__(
        self,
        repository: PendingActionRepository,
        inventory: InventoryRepository,
        ttl_minutes: int = 15,
    ) -> None:
        self.repository = repository
        self.inventory = inventory
        self.ttl_minutes = ttl_minutes

    def propose(
        self, user_id: str, operation: str, arguments: dict[str, object]
    ) -> PendingAction:
        validated = validate_operation(operation, arguments)
        return self.repository.create(
            user_id=user_id,
            operation=operation,
            arguments=validated.model_dump(),
            summary=action_summary(operation, validated),
            ttl_minutes=self.ttl_minutes,
        )

    def confirm(self, action_id: str, user_id: str) -> PendingAction:
        action, should_execute = self.repository.claim(action_id, user_id)
        if not should_execute:
            return action
        try:
            arguments = validate_operation(action.operation, action.arguments)
            result = self._execute(action.operation, arguments)
            return self.repository.complete(action_id, result)
        except Exception as exc:
            self.repository.fail(action_id, str(exc))
            raise

    def cancel(self, action_id: str, user_id: str) -> PendingAction:
        return self.repository.cancel(action_id, user_id)

    def _execute(self, operation: str, arguments: BaseModel) -> dict[str, object]:
        values = arguments.model_dump()
        if operation == "inventory.add":
            return self.inventory.add_item(**values)  # type: ignore[arg-type]
        if operation == "inventory.update":
            return self.inventory.update_item(**values)  # type: ignore[arg-type]
        if operation == "inventory.remove":
            return self.inventory.remove_item(**values)  # type: ignore[arg-type]
        raise ValueError(f"Operation is not allowlisted: {operation}")
