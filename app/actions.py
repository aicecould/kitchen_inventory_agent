"""Two-phase, allowlisted execution for inventory mutations."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.expiry import ExpiryInferer
from app.tools.inventory import InventoryRepository

ActionStatus = Literal[
    "pending", "executing", "executed", "cancelled", "expired", "failed"
]


class AddInventoryArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=100)
    quantity_text: str = Field(min_length=1, max_length=100)
    expires_at: datetime | None
    expiry_source: Literal["llm", "user", "unknown"]

    @field_validator("expires_at")
    @classmethod
    def expires_at_must_be_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("expires_at must be timezone-aware")
        return value


class ReplaceInventoryArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instance_id: str = Field(min_length=1, max_length=100)
    quantity_text: str = Field(min_length=1, max_length=100)
    expires_at: datetime | None
    expiry_source: Literal["llm", "user", "unknown"]

    @field_validator("expires_at")
    @classmethod
    def expires_at_must_be_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("expires_at must be timezone-aware")
        return value


class RemoveInventoryArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instance_id: str = Field(min_length=1, max_length=100)


class RemoveAllInventoryArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=100)


class ConfirmInventoryArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instance_id: str = Field(min_length=1, max_length=100)


class PendingAction(BaseModel):
    action_id: str
    user_id: str
    operation: str
    arguments: dict[str, object]
    summary: str
    preview: dict[str, object] = Field(default_factory=dict)
    status: ActionStatus
    created_at: datetime
    expires_at: datetime
    result: dict[str, object] | None = None
    error: str | None = None


ARGUMENT_MODELS = {
    "inventory.add": AddInventoryArguments,
    "inventory.replace": ReplaceInventoryArguments,
    "inventory.remove": RemoveInventoryArguments,
    "inventory.remove_all": RemoveAllInventoryArguments,
    "inventory.confirm": ConfirmInventoryArguments,
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
        return f"向库存添加 {values['quantity_text']} {values['name']}"
    if operation == "inventory.replace":
        return f"更新库存项 {values['instance_id']}"
    if operation == "inventory.remove":
        return f"从库存删除库存项 {values['instance_id']}"
    if operation == "inventory.remove_all":
        return f"从库存删除所有 {values['name']}"
    if operation == "inventory.confirm":
        return f"确认库存项 {values['instance_id']}"
    raise ValueError(f"Operation is not allowlisted: {operation}")


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
                    error TEXT,
                    preview_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(pending_actions)"
                ).fetchall()
            }
            if "preview_json" not in columns:
                connection.execute(
                    "ALTER TABLE pending_actions ADD COLUMN preview_json "
                    "TEXT NOT NULL DEFAULT '{}'"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_pending_actions_user_status "
                "ON pending_actions(user_id, status)"
            )
            legacy_rows = connection.execute(
                "SELECT action_id, arguments_json FROM pending_actions "
                "WHERE operation LIKE 'inventory.%' AND status = 'pending'"
            ).fetchall()
            legacy_action_ids = [
                str(action_id)
                for action_id, arguments_json in legacy_rows
                if {"quantity", "unit"} & set(json.loads(str(arguments_json)))
            ]
            connection.executemany(
                "DELETE FROM pending_actions WHERE action_id = ?",
                [(action_id,) for action_id in legacy_action_ids],
            )

    def create(
        self,
        *,
        user_id: str,
        operation: str,
        arguments: dict[str, object],
        summary: str,
        preview: dict[str, object],
        ttl_minutes: int,
    ) -> PendingAction:
        now = datetime.now(timezone.utc)
        action = PendingAction(
            action_id=f"act_{uuid4().hex}",
            user_id=user_id,
            operation=operation,
            arguments=arguments,
            summary=summary,
            preview=preview,
            status="pending",
            created_at=now,
            expires_at=now + timedelta(minutes=ttl_minutes),
        )
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO pending_actions(
                    action_id, user_id, operation, arguments_json, summary,
                    status, created_at, expires_at, preview_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    json.dumps(action.preview, ensure_ascii=False),
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
            preview=json.loads(str(row[10])) if len(row) > 10 and row[10] else {},
        )


class InventoryActionService:
    def __init__(
        self,
        repository: PendingActionRepository,
        inventory: InventoryRepository,
        ttl_minutes: int = 15,
        *,
        expiry_inferer: ExpiryInferer | None = None,
    ) -> None:
        self.repository = repository
        self.inventory = inventory
        self.expiry_inferer = expiry_inferer
        self.ttl_minutes = ttl_minutes

    def propose_add(
        self,
        user_id: str,
        name: str,
        quantity_text: str,
        *,
        user_expires_at: datetime | None = None,
    ) -> PendingAction:
        if user_expires_at is not None:
            expires_at = user_expires_at
            expiry_source = "user"
        else:
            now = datetime.now(timezone.utc)
            try:
                inferred = (
                    self.expiry_inferer.infer(name, now)
                    if self.expiry_inferer is not None
                    else None
                )
            except Exception:
                inferred = None
            if (
                inferred is not None
                and inferred.tzinfo is not None
                and inferred.utcoffset() is not None
                and inferred > now
            ):
                expires_at = inferred
                expiry_source = "llm"
            else:
                expires_at = None
                expiry_source = "unknown"

        return self.propose(
            user_id,
            "inventory.add",
            {
                "name": name,
                "quantity_text": quantity_text,
                "expires_at": expires_at.isoformat() if expires_at is not None else None,
                "expiry_source": expiry_source,
            },
        )

    def propose(
        self, user_id: str, operation: str, arguments: dict[str, object]
    ) -> PendingAction:
        validated = validate_operation(operation, arguments)
        preview = self._build_preview(operation, validated)
        return self.repository.create(
            user_id=user_id,
            operation=operation,
            arguments=validated.model_dump(mode="json"),
            summary=action_summary(operation, validated),
            preview=preview,
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

    def _build_preview(
        self, operation: str, arguments: BaseModel
    ) -> dict[str, object]:
        values = arguments.model_dump(mode="json")
        if operation == "inventory.add":
            return {"proposed": values}
        if operation in {
            "inventory.replace",
            "inventory.remove",
            "inventory.confirm",
        }:
            instance_id = str(values["instance_id"])
            current = self.inventory.get_instance(instance_id)
            if current is None:
                raise KeyError(f"inventory instance not found: {instance_id}")
            if operation == "inventory.replace":
                return {
                    "current": current,
                    "proposed": {
                        "instance_id": instance_id,
                        "name": current["name"],
                        "quantity_text": values["quantity_text"],
                        "expires_at": values["expires_at"],
                        "expiry_source": values["expiry_source"],
                    },
                }
            return {"current": current}
        if operation == "inventory.remove_all":
            name = str(values["name"])
            items = self.inventory.find_by_name(name)
            if not items:
                raise KeyError(f"inventory instances not found: {name}")
            return {"name": name, "items": items, "expected_count": len(items)}
        raise ValueError(f"Operation is not allowlisted: {operation}")

    def _execute(self, operation: str, arguments: BaseModel) -> dict[str, object]:
        values = arguments.model_dump(mode="python")
        if operation == "inventory.add":
            return self.inventory.add_instance(**values)  # type: ignore[arg-type]
        if operation == "inventory.replace":
            return self.inventory.replace_instance(**values)  # type: ignore[arg-type]
        if operation == "inventory.remove":
            return self.inventory.remove_instance(**values)  # type: ignore[arg-type]
        if operation == "inventory.remove_all":
            removed = self.inventory.remove_all_by_name(**values)  # type: ignore[arg-type]
            items = removed["removed_instances"]
            assert isinstance(items, list)
            return {"name": values["name"], "removed_count": len(items), "items": items}
        if operation == "inventory.confirm":
            return self.inventory.confirm_instance(**values)  # type: ignore[arg-type]
        raise ValueError(f"Operation is not allowlisted: {operation}")
