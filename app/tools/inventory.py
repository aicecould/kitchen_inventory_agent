"""Instance-level kitchen inventory backed by SQLite."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
import sqlite3
from uuid import uuid4


_INVENTORY_COLUMNS = {
    "instance_id",
    "name",
    "quantity_text",
    "expires_at",
    "expiry_source",
    "created_at",
    "updated_at",
    "last_confirmed_at",
}
_EXPIRY_SOURCES = {"llm", "user", "unknown"}


class InventoryRepository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(inventory)").fetchall()
            }
            if columns and columns != _INVENTORY_COLUMNS:
                connection.execute("DROP TABLE inventory")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_versions (version INTEGER PRIMARY KEY)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS inventory (
                    instance_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    quantity_text TEXT NOT NULL,
                    expires_at TEXT,
                    expiry_source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_confirmed_at TEXT NOT NULL
                )
                """
            )
            connection.execute("INSERT OR IGNORE INTO schema_versions(version) VALUES (2)")

    def list_instances(self) -> list[dict[str, object]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM inventory ORDER BY rowid"
            ).fetchall()
        return [self._snapshot(row) for row in rows]

    def find_by_name(self, name: str) -> list[dict[str, object]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM inventory WHERE name = ? ORDER BY rowid", (name,)
            ).fetchall()
        return [self._snapshot(row) for row in rows]

    def get_instance(self, instance_id: str) -> dict[str, object] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM inventory WHERE instance_id = ?", (instance_id,)
            ).fetchone()
        return None if row is None else self._snapshot(row)

    def add_instance(
        self,
        name: str,
        quantity_text: str,
        expires_at: datetime | None,
        expiry_source: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, object]:
        self._validate_text(name, "name")
        self._validate_text(quantity_text, "quantity_text")
        self._validate_expiry_source(expiry_source)
        timestamp = self._timestamp(now)
        snapshot = {
            "instance_id": f"inv_{uuid4().hex}",
            "name": name,
            "quantity_text": quantity_text,
            "expires_at": self._serialize_datetime(expires_at),
            "expiry_source": expiry_source,
            "created_at": timestamp,
            "updated_at": timestamp,
            "last_confirmed_at": timestamp,
        }
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO inventory VALUES (
                    :instance_id, :name, :quantity_text, :expires_at, :expiry_source,
                    :created_at, :updated_at, :last_confirmed_at
                )
                """,
                snapshot,
            )
        return snapshot

    def replace_instance(
        self,
        instance_id: str,
        quantity_text: str,
        expires_at: datetime | None,
        expiry_source: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, object]:
        self._validate_text(quantity_text, "quantity_text")
        self._validate_expiry_source(expiry_source)
        updated_at = self._timestamp(now)
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE inventory
                SET quantity_text = ?, expires_at = ?, expiry_source = ?, updated_at = ?
                WHERE instance_id = ?
                """,
                (
                    quantity_text,
                    self._serialize_datetime(expires_at),
                    expiry_source,
                    updated_at,
                    instance_id,
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"inventory instance not found: {instance_id}")
            row = connection.execute(
                "SELECT * FROM inventory WHERE instance_id = ?", (instance_id,)
            ).fetchone()
        assert row is not None
        return self._snapshot(row)

    def confirm_instance(
        self, instance_id: str, *, now: datetime | None = None
    ) -> dict[str, object]:
        timestamp = self._timestamp(now)
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE inventory
                SET last_confirmed_at = ?, updated_at = ?
                WHERE instance_id = ?
                """,
                (timestamp, timestamp, instance_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"inventory instance not found: {instance_id}")
            row = connection.execute(
                "SELECT * FROM inventory WHERE instance_id = ?", (instance_id,)
            ).fetchone()
        assert row is not None
        return self._snapshot(row)

    def remove_instance(self, instance_id: str) -> dict[str, object]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM inventory WHERE instance_id = ?", (instance_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"inventory instance not found: {instance_id}")
            connection.execute("DELETE FROM inventory WHERE instance_id = ?", (instance_id,))
        return self._snapshot(row)

    def remove_all_by_name(self, name: str) -> dict[str, object]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM inventory WHERE name = ? ORDER BY rowid", (name,)
            ).fetchall()
            if not rows:
                raise KeyError(f"inventory instances not found: {name}")
            connection.execute("DELETE FROM inventory WHERE name = ?", (name,))
        return {"name": name, "removed_instances": [self._snapshot(row) for row in rows]}

    def reminder_candidates(self, now: datetime) -> dict[str, list[dict[str, object]]]:
        reference = self._as_utc(now)
        candidates: dict[str, list[dict[str, object]]] = {
            "expired": [],
            "expiring": [],
            "stale": [],
        }
        for snapshot in self.list_instances():
            expires_at = self._deserialize_datetime(snapshot["expires_at"])
            last_confirmed_at = self._deserialize_datetime(snapshot["last_confirmed_at"])
            if expires_at is not None and expires_at < reference:
                candidates["expired"].append(snapshot)
            elif expires_at is not None and reference <= expires_at <= reference + timedelta(days=3):
                candidates["expiring"].append(snapshot)
            if last_confirmed_at < reference - timedelta(days=14):
                candidates["stale"].append(snapshot)
        return candidates

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _snapshot(row: sqlite3.Row) -> dict[str, object]:
        return dict(row)

    @staticmethod
    def _validate_text(value: str, field: str) -> None:
        if not value.strip() or "\n" in value or "\r" in value:
            raise ValueError(f"{field} must be non-empty and single-line")

    @staticmethod
    def _validate_expiry_source(expiry_source: str) -> None:
        if expiry_source not in _EXPIRY_SOURCES:
            raise ValueError("expiry_source must be llm, user, or unknown")

    @classmethod
    def _timestamp(cls, now: datetime | None) -> str:
        return cls._serialize_datetime(now or datetime.now(UTC))

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("datetime values must be timezone-aware")
        return value.astimezone(UTC)

    @classmethod
    def _serialize_datetime(cls, value: datetime | None) -> str | None:
        return None if value is None else cls._as_utc(value).isoformat()

    @staticmethod
    def _deserialize_datetime(value: object) -> datetime | None:
        return None if value is None else datetime.fromisoformat(str(value))
