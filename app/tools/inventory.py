"""Instance-level kitchen inventory backed by SQLite."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
import re
import sqlite3
from uuid import uuid4


_INVENTORY_COLUMNS = (
    "instance_id",
    "name",
    "quantity_text",
    "stocked_at",
    "expires_at",
    "expiry_source",
    "last_confirmed_at",
)
_INVENTORY_SELECT = ", ".join(_INVENTORY_COLUMNS)
_EXPIRY_SOURCES = {"llm", "user", "unknown"}
_LEGACY_INVENTORY_SQL = """
CREATE TABLE inventory (
    name TEXT PRIMARY KEY,
    quantity REAL NOT NULL CHECK (quantity >= 0),
    unit TEXT NOT NULL
)
"""
_INVENTORY_SQL = """
CREATE TABLE inventory (
    instance_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    quantity_text TEXT NOT NULL,
    stocked_at TEXT NOT NULL,
    expires_at TEXT,
    expiry_source TEXT NOT NULL,
    last_confirmed_at TEXT NOT NULL
)
"""
_SCHEMA_VERSIONS_SQL = """
CREATE TABLE schema_versions (version INTEGER PRIMARY KEY)
"""
_LEGACY_INVENTORY_SIGNATURE = (
    ("name", "TEXT", 0, None, 1, 0),
    ("quantity", "REAL", 1, None, 0, 0),
    ("unit", "TEXT", 1, None, 0, 0),
)
_INVENTORY_SIGNATURE = (
    ("instance_id", "TEXT", 0, None, 1, 0),
    ("name", "TEXT", 1, None, 0, 0),
    ("quantity_text", "TEXT", 1, None, 0, 0),
    ("stocked_at", "TEXT", 1, None, 0, 0),
    ("expires_at", "TEXT", 0, None, 0, 0),
    ("expiry_source", "TEXT", 1, None, 0, 0),
    ("last_confirmed_at", "TEXT", 1, None, 0, 0),
)
_SCHEMA_VERSIONS_SIGNATURE = (("version", "INTEGER", 0, None, 1, 0),)


class InventorySchemaError(RuntimeError):
    """Raised when an inventory database is not a known safe schema state."""


class InventoryRepository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            inventory_schema = self._table_schema(connection, "inventory")
            version_schema = self._table_schema(connection, "schema_versions")

            inventory_kind = self._inventory_schema_kind(inventory_schema)
            version_schema_is_exact = self._schema_matches(
                version_schema,
                _SCHEMA_VERSIONS_SIGNATURE,
                _SCHEMA_VERSIONS_SQL,
            )
            if version_schema is not None and not version_schema_is_exact:
                raise InventorySchemaError(
                    "unknown schema_versions shape; refusing destructive migration"
                )
            versions: set[int] | None = None
            if version_schema_is_exact:
                versions = {
                    int(row[0])
                    for row in connection.execute(
                        "SELECT version FROM schema_versions"
                    ).fetchall()
                }
                unsupported = versions - {2}
                if unsupported:
                    rendered = ", ".join(str(value) for value in sorted(unsupported))
                    raise InventorySchemaError(
                        f"unsupported schema version(s): {rendered}"
                    )

            if inventory_kind == "unknown":
                rendered = ", ".join(
                    column[0] for column in inventory_schema[0]
                ) if inventory_schema is not None else "missing"
                raise InventorySchemaError(
                    "unknown inventory schema "
                    f"({rendered}); refusing destructive migration"
                )

            if inventory_kind == "legacy":
                if version_schema is not None:
                    raise InventorySchemaError(
                        "contradictory schema state: legacy inventory has a "
                        "schema version table; refusing destructive migration"
                    )
                connection.execute("DROP TABLE inventory")
            elif inventory_kind == "v2":
                if versions != {2}:
                    raise InventorySchemaError(
                        "existing v2 inventory requires a coherent version 2 marker"
                    )
            elif version_schema is not None:
                raise InventorySchemaError(
                    "contradictory schema state: schema version table exists without "
                    "inventory; refusing initialization"
                )

            if inventory_kind in {"missing", "legacy"}:
                connection.execute(_SCHEMA_VERSIONS_SQL)
                connection.execute(_INVENTORY_SQL)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_inventory_name ON inventory(name)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_inventory_expires_at "
                "ON inventory(expires_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_inventory_last_confirmed_at "
                "ON inventory(last_confirmed_at)"
            )
            connection.execute("INSERT OR IGNORE INTO schema_versions(version) VALUES (2)")

    @classmethod
    def _inventory_schema_kind(
        cls,
        schema: tuple[tuple[tuple[object, ...], ...], str] | None,
    ) -> str:
        if schema is None:
            return "missing"
        if cls._schema_matches(
            schema, _LEGACY_INVENTORY_SIGNATURE, _LEGACY_INVENTORY_SQL
        ):
            return "legacy"
        if cls._schema_matches(schema, _INVENTORY_SIGNATURE, _INVENTORY_SQL):
            return "v2"
        return "unknown"

    @classmethod
    def _schema_matches(
        cls,
        actual: tuple[tuple[tuple[object, ...], ...], str] | None,
        expected_signature: tuple[tuple[object, ...], ...],
        expected_sql: str,
    ) -> bool:
        if actual is None:
            return False
        signature, table_sql = actual
        return signature == expected_signature and table_sql == cls._normalize_sql(
            expected_sql
        )

    @classmethod
    def _table_schema(
        cls, connection: sqlite3.Connection, table: str
    ) -> tuple[tuple[tuple[object, ...], ...], str] | None:
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        if row is None or row[0] is None:
            return None
        pragma_rows = connection.execute(f"PRAGMA table_xinfo({table})").fetchall()
        signature = tuple(
            (
                str(column[1]),
                " ".join(str(column[2]).upper().split()),
                int(column[3]),
                None if column[4] is None else str(column[4]).strip(),
                int(column[5]),
                int(column[6]),
            )
            for column in pragma_rows
        )
        return signature, cls._normalize_sql(str(row[0]))

    @staticmethod
    def _normalize_sql(sql: str) -> str:
        return re.sub(r"\s+", "", sql).casefold()

    def list_instances(self) -> list[dict[str, object]]:
        with self._connection() as connection:
            rows = connection.execute(
                f"SELECT {_INVENTORY_SELECT} FROM inventory "
                "ORDER BY name, stocked_at, instance_id"
            ).fetchall()
        return [self._snapshot(row) for row in rows]

    def find_by_name(self, name: str) -> list[dict[str, object]]:
        with self._connection() as connection:
            rows = connection.execute(
                f"SELECT {_INVENTORY_SELECT} FROM inventory WHERE name = ? "
                "ORDER BY name, stocked_at, instance_id",
                (name,),
            ).fetchall()
        return [self._snapshot(row) for row in rows]

    def get_instance(self, instance_id: str) -> dict[str, object] | None:
        with self._connection() as connection:
            row = connection.execute(
                f"SELECT {_INVENTORY_SELECT} FROM inventory WHERE instance_id = ?",
                (instance_id,),
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
            "stocked_at": timestamp,
            "expires_at": self._serialize_datetime(expires_at),
            "expiry_source": expiry_source,
            "last_confirmed_at": timestamp,
        }
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO inventory(
                    instance_id, name, quantity_text, stocked_at, expires_at,
                    expiry_source, last_confirmed_at
                ) VALUES (
                    :instance_id, :name, :quantity_text, :stocked_at, :expires_at,
                    :expiry_source, :last_confirmed_at
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
        if now is not None:
            self._as_utc(now)
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE inventory
                SET quantity_text = ?, expires_at = ?, expiry_source = ?
                WHERE instance_id = ?
                """,
                (
                    quantity_text,
                    self._serialize_datetime(expires_at),
                    expiry_source,
                    instance_id,
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"inventory instance not found: {instance_id}")
            row = connection.execute(
                f"SELECT {_INVENTORY_SELECT} FROM inventory WHERE instance_id = ?",
                (instance_id,),
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
                SET last_confirmed_at = ?
                WHERE instance_id = ?
                """,
                (timestamp, instance_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"inventory instance not found: {instance_id}")
            row = connection.execute(
                f"SELECT {_INVENTORY_SELECT} FROM inventory WHERE instance_id = ?",
                (instance_id,),
            ).fetchone()
        assert row is not None
        return self._snapshot(row)

    def remove_instance(self, instance_id: str) -> dict[str, object]:
        with self._connection() as connection:
            row = connection.execute(
                f"SELECT {_INVENTORY_SELECT} FROM inventory WHERE instance_id = ?",
                (instance_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"inventory instance not found: {instance_id}")
            connection.execute("DELETE FROM inventory WHERE instance_id = ?", (instance_id,))
        return self._snapshot(row)

    def remove_all_by_name(self, name: str) -> dict[str, object]:
        with self._connection() as connection:
            rows = connection.execute(
                f"SELECT {_INVENTORY_SELECT} FROM inventory WHERE name = ? "
                "ORDER BY name, stocked_at, instance_id",
                (name,),
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
