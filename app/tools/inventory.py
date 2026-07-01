"""Inventory business functions backed by a small SQLite database."""

from __future__ import annotations

import sqlite3
from pathlib import Path


class InventoryRepository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS inventory (
                    name TEXT PRIMARY KEY,
                    quantity REAL NOT NULL CHECK (quantity >= 0),
                    unit TEXT NOT NULL
                )
                """
            )

    def list_items(self) -> list[dict[str, object]]:
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                "SELECT name, quantity, unit FROM inventory ORDER BY name"
            ).fetchall()
        return [
            {"name": name, "quantity": quantity, "unit": unit}
            for name, quantity, unit in rows
        ]

    def get_item(self, name: str) -> dict[str, object] | None:
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT name, quantity, unit FROM inventory WHERE name = ?",
                (name.strip(),),
            ).fetchone()
        if row is None:
            return None
        return {"name": row[0], "quantity": row[1], "unit": row[2]}

    def add_item(self, name: str, quantity: float, unit: str) -> dict[str, object]:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        clean_name = name.strip()
        clean_unit = unit.strip()
        if not clean_name or not clean_unit:
            raise ValueError("name and unit are required")
        with sqlite3.connect(self.db_path) as connection:
            existing = connection.execute(
                "SELECT quantity, unit FROM inventory WHERE name = ?", (clean_name,)
            ).fetchone()
            if existing and existing[1] != clean_unit:
                raise ValueError("unit does not match the existing inventory item")
            connection.execute(
                """
                INSERT INTO inventory(name, quantity, unit) VALUES (?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET quantity = quantity + excluded.quantity
                """,
                (clean_name, quantity, clean_unit),
            )
        item = self.get_item(clean_name)
        assert item is not None
        return item

    def update_item(self, name: str, quantity: float, unit: str) -> dict[str, object]:
        if quantity < 0:
            raise ValueError("quantity cannot be negative")
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                "UPDATE inventory SET quantity = ?, unit = ? WHERE name = ?",
                (quantity, unit.strip(), name.strip()),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"inventory item not found: {name}")
        item = self.get_item(name)
        assert item is not None
        return item

    def remove_item(self, name: str, quantity: float | None = None) -> dict[str, object]:
        item = self.get_item(name)
        if item is None:
            raise KeyError(f"inventory item not found: {name}")
        if quantity is None or quantity >= float(item["quantity"]):
            with sqlite3.connect(self.db_path) as connection:
                connection.execute("DELETE FROM inventory WHERE name = ?", (name.strip(),))
            return {**item, "removed": True}
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        remaining = float(item["quantity"]) - quantity
        return self.update_item(name, remaining, str(item["unit"]))
