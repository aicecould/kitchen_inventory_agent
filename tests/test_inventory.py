from datetime import UTC, datetime, timedelta
from pathlib import Path
import sqlite3

import pytest

from app.tools.inventory import InventoryRepository


def test_instances_keep_duplicate_names_and_original_quantity_text(tmp_path: Path) -> None:
    """Removing a uniqueness constraint by name must not merge separate purchases."""
    repository = InventoryRepository(tmp_path / "inventory.db")
    repository.initialize()

    first = repository.add_instance("番茄", "2个", None, "unknown")
    second = repository.add_instance("番茄", "400g-500g", None, "unknown")

    assert first["instance_id"] != second["instance_id"]
    assert repository.find_by_name("番茄") == [first, second]
    assert [item["quantity_text"] for item in repository.list_instances()] == [
        "2个",
        "400g-500g",
    ]


def test_legacy_schema_is_destructively_replaced_once(tmp_path: Path) -> None:
    """A legacy aggregate table must be cleared, while new instance rows survive restart."""
    db = tmp_path / "inventory.db"
    with sqlite3.connect(db) as connection:
        connection.execute(
            "CREATE TABLE inventory(name TEXT PRIMARY KEY, quantity REAL, unit TEXT)"
        )
        connection.execute("INSERT INTO inventory VALUES ('土豆', 2, '个')")
    repository = InventoryRepository(db)
    repository.initialize()
    assert repository.list_instances() == []
    kept = repository.add_instance("土豆", "2个", None, "unknown")
    repository.initialize()
    assert repository.get_instance(str(kept["instance_id"])) == kept


@pytest.mark.parametrize(
    "definition",
    [
        (
            "CREATE TABLE inventory("
            "instance_id TEXT PRIMARY KEY, name TEXT NOT NULL, "
            "quantity_text TEXT NOT NULL, stocked_at TEXT NOT NULL, "
            "expires_at TEXT, expiry_source TEXT NOT NULL, "
            "last_confirmed_at TEXT NOT NULL, deployment_note TEXT)"
        ),
        "CREATE TABLE inventory(quantity REAL, name TEXT PRIMARY KEY, unit TEXT)",
        (
            "CREATE TABLE inventory("
            "name TEXT NOT NULL, instance_id TEXT PRIMARY KEY, "
            "quantity_text TEXT NOT NULL, stocked_at TEXT NOT NULL, "
            "expires_at TEXT, expiry_source TEXT NOT NULL, "
            "last_confirmed_at TEXT NOT NULL)"
        ),
    ],
)
def test_unknown_or_reordered_schema_fails_closed_without_losing_data(
    tmp_path: Path, definition: str
) -> None:
    """Any shape other than exact legacy or v2 must survive a refused migration."""
    database = tmp_path / "inventory.db"
    with sqlite3.connect(database) as connection:
        connection.execute(definition)
        columns = [
            str(row[1])
            for row in connection.execute("PRAGMA table_info(inventory)").fetchall()
        ]
        values: dict[str, object] = {
            "instance_id": "inv_11111111111111111111111111111111",
            "name": "土豆",
            "quantity": 2,
            "quantity_text": "2个",
            "unit": "个",
            "stocked_at": "2026-08-07T00:00:00+00:00",
            "expires_at": None,
            "expiry_source": "unknown",
            "last_confirmed_at": "2026-08-07T00:00:00+00:00",
            "deployment_note": "keep-me",
        }
        placeholders = ", ".join("?" for _ in columns)
        connection.execute(
            f"INSERT INTO inventory({', '.join(columns)}) VALUES ({placeholders})",
            [values[column] for column in columns],
        )

    with pytest.raises(RuntimeError, match="refusing destructive migration"):
        InventoryRepository(database).initialize()

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM inventory").fetchone() == (1,)
        remaining_columns = tuple(
            str(row[1])
            for row in connection.execute("PRAGMA table_info(inventory)").fetchall()
        )
    assert remaining_columns == tuple(columns)


def test_v2_schema_uses_stocked_at_indexes_and_stable_order(tmp_path: Path) -> None:
    """The public v2 contract must be exact and deterministic across insertion order."""
    database = tmp_path / "inventory.db"
    repository = InventoryRepository(database)
    repository.initialize()
    later = datetime(2026, 8, 8, 9, 0, tzinfo=UTC)
    earlier = datetime(2026, 8, 7, 9, 0, tzinfo=UTC)
    pear_later = repository.add_instance("pear", "2个", None, "unknown", now=later)
    apple = repository.add_instance("apple", "1个", None, "unknown", now=later)
    pear_earlier = repository.add_instance("pear", "1个", None, "unknown", now=earlier)

    with sqlite3.connect(database) as connection:
        columns = tuple(
            str(row[1])
            for row in connection.execute("PRAGMA table_info(inventory)").fetchall()
        )
        indexes = {
            str(row[1])
            for row in connection.execute("PRAGMA index_list(inventory)").fetchall()
        }

    assert columns == (
        "instance_id",
        "name",
        "quantity_text",
        "stocked_at",
        "expires_at",
        "expiry_source",
        "last_confirmed_at",
    )
    assert {
        "idx_inventory_name",
        "idx_inventory_expires_at",
        "idx_inventory_last_confirmed_at",
    }.issubset(indexes)
    assert repository.list_instances() == [apple, pear_earlier, pear_later]
    assert set(apple) == {
        "instance_id",
        "name",
        "quantity_text",
        "stocked_at",
        "expires_at",
        "expiry_source",
        "last_confirmed_at",
    }


def test_future_schema_version_fails_closed_and_preserves_v2_rows(tmp_path: Path) -> None:
    """A future version marker must not be normalized back to v2 on read startup."""
    database = tmp_path / "inventory.db"
    repository = InventoryRepository(database)
    repository.initialize()
    kept = repository.add_instance("土豆", "2个", None, "unknown")
    with sqlite3.connect(database) as connection:
        connection.execute("INSERT INTO schema_versions(version) VALUES (3)")

    with pytest.raises(RuntimeError, match="unsupported schema version"):
        repository.initialize()

    assert repository.get_instance(str(kept["instance_id"])) == kept


def test_instance_updates_confirmation_and_removals_return_snapshots(tmp_path: Path) -> None:
    """Instance mutations must affect only their target and expose the removed record."""
    repository = InventoryRepository(tmp_path / "inventory.db")
    repository.initialize()
    now = datetime(2026, 8, 7, 9, 0, tzinfo=UTC)
    first = repository.add_instance("牛奶", "1盒", None, "unknown", now=now)
    second = repository.add_instance("牛奶", "2盒", None, "unknown", now=now)

    replaced = repository.replace_instance(
        str(first["instance_id"]),
        "750ml",
        datetime(2026, 8, 10, 9, 0, tzinfo=UTC),
        "user",
        now=now + timedelta(minutes=1),
    )
    confirmed = repository.confirm_instance(
        str(replaced["instance_id"]), now=now + timedelta(minutes=2)
    )
    removed = repository.remove_instance(str(second["instance_id"]))

    assert replaced["quantity_text"] == "750ml"
    assert replaced["expires_at"] == "2026-08-10T09:00:00+00:00"
    assert confirmed["last_confirmed_at"] == "2026-08-07T09:02:00+00:00"
    assert removed == second
    assert repository.find_by_name("牛奶") == [confirmed]


def test_remove_all_by_name_returns_every_deleted_instance(tmp_path: Path) -> None:
    """Bulk removal must return the concrete rows it deleted, not an aggregate."""
    repository = InventoryRepository(tmp_path / "inventory.db")
    repository.initialize()
    first = repository.add_instance("鸡蛋", "6个", None, "unknown")
    second = repository.add_instance("鸡蛋", "12个", None, "unknown")

    removed = repository.remove_all_by_name("鸡蛋")

    assert removed == {"name": "鸡蛋", "removed_instances": [first, second]}
    assert repository.find_by_name("鸡蛋") == []
    with pytest.raises(KeyError):
        repository.remove_all_by_name("鸡蛋")


def test_reminder_candidates_use_expiry_and_confirmation_boundaries(tmp_path: Path) -> None:
    """Expiry at the three-day cutoff is expiring; fourteen-day-old confirmation is not stale."""
    repository = InventoryRepository(tmp_path / "inventory.db")
    repository.initialize()
    now = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)
    expired = repository.add_instance(
        "过期豆腐", "1盒", now - timedelta(seconds=1), "user", now=now
    )
    expiring = repository.add_instance(
        "酸奶", "2杯", now + timedelta(days=3), "user", now=now
    )
    stale = repository.add_instance("冷冻玉米", "1袋", None, "unknown", now=now)
    fresh = repository.add_instance("米", "1kg", None, "unknown", now=now)
    repository.confirm_instance(str(stale["instance_id"]), now=now - timedelta(days=14, seconds=1))
    repository.confirm_instance(str(fresh["instance_id"]), now=now - timedelta(days=14))

    candidates = repository.reminder_candidates(now)

    assert candidates == {
        "expired": [expired],
        "expiring": [expiring],
        "stale": [repository.get_instance(str(stale["instance_id"]))],
    }


@pytest.mark.parametrize(
"name,quantity_text,expiry_source",
[
    ("", "1个", "unknown"),
    ("\n番茄", "1个", "unknown"),
    ("番茄", "", "unknown"),
    ("番茄", "1\n个", "unknown"),
    ("番茄", "1个", "estimate"),
],
)
def test_instances_reject_invalid_text_and_expiry_source(
    tmp_path: Path, name: str, quantity_text: str, expiry_source: str
) -> None:
    """Malformed labels and untrusted expiry provenance cannot enter inventory."""
    repository = InventoryRepository(tmp_path / "inventory.db")
    repository.initialize()

    with pytest.raises(ValueError):
        repository.add_instance(name, quantity_text, None, expiry_source)
