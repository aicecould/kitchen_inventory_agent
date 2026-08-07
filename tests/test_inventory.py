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
