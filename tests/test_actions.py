from datetime import UTC, datetime
from pathlib import Path
import sqlite3

import pytest
from pydantic import ValidationError

from app.actions import (
    InventoryActionService,
    PendingActionRepository,
    validate_operation,
)
from app.tools.inventory import InventoryRepository


def build_service(tmp_path: Path) -> tuple[InventoryActionService, InventoryRepository]:
    inventory = InventoryRepository(tmp_path / "inventory.db")
    inventory.initialize()
    pending = PendingActionRepository(tmp_path / "inventory.db")
    pending.initialize()
    return InventoryActionService(pending, inventory), inventory


def test_add_is_delayed_and_idempotent_when_confirmed(tmp_path: Path) -> None:
    service, inventory = build_service(tmp_path)
    action = service.propose(
        "user-1",
        "inventory.add",
        {
            "name": "番茄",
            "quantity_text": "2 个",
            "expires_at": "2026-08-10T09:00:00+00:00",
            "expiry_source": "user",
        },
    )

    assert inventory.find_by_name("番茄") == []
    assert action.arguments["expires_at"] == "2026-08-10T09:00:00Z"

    executed = service.confirm(action.action_id, "user-1")
    repeated = service.confirm(action.action_id, "user-1")

    assert executed.status == repeated.status == "executed"
    assert len(inventory.find_by_name("番茄")) == 1
    assert inventory.find_by_name("番茄")[0]["quantity_text"] == "2 个"


def test_replace_replaces_the_identified_instance(tmp_path: Path) -> None:
    service, inventory = build_service(tmp_path)
    item = inventory.add_instance("牛奶", "1 盒", None, "unknown")

    action = service.propose(
        "user-1",
        "inventory.replace",
        {
            "instance_id": item["instance_id"],
            "quantity_text": "2 盒",
            "expires_at": "2026-08-12T09:00:00+00:00",
            "expiry_source": "llm",
        },
    )

    assert str(item["instance_id"])[:8] in action.summary
    assert inventory.get_instance(str(item["instance_id"]))["quantity_text"] == "1 盒"  # type: ignore[index]

    executed = service.confirm(action.action_id, "user-1")

    assert executed.result is not None
    assert executed.result["quantity_text"] == "2 盒"
    assert inventory.get_instance(str(item["instance_id"]))["expiry_source"] == "llm"  # type: ignore[index]


def test_remove_deletes_the_identified_instance(tmp_path: Path) -> None:
    service, inventory = build_service(tmp_path)
    item = inventory.add_instance("鸡蛋", "6 个", None, "unknown")
    action = service.propose(
        "user-1", "inventory.remove", {"instance_id": item["instance_id"]}
    )

    assert str(item["instance_id"])[:8] in action.summary
    assert inventory.get_instance(str(item["instance_id"])) is not None

    service.confirm(action.action_id, "user-1")

    assert inventory.get_instance(str(item["instance_id"])) is None


def test_remove_all_stays_pending_then_removes_exact_name(tmp_path: Path) -> None:
    service, inventory = build_service(tmp_path)
    items = [
        inventory.add_instance("番茄", "2 个", None, "unknown"),
        inventory.add_instance("番茄", "3 个", None, "unknown"),
    ]
    inventory.add_instance("小番茄", "1 盒", None, "unknown")
    action = service.propose("user-1", "inventory.remove_all", {"name": "番茄"})

    assert len(inventory.find_by_name("番茄")) == 2

    executed = service.confirm(action.action_id, "user-1")

    assert executed.result == {"name": "番茄", "removed_count": 2, "items": items}
    assert inventory.find_by_name("番茄") == []
    assert len(inventory.find_by_name("小番茄")) == 1


def test_confirm_only_calibrates_the_identified_instance(tmp_path: Path) -> None:
    service, inventory = build_service(tmp_path)
    item = inventory.add_instance("黄瓜", "1 根", None, "unknown")
    action = service.propose(
        "user-1", "inventory.confirm", {"instance_id": item["instance_id"]}
    )

    executed = service.confirm(action.action_id, "user-1")

    assert executed.result is not None
    assert executed.result["quantity_text"] == "1 根"
    assert executed.result["last_confirmed_at"] >= item["last_confirmed_at"]
    assert str(item["instance_id"])[:8] in action.summary


@pytest.mark.parametrize(
    ("operation", "arguments"),
    [
        (
            "inventory.add",
            {
                "name": "番茄",
                "quantity_text": "2 个",
                "expires_at": None,
                "expiry_source": "unknown",
                "unit": "个",
            },
        ),
        ("inventory.remove", {"instance_id": "inv_missing", "name": "番茄"}),
    ],
)
def test_operation_arguments_reject_extra_fields(
    operation: str, arguments: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        validate_operation(operation, arguments)


@pytest.mark.parametrize(
    ("operation", "arguments"),
    [
        (
            "inventory.add",
            {
                "name": "番茄",
                "quantity_text": "2 个",
                "expires_at": datetime(2026, 8, 10, 9),
                "expiry_source": "unknown",
            },
        ),
        (
            "inventory.add",
            {
                "name": "番茄",
                "quantity_text": "2 个",
                "expires_at": datetime(2026, 8, 10, 9, tzinfo=UTC),
                "expiry_source": "bad-source",
            },
        ),
    ],
)
def test_add_rejects_naive_expiry_and_unknown_source(
    operation: str, arguments: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        validate_operation(operation, arguments)


@pytest.mark.parametrize(
    ("operation", "arguments"),
    [
        ("inventory.remove", {"instance_id": "inv_missing"}),
        ("inventory.replace", {
            "instance_id": "inv_missing",
            "quantity_text": "2 个",
            "expires_at": None,
            "expiry_source": "unknown",
        }),
        ("inventory.confirm", {"instance_id": "inv_missing"}),
    ],
)
def test_unknown_instance_id_fails_when_confirmed(
    tmp_path: Path, operation: str, arguments: dict[str, object]
) -> None:
    service, _ = build_service(tmp_path)
    action = service.propose("user-1", operation, arguments)

    with pytest.raises(KeyError, match="inventory instance not found"):
        service.confirm(action.action_id, "user-1")


def test_non_allowlisted_operation_is_rejected() -> None:
    with pytest.raises(ValueError):
        validate_operation("database.execute_sql", {"sql": "DROP TABLE inventory"})


def test_initialize_discards_legacy_pending_inventory_actions(tmp_path: Path) -> None:
    database = tmp_path / "inventory.db"
    pending = PendingActionRepository(database)
    pending.initialize()
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO pending_actions(
                action_id, user_id, operation, arguments_json, summary,
                status, created_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "act_legacy",
                "user-1",
                "inventory.add",
                '{"name": "番茄", "quantity": 2, "unit": "个"}',
                "旧操作",
                "pending",
                "2026-08-07T00:00:00+00:00",
                "2026-08-08T00:00:00+00:00",
            ),
        )

    pending.initialize()

    assert pending.get("act_legacy") is None
