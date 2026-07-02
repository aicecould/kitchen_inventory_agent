from pathlib import Path

import pytest

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


def test_proposal_does_not_write_until_confirmed_and_confirmation_is_idempotent(
    tmp_path: Path,
) -> None:
    service, inventory = build_service(tmp_path)
    action = service.propose(
        "user-1",
        "inventory.add",
        {"name": "番茄", "quantity": 2, "unit": "个"},
    )

    assert inventory.get_item("番茄") is None
    assert action.status == "pending"

    executed = service.confirm(action.action_id, "user-1")
    repeated = service.confirm(action.action_id, "user-1")

    assert executed.status == "executed"
    assert repeated.status == "executed"
    assert inventory.get_item("番茄")["quantity"] == 2  # type: ignore[index]


def test_cancelled_action_cannot_execute(tmp_path: Path) -> None:
    service, inventory = build_service(tmp_path)
    action = service.propose(
        "user-1",
        "inventory.add",
        {"name": "鸡蛋", "quantity": 3, "unit": "个"},
    )
    service.cancel(action.action_id, "user-1")

    with pytest.raises(ValueError):
        service.confirm(action.action_id, "user-1")
    assert inventory.get_item("鸡蛋") is None


def test_other_user_cannot_confirm(tmp_path: Path) -> None:
    service, _ = build_service(tmp_path)
    action = service.propose(
        "user-1",
        "inventory.remove",
        {"name": "牛奶", "quantity": None},
    )
    with pytest.raises(PermissionError):
        service.confirm(action.action_id, "user-2")


def test_non_allowlisted_operation_is_rejected() -> None:
    with pytest.raises(ValueError):
        validate_operation("database.execute_sql", {"sql": "DROP TABLE inventory"})
