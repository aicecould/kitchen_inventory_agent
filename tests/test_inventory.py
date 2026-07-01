from pathlib import Path

import pytest

from app.tools.inventory import InventoryRepository


def test_inventory_crud(tmp_path: Path) -> None:
    repository = InventoryRepository(tmp_path / "inventory.db")
    repository.initialize()

    assert repository.add_item("番茄", 2, "个")["quantity"] == 2
    assert repository.add_item("番茄", 1, "个")["quantity"] == 3
    assert repository.update_item("番茄", 4, "个")["quantity"] == 4
    assert repository.remove_item("番茄", 1)["quantity"] == 3
    assert repository.remove_item("番茄")["removed"] is True
    assert repository.get_item("番茄") is None


def test_inventory_rejects_invalid_quantity(tmp_path: Path) -> None:
    repository = InventoryRepository(tmp_path / "inventory.db")
    repository.initialize()
    with pytest.raises(ValueError):
        repository.add_item("番茄", 0, "个")
