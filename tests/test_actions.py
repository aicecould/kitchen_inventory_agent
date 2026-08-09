from datetime import UTC, datetime, timedelta
from pathlib import Path
import sqlite3

import pytest
from pydantic import ValidationError

from app.actions import (
    InventoryActionService,
    PendingActionRepository,
    validate_operation,
)
from app.expiry import ExpiryInferer
from app.tools.inventory import InventoryRepository


def build_service(
    tmp_path: Path, expiry_inferer: ExpiryInferer | None = None
) -> tuple[InventoryActionService, InventoryRepository]:
    inventory = InventoryRepository(tmp_path / "inventory.db")
    inventory.initialize()
    pending = PendingActionRepository(tmp_path / "inventory.db")
    pending.initialize()
    service = (
        InventoryActionService(pending, inventory)
        if expiry_inferer is None
        else InventoryActionService(
            pending, inventory, expiry_inferer=expiry_inferer
        )
    )
    return service, inventory


class FutureInferer:
    def infer(self, name: str, now: datetime) -> datetime | None:
        return now + timedelta(days=5)


class FailingInferer:
    def infer(self, name: str, now: datetime) -> datetime | None:
        raise RuntimeError("provider unavailable")


class FixedInferer:
    def __init__(self, result: datetime | None) -> None:
        self.result = result

    def infer(self, name: str, now: datetime) -> datetime | None:
        return self.result


class ForbiddenInferer:
    def infer(self, name: str, now: datetime) -> datetime | None:
        raise AssertionError("explicit user expiry must bypass inference")


def test_propose_add_stores_valid_inferred_expiry(tmp_path: Path) -> None:
    service, _ = build_service(tmp_path, FutureInferer())

    action = service.propose_add("u", "牛奶", "一盒")

    expires_at = datetime.fromisoformat(str(action.arguments["expires_at"]))
    assert expires_at > action.created_at
    assert action.arguments["expiry_source"] == "llm"


def test_propose_add_falls_back_to_unknown_when_inference_raises(
    tmp_path: Path,
) -> None:
    service, _ = build_service(tmp_path, FailingInferer())

    action = service.propose_add("u", "牛奶", "一盒")

    assert action.arguments["expires_at"] is None
    assert action.arguments["expiry_source"] == "unknown"


@pytest.mark.parametrize(
    "prediction",
    [datetime(2000, 1, 1, tzinfo=UTC), datetime(2099, 1, 1)],
)
def test_propose_add_falls_back_to_unknown_for_invalid_inferred_expiry(
    tmp_path: Path, prediction: datetime
) -> None:
    service, _ = build_service(tmp_path, FixedInferer(prediction))

    action = service.propose_add("u", "牛奶", "一盒")

    assert action.arguments["expires_at"] is None
    assert action.arguments["expiry_source"] == "unknown"


def test_propose_add_uses_explicit_user_expiry_without_inference(tmp_path: Path) -> None:
    service, _ = build_service(tmp_path, ForbiddenInferer())
    user_expiry = datetime(2026, 8, 12, 9, tzinfo=UTC)

    action = service.propose_add(
        "u", "牛奶", "一盒", user_expires_at=user_expiry
    )

    assert action.arguments["expires_at"] == "2026-08-12T09:00:00Z"
    assert action.arguments["expiry_source"] == "user"


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


def test_add_preview_persists_all_proposed_values(tmp_path: Path) -> None:
    """The confirmation payload must expose every value the add will create."""
    service, _ = build_service(tmp_path)

    action = service.propose(
        "user-1",
        "inventory.add",
        {
            "name": "牛奶",
            "quantity_text": "大约一盒",
            "expires_at": "2026-08-12T09:00:00+00:00",
            "expiry_source": "user",
        },
    )

    assert action.preview == {
        "proposed": {
            "name": "牛奶",
            "quantity_text": "大约一盒",
            "expires_at": "2026-08-12T09:00:00Z",
            "expiry_source": "user",
        }
    }
    assert service.repository.get(action.action_id) == action


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

    assert str(item["instance_id"]) in action.summary
    assert inventory.get_instance(str(item["instance_id"]))["quantity_text"] == "1 盒"  # type: ignore[index]

    executed = service.confirm(action.action_id, "user-1")

    assert executed.result is not None
    assert executed.result["quantity_text"] == "2 盒"
    assert inventory.get_instance(str(item["instance_id"]))["expiry_source"] == "llm"  # type: ignore[index]


def test_replace_preview_has_full_id_and_current_and_proposed_values(
    tmp_path: Path,
) -> None:
    """A replace approval must compare the actual row with the complete replacement."""
    service, inventory = build_service(tmp_path)
    item = inventory.add_instance("牛奶", "1盒", None, "unknown")
    instance_id = str(item["instance_id"])

    action = service.propose(
        "user-1",
        "inventory.replace",
        {
            "instance_id": instance_id,
            "quantity_text": "半盒",
            "expires_at": "2026-08-12T09:00:00+00:00",
            "expiry_source": "user",
        },
    )

    assert action.summary == f"更新库存项 {instance_id}"
    assert action.preview == {
        "current": item,
        "proposed": {
            "instance_id": instance_id,
            "name": "牛奶",
            "quantity_text": "半盒",
            "expires_at": "2026-08-12T09:00:00Z",
            "expiry_source": "user",
        },
    }


def test_remove_deletes_the_identified_instance(tmp_path: Path) -> None:
    service, inventory = build_service(tmp_path)
    item = inventory.add_instance("鸡蛋", "6 个", None, "unknown")
    action = service.propose(
        "user-1", "inventory.remove", {"instance_id": item["instance_id"]}
    )

    assert str(item["instance_id"]) in action.summary
    assert inventory.get_instance(str(item["instance_id"])) is not None

    service.confirm(action.action_id, "user-1")

    assert inventory.get_instance(str(item["instance_id"])) is None


@pytest.mark.parametrize(
    ("operation", "summary_prefix"),
    [
        ("inventory.remove", "从库存删除库存项"),
        ("inventory.confirm", "确认库存项"),
    ],
)
def test_remove_and_confirm_previews_show_the_full_current_instance(
    tmp_path: Path, operation: str, summary_prefix: str
) -> None:
    """Single-instance approvals must expose the full authoritative target."""
    service, inventory = build_service(tmp_path)
    item = inventory.add_instance("鸡蛋", "6个", None, "unknown")
    instance_id = str(item["instance_id"])

    action = service.propose("user-1", operation, {"instance_id": instance_id})

    assert action.summary == f"{summary_prefix} {instance_id}"
    assert action.preview == {"current": item}


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


def test_remove_all_preview_has_exact_items_and_expected_count(tmp_path: Path) -> None:
    """Bulk approval must disclose exactly which duplicate-name rows will be removed."""
    service, inventory = build_service(tmp_path)
    items = [
        inventory.add_instance("番茄", "2个", None, "unknown"),
        inventory.add_instance("番茄", "半盒", None, "unknown"),
    ]

    action = service.propose("user-1", "inventory.remove_all", {"name": "番茄"})

    assert action.preview == {
        "name": "番茄",
        "items": inventory.find_by_name("番茄"),
        "expected_count": 2,
    }
    assert {str(item["instance_id"]) for item in action.preview["items"]} == {
        str(item["instance_id"]) for item in items
    }


def test_remove_all_without_matches_fails_before_creating_proposal(
    tmp_path: Path,
) -> None:
    """An empty bulk target must never produce a misleading confirmable action."""
    service, _ = build_service(tmp_path)

    with pytest.raises(KeyError, match="inventory instances not found: 番茄"):
        service.propose("user-1", "inventory.remove_all", {"name": "番茄"})

    assert service.repository.list_pending("user-1") == []


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
    assert str(item["instance_id"]) in action.summary


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
        ("inventory.remove", {"instance_id": "inv_abcd"}),
        (
            "inventory.replace",
            {
                "instance_id": "inv_abcd",
                "quantity_text": "半盒",
                "expires_at": None,
                "expiry_source": "unknown",
            },
        ),
        ("inventory.confirm", {"instance_id": "inv_abcd"}),
    ],
)
def test_mutation_arguments_reject_short_instance_prefixes(
    operation: str, arguments: dict[str, object]
) -> None:
    """A displayed prefix must never validate as mutation authority."""
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
        (
            "inventory.remove",
            {"instance_id": "inv_99999999999999999999999999999999"},
        ),
        ("inventory.replace", {
            "instance_id": "inv_99999999999999999999999999999999",
            "quantity_text": "2 个",
            "expires_at": None,
            "expiry_source": "unknown",
        }),
        (
            "inventory.confirm",
            {"instance_id": "inv_99999999999999999999999999999999"},
        ),
    ],
)
def test_unknown_instance_id_fails_before_proposal(
    tmp_path: Path, operation: str, arguments: dict[str, object]
) -> None:
    service, _ = build_service(tmp_path)

    with pytest.raises(KeyError, match="inventory instance not found"):
        service.propose("user-1", operation, arguments)

    assert service.repository.list_pending("user-1") == []


def test_non_allowlisted_operation_is_rejected() -> None:
    with pytest.raises(ValueError):
        validate_operation("database.execute_sql", {"sql": "DROP TABLE inventory"})


def test_initialize_fails_legacy_pending_inventory_actions(tmp_path: Path) -> None:
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

    legacy = pending.get("act_legacy")
    assert legacy is not None
    assert legacy.status == "failed"
    assert legacy.error == (
        "Structured confirmation preview unavailable; recreate this proposal."
    )


def test_initialize_fails_pre_preview_pending_actions_and_preserves_executed_history(
    tmp_path: Path,
) -> None:
    """Migration must block undisclosed writes without rewriting executed history."""
    database = tmp_path / "inventory.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE pending_actions (
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
            """
            INSERT INTO pending_actions(
                action_id, user_id, operation, arguments_json, summary,
                status, created_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "act_existing",
                "user-1",
                "inventory.remove_all",
                '{"name": "番茄"}',
                "已有提案",
                "pending",
                "2026-08-07T00:00:00+00:00",
                "2099-08-08T00:00:00+00:00",
            ),
        )
        connection.execute(
            """
            INSERT INTO pending_actions(
                action_id, user_id, operation, arguments_json, summary,
                status, created_at, expires_at, result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "act_executed",
                "user-1",
                "inventory.remove_all",
                '{"name": "番茄"}',
                "历史操作",
                "executed",
                "2026-08-06T00:00:00+00:00",
                "2026-08-06T00:15:00+00:00",
                '{"removed_count": 1}',
            ),
        )

    repository = PendingActionRepository(database)
    repository.initialize()

    existing = repository.get("act_existing")
    assert existing is not None
    assert existing.preview == {}
    assert existing.status == "failed"
    assert existing.error == (
        "Structured confirmation preview unavailable; recreate this proposal."
    )
    assert repository.list_pending("user-1") == []
    executed = repository.get("act_executed")
    assert executed is not None
    assert executed.status == "executed"
    assert executed.result == {"removed_count": 1}
    assert executed.error is None
    with sqlite3.connect(database) as connection:
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(pending_actions)")
        }
    assert "preview_json" in columns


def test_confirm_rejects_a_pending_inventory_action_with_tampered_empty_preview(
    tmp_path: Path,
) -> None:
    """Server confirmation must revalidate the persisted preview before mutation."""
    service, inventory = build_service(tmp_path)
    item = inventory.add_instance("番茄", "2个", None, "unknown")
    action = service.propose(
        "user-1",
        "inventory.remove",
        {"instance_id": item["instance_id"]},
    )
    with sqlite3.connect(service.repository.db_path) as connection:
        connection.execute(
            "UPDATE pending_actions SET preview_json = '{}' WHERE action_id = ?",
            (action.action_id,),
        )

    with pytest.raises(ValueError, match="Structured confirmation preview unavailable"):
        service.confirm(action.action_id, "user-1")

    rejected = service.repository.get(action.action_id)
    assert rejected is not None
    assert rejected.status == "failed"
    assert rejected.error == (
        "Structured confirmation preview unavailable; recreate this proposal."
    )
    assert inventory.get_instance(str(item["instance_id"])) == item


def test_initialize_fails_a_pending_action_with_nonempty_invalid_preview(
    tmp_path: Path,
) -> None:
    """A nonempty preview is unsafe when its count and disclosed targets are invalid."""
    service, _ = build_service(tmp_path)
    with sqlite3.connect(service.repository.db_path) as connection:
        connection.execute(
            """
            INSERT INTO pending_actions(
                action_id, user_id, operation, arguments_json, summary,
                status, created_at, expires_at, preview_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "act_invalid_preview",
                "user-1",
                "inventory.remove_all",
                '{"name": "番茄"}',
                "不完整预览",
                "pending",
                "2026-08-07T00:00:00+00:00",
                "2099-08-08T00:00:00+00:00",
                '{"name": "番茄", "items": [], "expected_count": 0}',
            ),
        )

    service.repository.initialize()

    rejected = service.repository.get("act_invalid_preview")
    assert rejected is not None
    assert rejected.status == "failed"
    assert rejected.error == (
        "Structured confirmation preview unavailable; recreate this proposal."
    )
