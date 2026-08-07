from app.audit import regex_audit, review_tool_calls


def test_regex_audit_placeholder_blocks_fake_order_claim() -> None:
    assert not regex_audit("已经完成下单").passed


def test_regex_audit_placeholder_allows_normal_content() -> None:
    assert regex_audit("推荐一道不含花生的菜").passed


def inventory_instance(
    *,
    instance_id: str = "inv_ab12cdef0123456789abcdef01234567",
    name: str = "番茄",
    quantity_text: str = "大约100个",
) -> dict[str, object]:
    return {
        "instance_id": instance_id,
        "name": name,
        "quantity_text": quantity_text,
        "expires_at": None,
        "expiry_source": "unknown",
        "created_at": "2026-08-07T00:00:00+00:00",
        "updated_at": "2026-08-07T00:00:00+00:00",
        "last_confirmed_at": "2026-08-07T00:00:00+00:00",
    }


def test_inventory_audit_rejects_changed_quantity_text() -> None:
    audit = review_tool_calls(
        "当前库存：\n- 番茄 [inv_ab12]：100个",
        [
            {
                "tool": "list_inventory_instances",
                "arguments": {},
                "result": [inventory_instance()],
            }
        ],
    )

    assert audit.passed is False
    assert audit.reason == "Inventory value in answer differs from tool result"


def test_inventory_audit_accepts_exact_instance_quantity_text() -> None:
    audit = review_tool_calls(
        "当前库存：\n- 番茄 [inv_ab12]：大约100个",
        [
            {
                "tool": "list_inventory_instances",
                "arguments": {},
                "result": [inventory_instance()],
            }
        ],
    )

    assert audit.passed is True


def test_inventory_audit_accepts_full_exact_instance_id() -> None:
    instance = inventory_instance()
    audit = review_tool_calls(
        f"当前库存：\n- 番茄 [{instance['instance_id']}]：大约100个",
        [
            {
                "tool": "get_inventory_instance",
                "arguments": {"instance_id": instance["instance_id"]},
                "result": instance,
            }
        ],
    )

    assert audit.passed is True


def test_inventory_audit_rejects_changed_displayed_instance_id() -> None:
    audit = review_tool_calls(
        "当前库存：\n- 番茄 [inv_cd34]：大约100个",
        [
            {
                "tool": "list_inventory_instances",
                "arguments": {},
                "result": [inventory_instance()],
            }
        ],
    )

    assert audit.reason == "Inventory value in answer differs from tool result"


def test_inventory_audit_validates_find_name_and_get_instance_id() -> None:
    wrong_name = review_tool_calls(
        "库存中没有找到该食材。",
        [
            {
                "tool": "find_inventory_instances",
                "arguments": {"name": "土豆"},
                "result": [inventory_instance()],
            }
        ],
    )
    wrong_id = review_tool_calls(
        "库存中没有找到该实例。",
        [
            {
                "tool": "get_inventory_instance",
                "arguments": {
                    "instance_id": "inv_cd34cdef0123456789abcdef01234567"
                },
                "result": inventory_instance(),
            }
        ],
    )

    assert wrong_name.reason == "Inventory instances differ from requested name"
    assert wrong_id.reason == "Inventory instance result differs from requested ID"


def test_inventory_audit_rejects_missing_or_duplicate_result_ids() -> None:
    missing_id = inventory_instance()
    missing_id["instance_id"] = ""
    missing = review_tool_calls(
        "当前库存为空。",
        [
            {
                "tool": "list_inventory_instances",
                "arguments": {},
                "result": [missing_id],
            }
        ],
    )
    item = inventory_instance()
    duplicate = review_tool_calls(
        "当前库存为空。",
        [
            {
                "tool": "list_inventory_instances",
                "arguments": {},
                "result": [item, dict(item)],
            }
        ],
    )

    assert missing.reason == "Inventory instance result has an invalid ID"
    assert duplicate.reason == "Inventory instance result IDs are duplicated"


def test_inventory_audit_rejects_malformed_result_id() -> None:
    malformed = review_tool_calls(
        "当前库存为空。",
        [
            {
                "tool": "list_inventory_instances",
                "arguments": {},
                "result": [inventory_instance(instance_id="not-an-instance-id")],
            }
        ],
    )

    assert malformed.reason == "Inventory instance result has an invalid ID"


def test_inventory_audit_accepts_exact_lines_from_multiple_reads() -> None:
    tomato = inventory_instance()
    potato = inventory_instance(
        instance_id="inv_cd34cdef0123456789abcdef01234567",
        name="土豆",
        quantity_text="半袋",
    )
    audit = review_tool_calls(
        "库存：\n- 番茄 [inv_ab12]：大约100个\n- 土豆 [inv_cd34]：半袋",
        [
            {
                "tool": "find_inventory_instances",
                "arguments": {"name": "番茄"},
                "result": [tomato],
            },
            {
                "tool": "find_inventory_instances",
                "arguments": {"name": "土豆"},
                "result": [potato],
            },
        ],
    )

    assert audit.passed is True
