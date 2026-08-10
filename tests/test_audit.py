import pytest

import app.audit as audit_module
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
        "stocked_at": "2026-08-07T00:00:00+00:00",
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
    assert audit.reason == "Inventory response does not match canonical tool output"


def test_inventory_audit_accepts_exact_instance_quantity_text() -> None:
    history = [
        {
            "tool": "list_inventory_instances",
            "arguments": {},
            "result": [inventory_instance()],
        }
    ]
    audit = review_tool_calls(
        audit_module.canonical_tool_response(history) or "",
        history,
    )

    assert audit.passed is True


def test_inventory_audit_accepts_full_exact_instance_id() -> None:
    instance = inventory_instance()
    history = [
        {
            "tool": "get_inventory_instance",
            "arguments": {"instance_id": instance["instance_id"]},
            "result": instance,
        }
    ]
    audit = review_tool_calls(
        audit_module.canonical_tool_response(history) or "",
        history,
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

    assert audit.reason == "Inventory response does not match canonical tool output"


def test_inventory_audit_rejects_unbulleted_changed_instance_claims() -> None:
    history = [
        {
            "tool": "list_inventory_instances",
            "arguments": {},
            "result": [inventory_instance()],
        }
    ]

    changed_quantity = review_tool_calls(
        "番茄 [inv_ab12]：100个",
        history,
    )
    changed_id = review_tool_calls(
        "番茄 [inv_cd34]：大约100个",
        history,
    )
    exact = review_tool_calls(
        "番茄 [inv_ab12]：大约100个",
        history,
    )

    assert changed_quantity.reason == "Inventory response does not match canonical tool output"
    assert changed_id.reason == "Inventory response does not match canonical tool output"
    assert exact.reason == "Inventory response does not match canonical tool output"


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
    history = [
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
    ]
    audit = review_tool_calls(
        audit_module.canonical_tool_response(history) or "", history
    )

    assert audit.passed is True


def test_canonical_inventory_renderer_preserves_complete_structured_snapshot() -> None:
    """Canonical output must expose every authoritative identity and quantity value."""
    assert hasattr(audit_module, "canonical_tool_response")
    item = inventory_instance()
    history = [
        {
            "tool": "list_inventory_instances",
            "arguments": {},
            "result": [item],
        }
    ]

    content = audit_module.canonical_tool_response(history)

    assert content is not None
    assert "库存实例快照（只读）" in content
    assert str(item["instance_id"]) in content
    assert "大约100个" in content
    assert "2026-08-07T00:00:00+00:00" in content
    assert review_tool_calls(content, history).passed is True


@pytest.mark.parametrize(
    "content",
    [
        "番茄还有大约100个。",
        "| 食材 | 数量 |\n| --- | --- |\n| 番茄 | 大约100个 |",
        "> 番茄 [inv_ab12cdef0123456789abcdef01234567]：大约100个",
        "番茄：大约100个",
        "Tomatoes: about 100.",
    ],
)
def test_inventory_audit_rejects_noncanonical_prose_table_quote_or_omitted_id(
    content: str,
) -> None:
    """Format changes must not bypass positive structure validation."""
    result = review_tool_calls(
        content,
        [
            {
                "tool": "list_inventory_instances",
                "arguments": {},
                "result": [inventory_instance()],
            }
        ],
    )

    assert result.passed is False
    assert result.reason == "Inventory response does not match canonical tool output"


def proposal_history(operation: str) -> tuple[list[dict[str, object]], list[str]]:
    item = inventory_instance()
    instance_id = str(item["instance_id"])
    base = {
        "action_id": "act_11111111111111111111111111111111",
        "user_id": "test",
        "operation": operation,
        "summary": "pending",
        "status": "pending",
        "created_at": "2026-08-07T00:00:00Z",
        "expires_at": "2026-08-07T00:15:00Z",
        "result": None,
        "error": None,
    }
    if operation == "inventory.add":
        tool = "propose_add_inventory_instance"
        arguments = {
            "name": "牛奶",
            "quantity_text": "一盒",
            "expires_at": None,
            "expiry_source": "unknown",
        }
        preview = {"proposed": arguments}
        fragments = ["牛奶", "一盒", "unknown"]
    elif operation == "inventory.replace":
        tool = "propose_replace_inventory_instance"
        arguments = {
            "instance_id": instance_id,
            "quantity_text": "半盒",
            "expires_at": None,
            "expiry_source": "unknown",
        }
        preview = {
            "current": item,
            "proposed": {**arguments, "name": "番茄"},
        }
        fragments = [instance_id, "大约100个", "半盒"]
    elif operation == "inventory.remove":
        tool = "propose_remove_inventory_instance"
        arguments = {"instance_id": instance_id}
        preview = {"current": item}
        fragments = [instance_id, "大约100个"]
    elif operation == "inventory.remove_all":
        tool = "propose_remove_all_inventory_instances"
        arguments = {"name": "番茄"}
        preview = {"name": "番茄", "items": [item], "expected_count": 1}
        fragments = ["番茄", "预计数量 1", instance_id]
    else:
        tool = "propose_confirm_inventory_instance"
        arguments = {"instance_id": instance_id}
        preview = {"current": item}
        fragments = [instance_id, "大约100个", "最后确认"]
    result = {**base, "arguments": arguments, "preview": preview}
    return [
        {"tool": tool, "arguments": arguments, "result": result}
    ], fragments


@pytest.mark.parametrize(
    "operation",
    [
        "inventory.add",
        "inventory.replace",
        "inventory.remove",
        "inventory.remove_all",
        "inventory.confirm",
    ],
)
def test_canonical_renderer_supports_every_pending_inventory_operation(
    operation: str,
) -> None:
    """Every confirmable operation must have a deterministic pending preview."""
    assert hasattr(audit_module, "canonical_tool_response")
    history, fragments = proposal_history(operation)

    content = audit_module.canonical_tool_response(history)

    assert content is not None
    assert "待确认操作（尚未执行）" in content
    assert "status=pending（尚未执行）" in content
    for fragment in fragments:
        assert fragment in content
    assert review_tool_calls(content, history).passed is True


@pytest.mark.parametrize(
    "content",
    [
        "I removed the inventory item.",
        "The item has now been removed.",
        "该库存项现已移除。",
        "库存项已经不在了。",
    ],
)
def test_pending_inventory_audit_rejects_english_and_paraphrased_execution_claims(
    content: str,
) -> None:
    """Pending state is accepted only through the positive canonical disclosure."""
    history, _ = proposal_history("inventory.remove")

    result = review_tool_calls(content, history)

    assert result.passed is False
    assert result.reason == "Inventory response does not match canonical tool output"


def test_canonical_mixed_response_appends_recipe_titles_and_source_links() -> None:
    """Replacing model prose must retain every useful structured recipe field."""
    assert hasattr(audit_module, "canonical_tool_response")
    item = inventory_instance()
    history = [
        {
            "tool": "list_inventory_instances",
            "arguments": {},
            "result": [item],
        },
        {
            "tool": "search_recipes",
            "arguments": {
                "ingredients": ["tomato"],
                "intolerances": [],
                "exclude_ingredients": [],
                "custom_allergen_mapping": [],
            },
            "result": [
                {
                    "id": "recipe-1",
                    "title": "Tomato Soup",
                    "instructions": "Simmer.",
                    "ingredients": ["tomato"],
                    "source": "example",
                    "source_url": "https://recipes.example/tomato-soup",
                    "image_url": None,
                }
            ],
        },
    ]

    content = audit_module.canonical_tool_response(history)

    assert content is not None
    assert "Tomato Soup" in content
    assert "食材：tomato" in content
    assert "步骤：Simmer." in content
    assert "来源：example" in content
    assert "https://recipes.example/tomato-soup" in content
    assert review_tool_calls(content, history).passed is True


def test_canonical_mixed_response_discloses_an_empty_recipe_result() -> None:
    """An empty recipe tool result must not disappear behind an inventory snapshot."""
    history = [
        {
            "tool": "list_inventory_instances",
            "arguments": {},
            "result": [inventory_instance()],
        },
        {
            "tool": "search_recipes",
            "arguments": {
                "ingredients": ["tomato"],
                "intolerances": [],
                "exclude_ingredients": [],
                "custom_allergen_mapping": [],
            },
            "result": [],
        },
    ]

    content = audit_module.canonical_tool_response(history)

    assert content is not None
    assert "菜谱工具结果：" in content
    assert "未找到符合条件的菜谱。" in content
    assert review_tool_calls(content, history).passed is True


def test_canonical_mixed_response_honors_english_target_language() -> None:
    """Non-Chinese inventory-bearing output must use the safe English rendering."""
    item = inventory_instance()
    proposal, _ = proposal_history("inventory.remove")
    history = [
        {"tool": "list_inventory_instances", "arguments": {}, "result": [item]},
        *proposal,
        {
            "tool": "search_recipes",
            "arguments": {
                "ingredients": ["tomato"],
                "intolerances": [],
                "exclude_ingredients": [],
                "custom_allergen_mapping": [],
            },
            "result": [
                {
                    "id": "recipe-1",
                    "title": "Tomato Soup",
                    "instructions": "Simmer, then serve.",
                    "ingredients": ["tomato", "salt"],
                    "source": "example",
                    "source_url": "https://recipes.example/tomato-soup",
                    "image_url": None,
                }
            ],
        },
    ]

    content = audit_module.canonical_tool_response(history, target_language="en")

    assert content is not None
    assert "Inventory snapshots (read-only):" in content
    assert "Pending actions (not executed):" in content
    assert "status=pending (not executed)" in content
    assert "Recipe tool results:" in content
    assert "Ingredients: tomato; salt" in content
    assert "Instructions: Simmer, then serve." in content
    assert "Source: example" in content
    assert "库存实例快照" not in content
    assert review_tool_calls(content, history, target_language="en").passed is True
