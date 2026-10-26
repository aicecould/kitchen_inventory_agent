import pytest

import app.audit as audit_module
from app.audit import (
    AuditCode,
    regex_audit,
    review_candidate,
    review_tool_calls,
    validate_profile_state_transition,
)


def test_regex_audit_placeholder_blocks_fake_order_claim() -> None:
    result = regex_audit("已经完成下单")

    assert result.passed is False
    assert result.code == AuditCode.REPAIRABLE_OUTPUT_VIOLATION


def test_regex_audit_placeholder_allows_normal_content() -> None:
    result = regex_audit("推荐一道不含花生的菜")

    assert result.passed is True
    assert result.code == AuditCode.PASSED


@pytest.mark.parametrize("internal", [
    "inv_" + "1" * 32, "act_" + "2" * 32,
    "expiry_source=llm", "stocked_at", "local_recipe_dataset",
    "<think>internal</think>", "<tool_call>{}</tool_call>",
    "2026-10-26T08:45:29+00:00",
])
def test_internal_metadata_and_protocol_never_pass_public_answer_audit(internal):
    result = regex_audit(f"推荐番茄汤。{internal}")
    assert not result.passed
    assert result.code == AuditCode.REPAIRABLE_OUTPUT_VIOLATION


def test_one_recipe_request_rejects_even_an_additional_side_suggestion():
    history = [{"tool": "search_recipes", "arguments": {"ingredients": ["tomato"], "ingredient_mapping": []}, "result": [
        {"id": "r1", "title": "Tomato Soup", "ingredients": ["tomato"], "instructions": "Simmer."},
        {"id": "r2", "title": "Apple Crisp", "ingredients": ["apple"], "instructions": "Bake."},
    ]}]
    result = review_candidate("推荐 Tomato Soup。另可做 Apple Crisp。", history,
                              intent="recipe", request_text="根据库存推荐一道菜")
    assert result.code == AuditCode.REPAIRABLE_OUTPUT_VIOLATION
    assert review_candidate("推荐 Tomato Soup（番茄汤）。", history,
                            intent="recipe", request_text="根据库存推荐一道菜").passed


def test_repeated_generation_cannot_pass_as_a_recipe_answer():
    result = regex_audit("推荐番茄汤。缺少材料：" + "黄油、鸡蛋、面粉、牛奶、柠檬、" * 8)
    assert not result.passed
    assert result.reason == "Model output contains repeated text"


def test_recipe_recommendation_requires_search_tool_history() -> None:
    result = review_candidate(
        "推荐一道番茄汤菜谱，步骤是把番茄煮熟。",
        [],
        intent="recipe",
    )

    assert result.passed is False
    assert result.code == AuditCode.RECIPE_TOOL_MISSING


def test_english_recipe_recommendation_requires_search_tool_history() -> None:
    result = review_candidate(
        "I recommend tomato soup. Recipe steps: simmer the tomatoes.",
        [],
        intent="recipe",
        target_language="en",
    )

    assert result.passed is False
    assert result.code == AuditCode.RECIPE_TOOL_MISSING


def test_recipe_recommendation_must_match_tool_result() -> None:
    result = review_candidate(
        "推荐一道土豆浓汤菜谱，步骤是煮熟土豆。",
        [
            {
                "tool": "search_recipes",
                "arguments": {
                    "ingredients": ["tomato"],
                    "ingredient_mapping": [],
                    "exclude_ingredients": [],
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
            }
        ],
        intent="recipe",
    )

    assert result.passed is False
    assert result.code == AuditCode.RECIPE_RESULT_MISMATCH


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
    assert audit.code == AuditCode.REPAIRABLE_OUTPUT_VIOLATION


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


def test_canonical_inventory_renderer_preserves_quantity_without_internal_fields() -> None:
    """Public inventory keeps quantities, while tool records retain identifiers."""
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
    assert "当前库存：" in content
    assert str(item["instance_id"]) not in content
    assert "大约100个" in content
    assert "2026-08-07T00:00:00+00:00" not in content
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
    assert "尚未执行" in content
    assert "status=pending" not in content
    for fragment in fragments:
        if fragment.startswith(("inv_", "act_")) or fragment == "最后确认":
            assert fragment not in content
        elif fragment not in {"预计数量 1", "unknown"}:
            assert fragment in content
    assert "inv_" not in content and "act_" not in content
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
                "ingredient_mapping": [],
                "exclude_ingredients": [],
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

    assert audit_module.canonical_tool_response(history) is None
    content = "推荐 **Tomato Soup**（番茄汤）：将番茄煮软。来源：[原始菜谱](https://recipes.example/tomato-soup)"
    assert review_tool_calls(content, history).passed is True
    assert history[0]["result"][0]["instance_id"] == item["instance_id"]


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
                "ingredient_mapping": [],
                "exclude_ingredients": [],
            },
            "result": [],
        },
    ]

    assert audit_module.canonical_tool_response(history) is None
    assert review_tool_calls("未找到符合条件的菜谱。", history).passed is True


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
                "ingredient_mapping": [],
                "exclude_ingredients": [],
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
    assert "Current inventory:" in content
    assert "Pending actions (not executed):" in content
    assert "(not executed)" in content
    assert "status=pending" not in content
    assert "Recipe suggestion:" in content
    assert "Ingredients: tomato; salt" in content
    assert "Instructions: Simmer, then serve." in content
    assert "Source: example" in content
    assert "库存实例快照" not in content
    assert review_tool_calls(content, history, target_language="en").passed is True


def profile_proposal_history() -> list[dict[str, object]]:
    arguments: dict[str, object] = {
        "operation": "upsert",
        "category": "preference",
        "key": "ingredient.cilantro",
        "value": "不喜欢香菜",
        "reason": "用户明确表达长期偏好",
    }
    preview: dict[str, object] = {
        "before_revision": 0,
        "after_revision": 1,
        "proposal_id": "prof_" + "1" * 32,
        "source": "explicit_user_statement",
        "evidence_ids": ["hist_" + "2" * 32],
        "current": None,
        "proposed": {
            "operation": "upsert",
            "category": "preference",
            "key": "ingredient.cilantro",
            "value": "不喜欢香菜",
        },
    }
    return [
        {
            "tool": "propose_profile_change",
            "arguments": arguments,
            "result": {
                "proposal_id": "prof_" + "1" * 32,
                "user_id": "showcase-user",
                **arguments,
                "source": "explicit_user_statement",
                "evidence_ids": ["hist_" + "2" * 32],
                "status": "pending",
                "base_revision": 0,
                "created_at": "2026-08-10T05:00:00Z",
                "expires_at": "2026-09-09T05:00:00Z",
                "preview": preview,
            },
        }
    ]


def test_profile_proposal_uses_existing_tool_audit_and_canonical_output() -> None:
    history = profile_proposal_history()

    content = audit_module.canonical_tool_response(history)

    assert content is not None
    assert "画像变更" in content
    assert "不喜欢香菜" in content
    assert "尚未写入" in content
    assert review_tool_calls(content, history).passed is True


def test_profile_proposal_rejects_invalid_id() -> None:
    history = profile_proposal_history()
    result = history[0]["result"]
    assert isinstance(result, dict)
    result["proposal_id"] = "not-a-profile-id"

    audit = review_tool_calls("任意回答", history)

    assert audit.passed is False
    assert audit.reason == "Profile proposal has an invalid ID"


def test_profile_proposal_rejects_preview_mismatch() -> None:
    history = profile_proposal_history()
    result = history[0]["result"]
    assert isinstance(result, dict)
    preview = result["preview"]
    assert isinstance(preview, dict)
    preview["after_revision"] = 8

    audit = review_tool_calls("任意回答", history)

    assert audit.passed is False
    assert audit.reason == "Profile preview revision is invalid"


def test_profile_state_transition_requires_matching_passed_audit_event() -> None:
    before = {
        "schema_version": 2,
        "revision": 4,
        "facts": [],
        "proposals": [],
        "audit_events": [],
    }
    after = {
        **before,
        "revision": 5,
        "audit_events": [
            {
                "operation": "reject_profile_change",
                "status": "passed",
                "reason": "validated",
                "created_at": "2026-08-10T05:00:00Z",
            }
        ],
    }

    assert (
        validate_profile_state_transition(before, after, "confirm_profile_change")
        == "Profile audit event does not match the mutation"
    )


def test_profile_state_transition_blocks_silent_inferred_requirement() -> None:
    before = {
        "schema_version": 2,
        "revision": 0,
        "facts": [],
        "proposals": [],
        "audit_events": [],
    }
    after = {
        **before,
        "revision": 1,
        "facts": [
            {
                "fact_id": "fact_" + "1" * 32,
                "category": "requirement",
                "key": "allergen.peanut",
                "value": "对花生过敏",
                "origin": "inferred",
                "status": "active",
                "evidence_ids": ["hist_" + "2" * 32, "hist_" + "3" * 32],
            }
        ],
        "audit_events": [
            {
                "operation": "apply_inferred_profile_change",
                "status": "passed",
                "reason": "validated",
                "created_at": "2026-08-10T05:00:00Z",
            }
        ],
    }

    assert (
        validate_profile_state_transition(
            before, after, "apply_inferred_profile_change"
        )
        == "Mandatory profile requirements cannot be inferred silently"
    )
