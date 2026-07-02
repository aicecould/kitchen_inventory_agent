from app.intent import match_intent, match_simple_inventory_operation


def test_inventory_intent_placeholder() -> None:
    assert match_intent("查询一下当前库存") == "inventory"


def test_order_is_explicitly_unsupported() -> None:
    assert match_intent("帮我加入购物车") == "order_unsupported"


def test_simple_inventory_write_is_structured() -> None:
    action = match_simple_inventory_operation("添加 2 个 番茄到库存")
    assert action is not None
    assert action.operation == "inventory.add"
    assert action.arguments == {"name": "番茄", "quantity": 2.0, "unit": "个"}
