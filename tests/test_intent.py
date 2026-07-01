from app.intent import match_intent


def test_inventory_intent_placeholder() -> None:
    assert match_intent("查询一下当前库存") == "inventory"


def test_order_is_explicitly_unsupported() -> None:
    assert match_intent("帮我加入购物车") == "order_unsupported"

