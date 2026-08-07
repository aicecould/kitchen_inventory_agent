import pytest

from app.intent import (
    RouteDecision,
    match_intent,
    match_simple_inventory_operation,
    match_simple_inventory_query,
    route_deterministically,
)


def test_inventory_intent_placeholder() -> None:
    assert match_intent("查询一下当前库存") == "inventory"


def test_order_is_explicitly_unsupported() -> None:
    assert match_intent("帮我加入购物车") == "order_unsupported"


def test_simple_inventory_write_is_structured() -> None:
    action = match_simple_inventory_operation("添加 2 个 番茄到库存")
    assert action is not None
    assert action.operation == "inventory.add"
    assert action.arguments == {"name": "番茄", "quantity_text": "2 个"}


def test_simple_inventory_query_matches_allowlisted_phrasing() -> None:
    assert match_simple_inventory_query("查询一下当前库存") is True
    assert match_simple_inventory_query("查看库存列表。") is True
    assert match_simple_inventory_query("请帮我查询库存") is False


@pytest.mark.parametrize("text", ["你好", "您好", "在吗", "谢谢", "再见"])
def test_exact_greetings_are_irrelevant(text: str) -> None:
    assert route_deterministically(text) == RouteDecision("irrelevant", None, {})


def test_greeting_matching_does_not_capture_a_longer_request() -> None:
    assert route_deterministically("你好，查询库存").kind == "undecided"


@pytest.mark.parametrize(
    ("text", "operation", "arguments"),
    [
        ("查询一下当前库存", "inventory.list", {}),
        ("查看土豆还有多少", "inventory.find", {"name": "土豆"}),
        ("删除所有土豆", "inventory.remove_all", {"name": "土豆"}),
        (
            "添加2个土豆到库存",
            "inventory.add",
            {"name": "土豆", "quantity_text": "2个"},
        ),
    ],
)
def test_supported_inventory_requests_have_structured_routes(
    text: str, operation: str, arguments: dict[str, object]
) -> None:
    assert route_deterministically(text) == RouteDecision(
        "supported", operation, arguments
    )


@pytest.mark.parametrize(
    "text", ["下一步呢", "然后呢", "继续", "用了一半", "都用完了", "删掉它"]
)
def test_follow_up_phrases_are_contextual(text: str) -> None:
    assert route_deterministically(text) == RouteDecision("contextual", None, {})


@pytest.mark.parametrize(
    "text",
    ["Python怎么安装", "明天天气如何", "查询我的订单", "怎么支付"],
)
def test_other_domains_remain_undecided_for_classifier(text: str) -> None:
    assert route_deterministically(text) == RouteDecision("undecided", None, {})
