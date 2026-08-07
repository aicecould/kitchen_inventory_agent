"""Simple intent matching placeholder.

TODO: Add vector intent recognition after this stage when required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


RouteKind = Literal["irrelevant", "supported", "contextual", "undecided"]


@dataclass(frozen=True, slots=True)
class RouteDecision:
    kind: RouteKind
    operation: str | None
    arguments: dict[str, object]


@dataclass(frozen=True, slots=True)
class SimpleInventoryOperation:
    operation: str
    arguments: dict[str, object]


_UNITS = "个|克|千克|公斤|瓶|袋|盒|根|颗|份"
_ADD_PATTERN = re.compile(
    rf"^(?:添加|增加|加入)(?:库存)?\s*"
    rf"(?P<quantity_text>\d+(?:\.\d+)?\s*(?:{_UNITS}))\s*"
    rf"(?P<name>.+?)(?:到库存)?$"
)
_UPDATE_PATTERN = re.compile(
    rf"^(?:将)?\s*(?P<name>.+?)\s*(?:更新|修改|设置)为\s*"
    rf"(?P<quantity>\d+(?:\.\d+)?)\s*(?P<unit>{_UNITS})$"
)
_REMOVE_PATTERN = re.compile(r"^(?:删除|移除)\s*(?P<name>.+?)(?:库存)?$")
_REDUCE_PATTERN = re.compile(
    rf"^(?:减少|移除)\s*(?P<name>.+?)\s*(?P<quantity>\d+(?:\.\d+)?)\s*"
    rf"(?P<unit>{_UNITS})$"
)
_LIST_INVENTORY_PATTERN = re.compile(
    r"^(?:查询|查看|看看|列出)(?:一下)?(?:当前|现有)?库存(?:情况|列表)?[。！？!?]?$"
)
_GREETING_PATTERN = re.compile(r"^(?:你好|您好|在吗|谢谢|再见)[。！？!?]?$")
_NAMED_INVENTORY_PATTERN = re.compile(
    r"^(?:查询|查看|看看)(?:一下)?\s*(?P<name>.+?)\s*"
    r"(?:还有多少|有多少|库存)[。！？!?]?$"
)
_REMOVE_ALL_PATTERN = re.compile(
    r"^(?:删除|移除)\s*所有\s*(?P<name>.+?)(?:库存)?[。！？!?]?$"
)
_CONTEXTUAL_PATTERN = re.compile(
    r"^(?:下一步呢?|然后呢?|继续|用了一半|都用完了|删掉它)[。！？!?]?$"
)


def route_deterministically(text: str) -> RouteDecision:
    normalized = text.strip()

    if _GREETING_PATTERN.fullmatch(normalized):
        return RouteDecision("irrelevant", None, {})
    if match_simple_inventory_query(normalized):
        return RouteDecision("supported", "inventory.list", {})

    named_match = _NAMED_INVENTORY_PATTERN.fullmatch(normalized)
    if named_match:
        return RouteDecision(
            "supported", "inventory.find", {"name": named_match.group("name").strip()}
        )

    remove_all_match = _REMOVE_ALL_PATTERN.fullmatch(normalized)
    if remove_all_match:
        return RouteDecision(
            "supported",
            "inventory.remove_all",
            {"name": remove_all_match.group("name").strip()},
        )

    simple_operation = match_simple_inventory_operation(normalized)
    if simple_operation is not None and simple_operation.operation == "inventory.add":
        return RouteDecision(
            "supported", simple_operation.operation, simple_operation.arguments
        )

    if _CONTEXTUAL_PATTERN.fullmatch(normalized):
        return RouteDecision("contextual", None, {})
    return RouteDecision("undecided", None, {})


def match_simple_inventory_query(text: str) -> bool:
    normalized = " ".join(text.strip().split())
    return _LIST_INVENTORY_PATTERN.fullmatch(normalized) is not None


def match_simple_inventory_operation(text: str) -> SimpleInventoryOperation | None:
    normalized = text.strip()
    for pattern, operation in (
        (_ADD_PATTERN, "inventory.add"),
        (_UPDATE_PATTERN, "inventory.update"),
        (_REDUCE_PATTERN, "inventory.remove"),
        (_REMOVE_PATTERN, "inventory.remove"),
    ):
        match = pattern.fullmatch(normalized)
        if not match:
            continue
        values: dict[str, object] = {
            key: value.strip() for key, value in match.groupdict().items() if value
        }
        if "quantity" in values:
            values["quantity"] = float(str(values["quantity"]))
        if operation == "inventory.remove":
            values.pop("unit", None)
        return SimpleInventoryOperation(operation, values)
    return None


def match_intent(text: str) -> str:
    normalized = text.strip().lower()

    if any(keyword in normalized for keyword in ("库存", "还有", "剩余", "添加食材")):
        return "inventory"
    if any(keyword in normalized for keyword in ("食谱", "菜谱", "做什么", "推荐一道菜")):
        return "recipe"
    if any(keyword in normalized for keyword in ("订单", "购物车", "下单")):
        return "order_unsupported"
    return "unknown"
