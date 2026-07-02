"""Simple intent matching placeholder.

TODO: Add vector intent recognition after this stage when required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SimpleInventoryOperation:
    operation: str
    arguments: dict[str, object]


_UNITS = "个|克|千克|公斤|瓶|袋|盒|根|颗|份"
_ADD_PATTERN = re.compile(
    rf"^(?:添加|增加|加入)(?:库存)?\s*(?P<quantity>\d+(?:\.\d+)?)\s*"
    rf"(?P<unit>{_UNITS})\s*(?P<name>.+?)(?:到库存)?$"
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


def match_simple_inventory_operation(text: str) -> SimpleInventoryOperation | None:
    normalized = " ".join(text.strip().split())
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
