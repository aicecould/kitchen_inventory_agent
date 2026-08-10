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
_FULL_INSTANCE_ID = r"inv_[0-9a-f]{32}"
_REPLACE_INSTANCE_PATTERN = re.compile(
    rf"^(?:将\s*)?(?P<instance_id>{_FULL_INSTANCE_ID})\s*"
    r"(?:更新|修改|替换|设置)为\s*(?P<quantity_text>.+?)[。！？!?]?$"
)
_REMOVE_INSTANCE_PATTERN = re.compile(
    rf"^(?:删除|移除)\s*(?P<instance_id>{_FULL_INSTANCE_ID})[。！？!?]?$"
)
_CONFIRM_INSTANCE_PATTERN = re.compile(
    rf"^(?:确认|校准)\s*(?P<instance_id>{_FULL_INSTANCE_ID})"
    r"(?:\s*(?:仍存在|还在))?[。！？!?]?$"
)
_LIST_INVENTORY_PATTERN = re.compile(
    r"^(?:查询|查看|看看|列出)(?:一下)?(?:当前|现有)?库存(?:情况|列表)?[。！？!?]?$"
)
_GREETING_PATTERN = re.compile(r"^(?:你好|您好|在吗|谢谢|再见)[。！？!?]?$")
_CANCEL_PATTERN = re.compile(r"^(?:取消|不要了|不用了|算了|停止)(?:吧)?[。！？!?]?$")
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

    if _CANCEL_PATTERN.fullmatch(normalized):
        return RouteDecision("supported", "request.cancel", {})
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

    for pattern, operation in (
        (_REPLACE_INSTANCE_PATTERN, "inventory.replace"),
        (_REMOVE_INSTANCE_PATTERN, "inventory.remove"),
        (_CONFIRM_INSTANCE_PATTERN, "inventory.confirm"),
    ):
        match = pattern.fullmatch(normalized)
        if match is not None:
            return RouteDecision(
                "supported",
                operation,
                {
                    key: value.strip()
                    for key, value in match.groupdict().items()
                    if value is not None
                },
            )

    if _CONTEXTUAL_PATTERN.fullmatch(normalized):
        return RouteDecision("contextual", None, {})
    return RouteDecision("undecided", None, {})


def match_simple_inventory_query(text: str) -> bool:
    normalized = " ".join(text.strip().split())
    return _LIST_INVENTORY_PATTERN.fullmatch(normalized) is not None


def match_simple_inventory_operation(text: str) -> SimpleInventoryOperation | None:
    normalized = text.strip()
    match = _ADD_PATTERN.fullmatch(normalized)
    if match is None:
        return None
    return SimpleInventoryOperation(
        "inventory.add",
        {
            key: value.strip()
            for key, value in match.groupdict().items()
            if value is not None
        },
    )


def match_intent(text: str) -> str:
    normalized = text.strip().lower()

    if any(keyword in normalized for keyword in ("库存", "还有", "剩余", "添加食材")):
        return "inventory"
    if any(keyword in normalized for keyword in ("食谱", "菜谱", "做什么", "推荐一道菜")):
        return "recipe"
    if any(keyword in normalized for keyword in ("订单", "购物车", "下单")):
        return "order_unsupported"
    return "unknown"
