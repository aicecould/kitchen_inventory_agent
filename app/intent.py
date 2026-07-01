"""Simple intent matching placeholder.

TODO: Add vector intent recognition after this stage when required.
"""

from __future__ import annotations


def match_intent(text: str) -> str:
    normalized = text.strip().lower()

    if any(keyword in normalized for keyword in ("库存", "还有", "剩余", "添加食材")):
        return "inventory"
    if any(keyword in normalized for keyword in ("食谱", "菜谱", "做什么", "推荐一道菜")):
        return "recipe"
    if any(keyword in normalized for keyword in ("订单", "购物车", "下单")):
        return "order_unsupported"
    return "unknown"

