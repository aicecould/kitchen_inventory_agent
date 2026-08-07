"""Deterministic output and tool-call audits."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any

from app.context import ConfidenceStatus, Ingredient


BLOCK_PATTERNS = (
    re.compile(r"忽略(?:之前|以上)(?:的)?指令"),
    re.compile(r"(?:已经|已)完成(?:付款|下单)"),
)
_EXECUTED_INVENTORY_PATTERNS = (
    re.compile(r"(?:已经|已)(?:成功)?(?:向库存)?(?:添加|加入|新增)"),
    re.compile(r"(?:库存|数量)(?:已经|已)(?:成功)?(?:更新|修改|设置)"),
    re.compile(r"(?:已经|已)(?:成功)?从库存(?:删除|移除)"),
    re.compile(r"(?:库存)?操作(?:已经|已)(?:成功)?执行"),
)
_MARKDOWN_URL = re.compile(r"\[[^\]]+\]\((https?://[^)\s]+)\)")
_INSTANCE_ID = re.compile(r"^inv_[0-9a-f]{32}$")
_INVENTORY_LINE = re.compile(
    r"^[-*]\s+(?P<name>.+?)\s+\[(?P<instance_id>[^\]\r\n]+)\]"
    r"[：:](?P<quantity_text>[^\r\n]+)$",
    re.MULTILINE,
)
_INVENTORY_READ_TOOLS = frozenset(
    {
        "list_inventory_instances",
        "find_inventory_instances",
        "get_inventory_instance",
    }
)


@dataclass(frozen=True, slots=True)
class AuditResult:
    passed: bool
    reason: str | None = None


def regex_audit(content: str) -> AuditResult:
    if not content.strip():
        return AuditResult(False, "Output is empty")
    for pattern in BLOCK_PATTERNS:
        if pattern.search(content):
            return AuditResult(False, f"Matched blocked pattern: {pattern.pattern}")
    return AuditResult(True)


def confidence_notice(ingredients: list[Ingredient]) -> str | None:
    uncertain = [
        item.name
        for item in ingredients
        if item.status == ConfidenceStatus.NEEDS_CONFIRMATION
    ]
    if not uncertain:
        return None
    names = "、".join(f"“{name}”" for name in dict.fromkeys(uncertain))
    return (
        f"识别提醒：图片中的{names}置信度中等，属于不确定识别结果，"
        "请核对食材名称。"
    )


def append_confidence_notice(content: str, ingredients: list[Ingredient]) -> str:
    notice = confidence_notice(ingredients)
    if notice is None or notice in content:
        return content
    return f"{content.rstrip()}\n\n> {notice}"


def review_tool_calls(
    content: str,
    tool_history: list[dict[str, object]],
    *,
    ingredients: list[Ingredient] | None = None,
    vision_confirmation_result: dict[str, object] | None = None,
) -> AuditResult:
    """Check claims against recorded calls and image-confidence decisions."""

    ingredient_values = ingredients or []
    if any(item.status == ConfidenceStatus.REJECTED for item in ingredient_values):
        return AuditResult(False, "Low-confidence image result reached the Agent")

    notice = confidence_notice(ingredient_values)
    if notice is not None and notice not in content:
        return AuditResult(False, "Medium-confidence image warning is missing")

    confirmed = {
        item.name
        for item in ingredient_values
        if item.status == ConfidenceStatus.USER_CONFIRMED
    }
    if confirmed:
        if vision_confirmation_result is None:
            return AuditResult(False, "Confirmed image result has no confirmation record")
        selected = {
            str(value.get("name", "")).strip()
            for value in vision_confirmation_result.get("selected", [])  # type: ignore[union-attr]
            if isinstance(value, dict)
        }
        if not confirmed.issubset(selected):
            return AuditResult(False, "Confirmed image result differs from user selection")

    parsed_calls = [
        {
            **entry,
            "arguments": _as_value(entry.get("arguments")),
            "result": _as_value(entry.get("result")),
        }
        for entry in tool_history
    ]
    for entry in parsed_calls:
        record_error = _validate_tool_record(entry)
        if record_error:
            return AuditResult(False, record_error)

    proposals = [
        entry
        for entry in parsed_calls
        if str(entry.get("tool", "")).startswith("propose_")
    ]
    for entry in proposals:
        result = entry["result"]
        arguments = entry["arguments"]
        if not isinstance(result, dict):
            return AuditResult(False, "Inventory proposal has no structured result")
        if result.get("status") != "pending":
            return AuditResult(False, "Inventory proposal result is not pending")
        result_arguments = result.get("arguments")
        if (
            isinstance(arguments, dict)
            and isinstance(result_arguments, dict)
            and not _arguments_match(arguments, result_arguments)
        ):
            return AuditResult(False, "Inventory proposal arguments differ from tool result")
    if proposals and any(
        pattern.search(content) for pattern in _EXECUTED_INVENTORY_PATTERNS
    ):
        return AuditResult(False, "Pending inventory operation was described as executed")

    inventory_snapshots: list[object] = []
    for entry in parsed_calls:
        if entry.get("tool") not in _INVENTORY_READ_TOOLS:
            continue
        result = entry["result"]
        if isinstance(result, dict):
            inventory_snapshots.append(result)
        elif isinstance(result, list):
            inventory_snapshots.extend(result)
    mismatch = _inventory_claim_mismatch(content, inventory_snapshots)
    if mismatch:
        return AuditResult(False, mismatch)

    recipe_calls = [
        entry for entry in parsed_calls if entry.get("tool") == "search_recipes"
    ]
    for entry in recipe_calls:
        result = entry["result"]
        if not isinstance(result, list):
            continue
        recipes = [item for item in result if isinstance(item, dict)]
        allowed_urls = {
            str(item[key])
            for item in recipes
            for key in ("source_url", "image_url")
            if item.get(key)
        }
        unexpected_urls = set(_MARKDOWN_URL.findall(content)) - allowed_urls
        if unexpected_urls:
            return AuditResult(False, "Recipe answer contains a URL absent from tool results")
        if recipes and _looks_like_recipe_recommendation(content):
            titles = [str(item.get("title", "")).strip() for item in recipes]
            if not any(title and title.casefold() in content.casefold() for title in titles):
                return AuditResult(False, "Recommended recipe is absent from tool results")

    return AuditResult(True)


def review_with_secondary_model() -> None:
    """Reserved for the independent audit Agent."""
    raise NotImplementedError


def _as_value(value: object) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(stripped)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    return value


def _arguments_match(call_arguments: dict[str, object], result_arguments: dict[str, object]) -> bool:
    for key, value in call_arguments.items():
        if key not in result_arguments or result_arguments[key] != value:
            return False
    return True


def _validate_tool_record(entry: dict[str, object]) -> str | None:
    tool_name = str(entry.get("tool", ""))
    arguments = entry.get("arguments")
    result = entry.get("result")
    if not isinstance(arguments, dict):
        return f"Tool arguments are not structured: {tool_name}"
    if tool_name == "list_inventory_instances":
        if arguments or not isinstance(result, list):
            return "Inventory list parameters or result are invalid"
        return _validate_inventory_instances(result)
    if tool_name == "find_inventory_instances":
        requested_name = arguments.get("name")
        if (
            not isinstance(requested_name, str)
            or not requested_name.strip()
            or not isinstance(result, list)
        ):
            return "Inventory find parameters or result are invalid"
        result_error = _validate_inventory_instances(result)
        if result_error:
            return result_error
        if any(item.get("name") != requested_name for item in result):
            return "Inventory instances differ from requested name"
    if tool_name == "get_inventory_instance":
        requested_id = arguments.get("instance_id")
        if not isinstance(requested_id, str) or _INSTANCE_ID.fullmatch(requested_id) is None:
            return "Inventory instance request has an invalid ID"
        if result is not None and not isinstance(result, dict):
            return "Inventory instance result is invalid"
        if isinstance(result, dict):
            result_error = _validate_inventory_instances([result])
            if result_error:
                return result_error
            if result.get("instance_id") != requested_id:
                return "Inventory instance result differs from requested ID"
    if tool_name == "search_recipes":
        requested_ingredients = arguments.get("ingredients")
        if not isinstance(requested_ingredients, list) or not requested_ingredients:
            return "Recipe search has invalid ingredient parameters"
        if not isinstance(result, list):
            return "Recipe search result is not structured"
        if any(
            not isinstance(item, dict) or not str(item.get("title", "")).strip()
            for item in result
        ):
            return "Recipe search returned an invalid recipe record"
    return None


def _inventory_claim_mismatch(content: str, result: object) -> str | None:
    if isinstance(result, dict):
        instances = [result]
    elif isinstance(result, list):
        instances = result
    else:
        return None
    actual: set[tuple[str, str, str]] = set()
    for item in instances:
        if not isinstance(item, dict):
            continue
        instance_id = str(item.get("instance_id", ""))
        for displayed_id in {instance_id, instance_id[:8]}:
            actual.add(
                (
                    str(item.get("name", "")),
                    displayed_id,
                    str(item.get("quantity_text", "")),
                )
            )
    for match in _INVENTORY_LINE.finditer(content):
        claim = (
            match.group("name"),
            match.group("instance_id"),
            match.group("quantity_text"),
        )
        if claim not in actual:
            return "Inventory value in answer differs from tool result"
    return None


def _validate_inventory_instances(result: list[object]) -> str | None:
    seen_ids: set[str] = set()
    for item in result:
        if not isinstance(item, dict):
            return "Inventory instance result is invalid"
        instance_id = item.get("instance_id")
        if not isinstance(instance_id, str) or _INSTANCE_ID.fullmatch(instance_id) is None:
            return "Inventory instance result has an invalid ID"
        if instance_id in seen_ids:
            return "Inventory instance result IDs are duplicated"
        seen_ids.add(instance_id)
        if not isinstance(item.get("name"), str) or not isinstance(
            item.get("quantity_text"), str
        ):
            return "Inventory instance result is invalid"
    return None


def _looks_like_recipe_recommendation(content: str) -> bool:
    return (
        ("菜谱" in content or "食谱" in content or "做法" in content)
        and ("推荐" in content or "可以做" in content or "步骤" in content)
    )
