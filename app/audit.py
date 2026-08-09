"""Deterministic output and tool-call audits."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.context import ConfidenceStatus, Ingredient


BLOCK_PATTERNS = (
    re.compile(r"忽略(?:之前|以上)(?:的)?指令"),
    re.compile(r"(?:已经|已)完成(?:付款|下单)"),
)
_MARKDOWN_URL = re.compile(r"\[[^\]]+\]\((https?://[^)\s]+)\)")
_INSTANCE_ID = re.compile(r"^inv_[0-9a-f]{32}$")
_ACTION_ID = re.compile(r"^act_[0-9a-f]{32}$")
_INVENTORY_READ_TOOLS = frozenset(
    {
        "list_inventory_instances",
        "find_inventory_instances",
        "get_inventory_instance",
    }
)
_INVENTORY_PROPOSAL_OPERATIONS = {
    "propose_add_inventory_instance": "inventory.add",
    "propose_replace_inventory_instance": "inventory.replace",
    "propose_remove_inventory_instance": "inventory.remove",
    "propose_remove_all_inventory_instances": "inventory.remove_all",
    "propose_confirm_inventory_instance": "inventory.confirm",
}
_INVENTORY_TOOLS = _INVENTORY_READ_TOOLS | frozenset(
    _INVENTORY_PROPOSAL_OPERATIONS
)
_OPERATION_LABELS = {
    "inventory.add": "库存新增",
    "inventory.replace": "库存替换",
    "inventory.remove": "单项删除",
    "inventory.remove_all": "同名全部删除",
    "inventory.confirm": "库存校准",
}
_INVENTORY_SNAPSHOT_FIELDS = {
    "instance_id",
    "name",
    "quantity_text",
    "stocked_at",
    "expires_at",
    "expiry_source",
    "last_confirmed_at",
}


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


def canonical_tool_response(
    tool_history: list[dict[str, object]],
) -> str | None:
    """Render inventory-bearing tool history without trusting model-authored facts."""

    parsed_calls = [
        {
            **entry,
            "arguments": _as_value(entry.get("arguments")),
            "result": _as_value(entry.get("result")),
        }
        for entry in tool_history
    ]
    return _canonical_response_from_calls(parsed_calls)


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
        if entry.get("tool") in _INVENTORY_PROPOSAL_OPERATIONS
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
    canonical = _canonical_response_from_calls(parsed_calls)
    if canonical is not None:
        expected = canonical if notice is None else f"{canonical}\n\n> {notice}"
        if content != expected:
            return AuditResult(
                False, "Inventory response does not match canonical tool output"
            )

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
        if key not in result_arguments:
            return False
        result_value = result_arguments[key]
        if result_value == value:
            continue
        if key == "expires_at" and _equivalent_datetimes(value, result_value):
            continue
        return False
    return True


def _equivalent_datetimes(left: object, right: object) -> bool:
    parsed: list[datetime] = []
    for value in (left, right):
        if isinstance(value, datetime):
            candidate = value
        elif isinstance(value, str):
            try:
                candidate = datetime.fromisoformat(value)
            except ValueError:
                return False
        else:
            return False
        if candidate.tzinfo is None or candidate.utcoffset() is None:
            return False
        parsed.append(candidate)
    return parsed[0] == parsed[1]


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
    if tool_name in _INVENTORY_PROPOSAL_OPERATIONS:
        if not isinstance(result, dict):
            return "Inventory proposal has no structured result"
        expected_operation = _INVENTORY_PROPOSAL_OPERATIONS[tool_name]
        if result.get("operation") != expected_operation:
            return "Inventory proposal operation differs from tool"
        action_id = result.get("action_id")
        if not isinstance(action_id, str) or _ACTION_ID.fullmatch(action_id) is None:
            return "Inventory proposal has an invalid action ID"
        result_arguments = result.get("arguments")
        preview = result.get("preview")
        if not isinstance(result_arguments, dict) or not isinstance(preview, dict):
            return "Inventory proposal arguments or preview are invalid"
        preview_error = _validate_proposal_preview(
            expected_operation, result_arguments, preview
        )
        if preview_error:
            return preview_error
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


def _validate_inventory_instances(result: list[object]) -> str | None:
    seen_ids: set[str] = set()
    for item in result:
        if not isinstance(item, dict):
            return "Inventory instance result is invalid"
        if set(item) != _INVENTORY_SNAPSHOT_FIELDS:
            return "Inventory instance result has an invalid field contract"
        instance_id = item.get("instance_id")
        if not isinstance(instance_id, str) or _INSTANCE_ID.fullmatch(instance_id) is None:
            return "Inventory instance result has an invalid ID"
        if instance_id in seen_ids:
            return "Inventory instance result IDs are duplicated"
        seen_ids.add(instance_id)
        if (
            not isinstance(item.get("name"), str)
            or not str(item["name"]).strip()
            or not isinstance(item.get("quantity_text"), str)
            or not str(item["quantity_text"]).strip()
            or item.get("expiry_source") not in {"llm", "user", "unknown"}
        ):
            return "Inventory instance result is invalid"
        for field in ("stocked_at", "last_confirmed_at"):
            if not _is_utc_iso(item.get(field)):
                return "Inventory instance result has an invalid UTC timestamp"
        if item.get("expires_at") is not None and not _is_utc_iso(
            item.get("expires_at")
        ):
            return "Inventory instance result has an invalid UTC expiry"
    return None


def _validate_proposal_preview(
    operation: str,
    arguments: dict[str, object],
    preview: dict[str, object],
) -> str | None:
    if operation == "inventory.add":
        proposed = preview.get("proposed")
        if not isinstance(proposed, dict) or proposed != arguments:
            return "Inventory add preview differs from proposal arguments"
        return None

    if operation in {"inventory.replace", "inventory.remove", "inventory.confirm"}:
        current = preview.get("current")
        if not isinstance(current, dict):
            return "Inventory proposal current preview is invalid"
        instance_error = _validate_inventory_instances([current])
        if instance_error:
            return instance_error
        if current.get("instance_id") != arguments.get("instance_id"):
            return "Inventory proposal preview differs from target instance"
        if operation == "inventory.replace":
            proposed = preview.get("proposed")
            if not isinstance(proposed, dict):
                return "Inventory replace proposed preview is invalid"
            expected = {
                "instance_id": arguments.get("instance_id"),
                "name": current.get("name"),
                "quantity_text": arguments.get("quantity_text"),
                "expires_at": arguments.get("expires_at"),
                "expiry_source": arguments.get("expiry_source"),
            }
            if proposed != expected:
                return "Inventory replace preview differs from proposal arguments"
        return None

    if operation == "inventory.remove_all":
        name = arguments.get("name")
        items = preview.get("items")
        count = preview.get("expected_count")
        if preview.get("name") != name or not isinstance(items, list):
            return "Inventory remove-all preview differs from proposal arguments"
        instance_error = _validate_inventory_instances(items)
        if instance_error:
            return instance_error
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
            or count != len(items)
            or any(item.get("name") != name for item in items)
        ):
            return "Inventory remove-all preview count or items are invalid"
        return None
    return "Inventory proposal operation is not supported"


def _canonical_response_from_calls(
    parsed_calls: list[dict[str, object]],
) -> str | None:
    if not any(entry.get("tool") in _INVENTORY_TOOLS for entry in parsed_calls):
        return None

    sections: list[str] = []
    read_lines = _canonical_read_lines(parsed_calls)
    if read_lines:
        sections.append("库存实例快照（只读）：\n" + "\n".join(read_lines))

    proposal_lines = _canonical_proposal_lines(parsed_calls)
    if proposal_lines:
        sections.append("待确认操作（尚未执行）：\n" + "\n".join(proposal_lines))

    recipe_lines = _canonical_recipe_lines(parsed_calls)
    if recipe_lines:
        sections.append("菜谱工具结果：\n" + "\n".join(recipe_lines))
    return "\n\n".join(sections)


def _canonical_read_lines(
    parsed_calls: list[dict[str, object]],
) -> list[str]:
    lines: list[str] = []
    seen_ids: set[str] = set()
    for entry in parsed_calls:
        tool_name = entry.get("tool")
        if tool_name not in _INVENTORY_READ_TOOLS:
            continue
        result = entry.get("result")
        instances: list[dict[str, object]] = []
        if isinstance(result, dict):
            instances = [result]
        elif isinstance(result, list):
            instances = [item for item in result if isinstance(item, dict)]
        for instance in instances:
            instance_id = str(instance.get("instance_id", ""))
            if instance_id in seen_ids:
                continue
            seen_ids.add(instance_id)
            lines.append(f"- {_canonical_snapshot(instance)}")
        if instances:
            continue
        arguments = entry.get("arguments")
        structured_arguments = arguments if isinstance(arguments, dict) else {}
        if tool_name == "list_inventory_instances":
            lines.append("- 当前库存为空。")
        elif tool_name == "find_inventory_instances":
            lines.append(
                f"- 精确名称“{structured_arguments.get('name', '')}”无匹配实例。"
            )
        else:
            lines.append(
                "- 完整实例 ID "
                f"{structured_arguments.get('instance_id', '')} 无匹配实例。"
            )
    return lines


def _canonical_proposal_lines(
    parsed_calls: list[dict[str, object]],
) -> list[str]:
    lines: list[str] = []
    for entry in parsed_calls:
        tool_name = entry.get("tool")
        operation = _INVENTORY_PROPOSAL_OPERATIONS.get(str(tool_name))
        result = entry.get("result")
        if operation is None or not isinstance(result, dict):
            continue
        action_id = result.get("action_id", "")
        lines.append(
            f"- {_OPERATION_LABELS[operation]} | action_id={action_id} | "
            "status=pending（尚未执行）"
        )
        preview = result.get("preview")
        structured_preview = preview if isinstance(preview, dict) else {}
        if operation == "inventory.add":
            proposed = structured_preview.get("proposed")
            value = proposed if isinstance(proposed, dict) else {}
            lines.append(
                "  - 拟新增："
                f"{value.get('name', '')}：{value.get('quantity_text', '')}；"
                f"过期 {_display_value(value.get('expires_at'))}；"
                f"日期来源 {value.get('expiry_source', '')}"
            )
        elif operation == "inventory.replace":
            current = structured_preview.get("current")
            proposed = structured_preview.get("proposed")
            current_value = current if isinstance(current, dict) else {}
            proposed_value = proposed if isinstance(proposed, dict) else {}
            lines.append(f"  - 当前：{_canonical_snapshot(current_value)}")
            lines.append(
                "  - 拟替换："
                f"{proposed_value.get('name', '')} "
                f"[{proposed_value.get('instance_id', '')}]："
                f"{proposed_value.get('quantity_text', '')}；"
                f"过期 {_display_value(proposed_value.get('expires_at'))}；"
                f"日期来源 {proposed_value.get('expiry_source', '')}"
            )
        elif operation in {"inventory.remove", "inventory.confirm"}:
            current = structured_preview.get("current")
            current_value = current if isinstance(current, dict) else {}
            label = "删除目标" if operation == "inventory.remove" else "确认目标"
            lines.append(f"  - {label}：{_canonical_snapshot(current_value)}")
        else:
            lines.append(
                "  - 精确名称 "
                f"{structured_preview.get('name', '')}；预计数量 "
                f"{structured_preview.get('expected_count', '')}"
            )
            items = structured_preview.get("items")
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        lines.append(f"  - 删除目标：{_canonical_snapshot(item)}")
    return lines


def _canonical_recipe_lines(
    parsed_calls: list[dict[str, object]],
) -> list[str]:
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()
    for entry in parsed_calls:
        if entry.get("tool") != "search_recipes":
            continue
        result = entry.get("result")
        if not isinstance(result, list):
            continue
        for item in result:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            source_url = str(item.get("source_url") or "").strip()
            key = (title, source_url)
            if not title or key in seen:
                continue
            seen.add(key)
            source = str(item.get("source") or "未知来源")
            rendered_title = (
                f"[{title}]({source_url})"
                if source_url.startswith(("http://", "https://"))
                else title
            )
            lines.append(f"- {rendered_title}；来源 {source}")
    return lines


def _canonical_snapshot(item: dict[str, object]) -> str:
    return (
        f"{item.get('name', '')} [{item.get('instance_id', '')}]："
        f"{item.get('quantity_text', '')}；"
        f"入库 {_display_value(item.get('stocked_at'))}；"
        f"过期 {_display_value(item.get('expires_at'))}；"
        f"日期来源 {item.get('expiry_source', '')}；"
        f"最后确认 {_display_value(item.get('last_confirmed_at'))}"
    )


def _display_value(value: object) -> str:
    return "未知" if value is None or value == "" else str(value)


def _is_utc_iso(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return (
        parsed.tzinfo is not None
        and parsed.utcoffset() is not None
        and parsed.utcoffset().total_seconds() == 0
    )


def _looks_like_recipe_recommendation(content: str) -> bool:
    return (
        ("菜谱" in content or "食谱" in content or "做法" in content)
        and ("推荐" in content or "可以做" in content or "步骤" in content)
    )
