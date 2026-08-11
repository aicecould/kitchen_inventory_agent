"""Deterministic output and tool-call audits."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any
from urllib.parse import urlsplit

from app.context import ConfidenceStatus, Ingredient


BLOCK_PATTERNS = (
    re.compile(r"忽略(?:之前|以上)(?:的)?指令"),
    re.compile(r"(?:已经|已)完成(?:付款|下单)"),
)
_MARKDOWN_URL = re.compile(r"\[[^\]]+\]\((https?://[^)\s]+)\)")
_INSTANCE_ID = re.compile(r"^inv_[0-9a-f]{32}$")
_ACTION_ID = re.compile(r"^act_[0-9a-f]{32}$")
_PROFILE_ID = re.compile(r"^prof_[0-9a-f]{32}$")
_HISTORY_ID = re.compile(r"^hist_[0-9a-f]{32}$")
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
_PROFILE_PROPOSAL_TOOLS = frozenset({"propose_profile_change"})
_OPERATION_LABELS = {
    "zh": {
        "inventory.add": "库存新增",
        "inventory.replace": "库存替换",
        "inventory.remove": "单项删除",
        "inventory.remove_all": "同名全部删除",
        "inventory.confirm": "库存校准",
    },
    "en": {
        "inventory.add": "Add inventory item",
        "inventory.replace": "Replace inventory item",
        "inventory.remove": "Remove inventory item",
        "inventory.remove_all": "Remove all exact-name items",
        "inventory.confirm": "Confirm inventory item",
    },
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


class AuditCode(StrEnum):
    PASSED = "passed"
    USER_CANCELLED = "user_cancelled"
    TASK_IMPOSSIBLE = "task_impossible"
    RECIPE_TOOL_MISSING = "recipe_tool_missing"
    RECIPE_RESULT_MISMATCH = "recipe_result_mismatch"
    REPAIRABLE_OUTPUT_VIOLATION = "repairable_output_violation"
    INVALID_SUPERVISOR_OUTPUT = "invalid_supervisor_output"
    RETRY_EXHAUSTED = "retry_exhausted"


@dataclass(frozen=True, slots=True)
class AuditResult:
    passed: bool
    reason: str | None = None
    code: AuditCode = AuditCode.PASSED


def audit_failure(
    reason: str,
    code: AuditCode = AuditCode.REPAIRABLE_OUTPUT_VIOLATION,
) -> AuditResult:
    return AuditResult(False, reason, code)


def regex_audit(content: str) -> AuditResult:
    if not content.strip():
        return audit_failure("Output is empty")
    for pattern in BLOCK_PATTERNS:
        if pattern.search(content):
            return audit_failure(f"Matched blocked pattern: {pattern.pattern}")
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
    *,
    target_language: str = "zh",
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
    return _canonical_response_from_calls(parsed_calls, target_language)


def review_tool_calls(
    content: str,
    tool_history: list[dict[str, object]],
    *,
    target_language: str = "zh",
    ingredients: list[Ingredient] | None = None,
    vision_confirmation_result: dict[str, object] | None = None,
) -> AuditResult:
    """Check claims against recorded calls and image-confidence decisions."""

    ingredient_values = ingredients or []
    if any(item.status == ConfidenceStatus.REJECTED for item in ingredient_values):
        return audit_failure("Low-confidence image result reached the Agent")

    notice = confidence_notice(ingredient_values)
    if notice is not None and notice not in content:
        return audit_failure("Medium-confidence image warning is missing")

    confirmed = {
        item.name
        for item in ingredient_values
        if item.status == ConfidenceStatus.USER_CONFIRMED
    }
    if confirmed:
        if vision_confirmation_result is None:
            return audit_failure("Confirmed image result has no confirmation record")
        selected = {
            str(value.get("name", "")).strip()
            for value in vision_confirmation_result.get("selected", [])  # type: ignore[union-attr]
            if isinstance(value, dict)
        }
        if not confirmed.issubset(selected):
            return audit_failure("Confirmed image result differs from user selection")

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
            return audit_failure(record_error)

    proposals = [
        entry
        for entry in parsed_calls
        if entry.get("tool") in _INVENTORY_PROPOSAL_OPERATIONS
    ]
    for entry in proposals:
        result = entry["result"]
        arguments = entry["arguments"]
        if not isinstance(result, dict):
            return audit_failure("Inventory proposal has no structured result")
        if result.get("status") != "pending":
            return audit_failure("Inventory proposal result is not pending")
        result_arguments = result.get("arguments")
        if (
            isinstance(arguments, dict)
            and isinstance(result_arguments, dict)
            and not _arguments_match(arguments, result_arguments)
        ):
                return audit_failure("Inventory proposal arguments differ from tool result")
    profile_proposals = [
        entry for entry in parsed_calls if entry.get("tool") in _PROFILE_PROPOSAL_TOOLS
    ]
    for entry in profile_proposals:
        result = entry["result"]
        arguments = entry["arguments"]
        if not isinstance(result, dict):
            return audit_failure("Profile proposal has no structured result")
        if result.get("status") != "pending":
            return audit_failure("Profile proposal result is not pending")
        if (
            isinstance(arguments, dict)
            and not _arguments_match(arguments, result)
        ):
            return audit_failure("Profile proposal arguments differ from tool result")
    canonical = _canonical_response_from_calls(parsed_calls, target_language)
    if canonical is not None:
        expected = canonical if notice is None else f"{canonical}\n\n> {notice}"
        if content != expected:
            return audit_failure(
                "Inventory response does not match canonical tool output"
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
            return audit_failure(
                "Recipe answer contains a URL absent from tool results",
                AuditCode.RECIPE_RESULT_MISMATCH,
            )
        if recipes and _looks_like_recipe_recommendation(content):
            titles = [str(item.get("title", "")).strip() for item in recipes]
            if not any(title and title.casefold() in content.casefold() for title in titles):
                return audit_failure(
                    "Recommended recipe is absent from tool results",
                    AuditCode.RECIPE_RESULT_MISMATCH,
                )

    return AuditResult(True)


def review_candidate(
    content: str,
    tool_history: list[dict[str, object]],
    *,
    intent: str = "unknown",
    target_language: str = "zh",
    ingredients: list[Ingredient] | None = None,
    vision_confirmation_result: dict[str, object] | None = None,
) -> AuditResult:
    """Return one stable review classification for supervisor orchestration."""

    has_recipe_tool = any(
        entry.get("tool") == "search_recipes" for entry in tool_history
    )
    if (
        intent == "recipe"
        and _looks_like_recipe_recommendation(content)
        and not has_recipe_tool
    ):
        return audit_failure(
            "Recipe recommendation has no search_recipes tool record",
            AuditCode.RECIPE_TOOL_MISSING,
        )

    tool_result = review_tool_calls(
        content,
        tool_history,
        target_language=target_language,
        ingredients=ingredients,
        vision_confirmation_result=vision_confirmation_result,
    )
    if not tool_result.passed:
        if tool_result.code == AuditCode.RECIPE_RESULT_MISMATCH:
            return tool_result
        return audit_failure(
            tool_result.reason or "Tool-call audit failed",
            AuditCode.REPAIRABLE_OUTPUT_VIOLATION,
        )

    regex_result = regex_audit(content)
    if not regex_result.passed:
        return regex_result
    return AuditResult(True)


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
    if tool_name in _PROFILE_PROPOSAL_TOOLS:
        if not isinstance(result, dict):
            return "Profile proposal has no structured result"
        proposal_id = result.get("proposal_id")
        if not isinstance(proposal_id, str) or _PROFILE_ID.fullmatch(proposal_id) is None:
            return "Profile proposal has an invalid ID"
        preview = result.get("preview")
        if not isinstance(preview, dict):
            return "Profile proposal preview is invalid"
        preview_error = validate_profile_mutation_preview(arguments, preview)
        if preview_error:
            return preview_error
    if tool_name == "search_recipes":
        requested_ingredients = arguments.get("ingredients")
        if not isinstance(requested_ingredients, list) or not requested_ingredients:
            return "Recipe search has invalid ingredient parameters"
        if "ingredient_mapping" not in arguments or not isinstance(
            arguments["ingredient_mapping"], list
        ):
            return "Recipe search has invalid ingredient mapping parameters"
        if "intolerances" in arguments or "custom_allergen_mapping" in arguments:
            return "Recipe search uses removed allergen parameters"
        if not isinstance(result, list):
            return "Recipe search result is not structured"
        for item in result:
            if not isinstance(item, dict):
                return "Recipe search returned an invalid recipe record"
            if (
                not isinstance(item.get("id"), str)
                or not item["id"].strip()
                or not str(item.get("title", "")).strip()
            ):
                return "Recipe search returned an invalid recipe record"
            for field in ("used_ingredient_count", "missing_ingredient_count"):
                value = item.get(field)
                if value is not None and (not isinstance(value, int) or value < 0):
                    return "Recipe search returned an invalid ingredient-count field"
            rrf_score = item.get("rrf_score")
            if rrf_score is not None and (
                not isinstance(rrf_score, (int, float)) or rrf_score <= 0
            ):
                return "Recipe search returned an invalid RRF field"
            for field in ("fts_rank", "semantic_rank"):
                value = item.get(field)
                if value is not None and (not isinstance(value, int) or value <= 0):
                    return "Recipe search returned an invalid rank field"
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


def validate_profile_mutation_preview(
    arguments: dict[str, object], preview: dict[str, object]
) -> str | None:
    """Validate a profile before/after preview before and after persistence."""

    before_revision = preview.get("before_revision")
    after_revision = preview.get("after_revision")
    if (
        not isinstance(before_revision, int)
        or isinstance(before_revision, bool)
        or not isinstance(after_revision, int)
        or isinstance(after_revision, bool)
        or after_revision != before_revision + 1
    ):
        return "Profile preview revision is invalid"
    operation = arguments.get("operation")
    if operation not in {"upsert", "replace", "delete"}:
        return "Profile operation is invalid"
    category = arguments.get("category")
    if category not in {
        "kitchen_setup",
        "preference",
        "requirement",
        "default_context",
    }:
        return "Profile category is invalid"
    key = arguments.get("key")
    if not isinstance(key, str) or not key.strip() or "\n" in key or "\r" in key:
        return "Profile key is invalid"
    value = arguments.get("value")
    if operation != "delete" and (
        not isinstance(value, str) or not value.strip() or "\n" in value or "\r" in value
    ):
        return "Profile value is invalid"
    evidence = preview.get("evidence_ids")
    if (
        not isinstance(evidence, list)
        or not 1 <= len(evidence) <= 5
        or len(evidence) != len(set(evidence))
        or any(not isinstance(item, str) or _HISTORY_ID.fullmatch(item) is None for item in evidence)
    ):
        return "Profile evidence is invalid"
    proposal_id = preview.get("proposal_id")
    if proposal_id is not None and (
        not isinstance(proposal_id, str) or _PROFILE_ID.fullmatch(proposal_id) is None
    ):
        return "Profile proposal ID is invalid"
    source = preview.get("source")
    if source == "autonomous_summary" and category == "requirement" and proposal_id is None:
        return "Mandatory profile requirements cannot be inferred silently"
    return None


def validate_profile_state_transition(
    before: dict[str, object],
    after: dict[str, object],
    operation: str,
) -> str | None:
    """Apply the existing deterministic audit contract to every profile write."""

    allowed_operations = {
        "initialize_profile_v2",
        "propose_profile_change",
        "confirm_profile_change",
        "reject_profile_change",
        "expire_profile_proposals",
        "apply_inferred_profile_change",
        "replace_confirmed_allergens",
        "prune_profile_state",
    }
    if operation not in allowed_operations:
        return "Profile mutation operation is not allowlisted"
    before_revision = before.get("revision")
    after_revision = after.get("revision")
    if (
        not isinstance(before_revision, int)
        or isinstance(before_revision, bool)
        or not isinstance(after_revision, int)
        or isinstance(after_revision, bool)
        or after_revision != before_revision + 1
    ):
        return "Profile state revision is invalid"
    if after.get("schema_version") != 2:
        return "Profile state schema version is invalid"
    audit_events = after.get("audit_events")
    if not isinstance(audit_events, list) or not audit_events:
        return "Profile mutation has no audit event"
    event = audit_events[-1]
    if (
        not isinstance(event, dict)
        or event.get("operation") != operation
        or event.get("status") != "passed"
    ):
        return "Profile audit event does not match the mutation"
    before_facts = {
        str(item.get("fact_id")): item
        for item in before.get("facts", [])
        if isinstance(item, dict)
    } if isinstance(before.get("facts"), list) else {}
    after_facts = after.get("facts")
    if not isinstance(after_facts, list):
        return "Profile facts are not structured"
    for item in after_facts:
        if not isinstance(item, dict):
            return "Profile fact is not structured"
        fact_id = str(item.get("fact_id", ""))
        evidence = item.get("evidence_ids")
        if fact_id not in before_facts and (
            not isinstance(evidence, list)
            or not evidence
            or any(not isinstance(value, str) or _HISTORY_ID.fullmatch(value) is None for value in evidence)
        ):
            return "New profile fact has no valid evidence"
        if (
            operation == "apply_inferred_profile_change"
            and item.get("category") == "requirement"
            and item.get("origin") == "inferred"
            and fact_id not in before_facts
        ):
            return "Mandatory profile requirements cannot be inferred silently"
    if operation == "apply_inferred_profile_change":
        after_by_id = {
            str(item.get("fact_id")): item
            for item in after_facts
            if isinstance(item, dict)
        }
        for fact_id, old in before_facts.items():
            current = after_by_id.get(fact_id)
            if (
                old.get("origin") == "user_confirmed"
                and old.get("status") == "active"
                and isinstance(current, dict)
                and current.get("status") != "active"
            ):
                return "Confirmed profile fact cannot be silently replaced"
    return None


def _canonical_response_from_calls(
    parsed_calls: list[dict[str, object]],
    target_language: str = "zh",
) -> str | None:
    if not any(
        entry.get("tool") in _INVENTORY_TOOLS | _PROFILE_PROPOSAL_TOOLS
        for entry in parsed_calls
    ):
        return None

    language = _canonical_language(target_language)
    sections: list[str] = []
    read_lines = _canonical_read_lines(parsed_calls, language)
    if read_lines:
        heading = (
            "库存实例快照（只读）："
            if language == "zh"
            else "Inventory snapshots (read-only):"
        )
        sections.append(heading + "\n" + "\n".join(read_lines))

    proposal_lines = _canonical_proposal_lines(parsed_calls, language)
    if proposal_lines:
        heading = (
            "待确认操作（尚未执行）："
            if language == "zh"
            else "Pending actions (not executed):"
        )
        sections.append(heading + "\n" + "\n".join(proposal_lines))

    profile_lines = _canonical_profile_proposal_lines(parsed_calls, language)
    if profile_lines:
        heading = (
            "待确认画像变更（尚未写入）："
            if language == "zh"
            else "Pending profile changes (not saved):"
        )
        sections.append(heading + "\n" + "\n".join(profile_lines))

    recipe_lines = _canonical_recipe_lines(parsed_calls, language)
    if recipe_lines is not None:
        heading = "菜谱工具结果：" if language == "zh" else "Recipe tool results:"
        sections.append(heading + "\n" + "\n".join(recipe_lines))
    return "\n\n".join(sections)


def _canonical_profile_proposal_lines(
    parsed_calls: list[dict[str, object]], language: str
) -> list[str]:
    lines: list[str] = []
    for entry in parsed_calls:
        if entry.get("tool") not in _PROFILE_PROPOSAL_TOOLS:
            continue
        result = entry.get("result")
        if not isinstance(result, dict):
            continue
        operation = result.get("operation", "")
        category = result.get("category", "")
        key = result.get("key", "")
        value = result.get("value")
        proposal_id = result.get("proposal_id", "")
        if language == "zh":
            target = "删除" if operation == "delete" else str(value or "")
            lines.append(
                f"- 画像变更 | proposal_id={proposal_id} | status=pending（尚未写入）"
            )
            lines.append(f"  - {category} / {key}：{target}")
        else:
            target = "delete" if operation == "delete" else str(value or "")
            lines.append(
                f"- Profile change | proposal_id={proposal_id} | status=pending (not saved)"
            )
            lines.append(f"  - {category} / {key}: {target}")
    return lines


def _canonical_read_lines(
    parsed_calls: list[dict[str, object]],
    language: str,
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
            lines.append(f"- {_canonical_snapshot(instance, language)}")
        if instances:
            continue
        arguments = entry.get("arguments")
        structured_arguments = arguments if isinstance(arguments, dict) else {}
        if tool_name == "list_inventory_instances":
            lines.append(
                "- 当前库存为空。"
                if language == "zh"
                else "- The inventory is empty."
            )
        elif tool_name == "find_inventory_instances":
            name = structured_arguments.get("name", "")
            lines.append(
                f"- 精确名称“{name}”无匹配实例。"
                if language == "zh"
                else f'- No instance matches the exact name "{name}".'
            )
        else:
            instance_id = structured_arguments.get("instance_id", "")
            lines.append(
                f"- 完整实例 ID {instance_id} 无匹配实例。"
                if language == "zh"
                else f"- No instance matches full instance ID {instance_id}."
            )
    return lines


def _canonical_proposal_lines(
    parsed_calls: list[dict[str, object]],
    language: str,
) -> list[str]:
    lines: list[str] = []
    for entry in parsed_calls:
        tool_name = entry.get("tool")
        operation = _INVENTORY_PROPOSAL_OPERATIONS.get(str(tool_name))
        result = entry.get("result")
        if operation is None or not isinstance(result, dict):
            continue
        action_id = result.get("action_id", "")
        pending_status = (
            "status=pending（尚未执行）"
            if language == "zh"
            else "status=pending (not executed)"
        )
        lines.append(
            f"- {_OPERATION_LABELS[language][operation]} | action_id={action_id} | "
            f"{pending_status}"
        )
        preview = result.get("preview")
        structured_preview = preview if isinstance(preview, dict) else {}
        if operation == "inventory.add":
            proposed = structured_preview.get("proposed")
            value = proposed if isinstance(proposed, dict) else {}
            if language == "zh":
                lines.append(
                    "  - 拟新增："
                    f"{value.get('name', '')}：{value.get('quantity_text', '')}；"
                    f"过期 {_display_value(value.get('expires_at'), language)}；"
                    f"日期来源 {value.get('expiry_source', '')}"
                )
            else:
                lines.append(
                    "  - Proposed addition: "
                    f"{value.get('name', '')}: {value.get('quantity_text', '')}; "
                    f"expires {_display_value(value.get('expires_at'), language)}; "
                    f"expiry source {value.get('expiry_source', '')}"
                )
        elif operation == "inventory.replace":
            current = structured_preview.get("current")
            proposed = structured_preview.get("proposed")
            current_value = current if isinstance(current, dict) else {}
            proposed_value = proposed if isinstance(proposed, dict) else {}
            current_label = "当前" if language == "zh" else "Current"
            lines.append(
                f"  - {current_label}: {_canonical_snapshot(current_value, language)}"
            )
            if language == "zh":
                lines.append(
                    "  - 拟替换："
                    f"{proposed_value.get('name', '')} "
                    f"[{proposed_value.get('instance_id', '')}]："
                    f"{proposed_value.get('quantity_text', '')}；"
                    f"过期 {_display_value(proposed_value.get('expires_at'), language)}；"
                    f"日期来源 {proposed_value.get('expiry_source', '')}"
                )
            else:
                lines.append(
                    "  - Proposed replacement: "
                    f"{proposed_value.get('name', '')} "
                    f"[{proposed_value.get('instance_id', '')}]: "
                    f"{proposed_value.get('quantity_text', '')}; "
                    f"expires {_display_value(proposed_value.get('expires_at'), language)}; "
                    f"expiry source {proposed_value.get('expiry_source', '')}"
                )
        elif operation in {"inventory.remove", "inventory.confirm"}:
            current = structured_preview.get("current")
            current_value = current if isinstance(current, dict) else {}
            if language == "zh":
                label = "删除目标" if operation == "inventory.remove" else "确认目标"
            else:
                label = "Removal target" if operation == "inventory.remove" else "Confirmation target"
            lines.append(
                f"  - {label}: {_canonical_snapshot(current_value, language)}"
            )
        else:
            if language == "zh":
                lines.append(
                    "  - 精确名称 "
                    f"{structured_preview.get('name', '')}；预计数量 "
                    f"{structured_preview.get('expected_count', '')}"
                )
            else:
                lines.append(
                    "  - Exact name "
                    f"{structured_preview.get('name', '')}; expected count "
                    f"{structured_preview.get('expected_count', '')}"
                )
            items = structured_preview.get("items")
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        target_label = "删除目标" if language == "zh" else "Removal target"
                        lines.append(
                            f"  - {target_label}: {_canonical_snapshot(item, language)}"
                        )
    return lines


def _canonical_recipe_lines(
    parsed_calls: list[dict[str, object]],
    language: str,
) -> list[str] | None:
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()
    searched = False
    for entry in parsed_calls:
        if entry.get("tool") != "search_recipes":
            continue
        result = entry.get("result")
        if not isinstance(result, list):
            continue
        searched = True
        for item in result:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            source_url = str(item.get("source_url") or "").strip()
            key = (title, source_url)
            if not title or key in seen:
                continue
            seen.add(key)
            source = str(
                item.get("source")
                or ("未知来源" if language == "zh" else "Unknown source")
            ).strip()
            rendered_title = (
                f"[{title}]({source_url})"
                if _is_safe_http_url(source_url)
                else title
            )
            ingredients = item.get("ingredients")
            ingredient_values = (
                [
                    _single_line(str(value))
                    for value in ingredients
                    if _single_line(str(value))
                ]
                if isinstance(ingredients, list)
                else []
            )
            instructions = _single_line(str(item.get("instructions") or ""))
            if language == "zh":
                lines.extend(
                    [
                        f"- {rendered_title}",
                        f"  - 食材：{'、'.join(ingredient_values) or '未提供'}",
                        f"  - 步骤：{instructions or '未提供'}",
                        f"  - 来源：{source}",
                    ]
                )
            else:
                lines.extend(
                    [
                        f"- {rendered_title}",
                        f"  - Ingredients: {'; '.join(ingredient_values) or 'Not provided'}",
                        f"  - Instructions: {instructions or 'Not provided'}",
                        f"  - Source: {source}",
                    ]
                )
    if not searched:
        return None
    if lines:
        return lines
    return [
        "- 未找到符合条件的菜谱。"
        if language == "zh"
        else "- No matching recipes were found."
    ]


def _canonical_snapshot(item: dict[str, object], language: str) -> str:
    if language == "zh":
        return (
            f"{item.get('name', '')} [{item.get('instance_id', '')}]："
            f"{item.get('quantity_text', '')}；"
            f"入库 {_display_value(item.get('stocked_at'), language)}；"
            f"过期 {_display_value(item.get('expires_at'), language)}；"
            f"日期来源 {item.get('expiry_source', '')}；"
            f"最后确认 {_display_value(item.get('last_confirmed_at'), language)}"
        )
    return (
        f"{item.get('name', '')} [{item.get('instance_id', '')}]: "
        f"{item.get('quantity_text', '')}; "
        f"stocked {_display_value(item.get('stocked_at'), language)}; "
        f"expires {_display_value(item.get('expires_at'), language)}; "
        f"expiry source {item.get('expiry_source', '')}; "
        f"last confirmed {_display_value(item.get('last_confirmed_at'), language)}"
    )


def _display_value(value: object, language: str) -> str:
    if value is None or value == "":
        return "未知" if language == "zh" else "unknown"
    return str(value)


def _canonical_language(target_language: str) -> str:
    return "zh" if str(target_language).strip().casefold().startswith("zh") else "en"


def _single_line(value: str) -> str:
    return " ".join(value.split())


def _is_safe_http_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return parsed.scheme.casefold() in {"http", "https"} and bool(parsed.netloc)


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
    chinese = (
        ("菜谱" in content or "食谱" in content or "做法" in content)
        and ("推荐" in content or "可以做" in content or "步骤" in content)
    )
    normalized = content.casefold()
    english = (
        "recipe" in normalized or "dish" in normalized
    ) and any(term in normalized for term in ("recommend", "steps", "instructions"))
    return chinese or english
