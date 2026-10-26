"""Visit-triggered, read-only inventory reminders."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from app.llm import ObservedChatOpenAI as ChatOpenAI
from app.llm import chat_model_kwargs
from pydantic import BaseModel

from app.llm_log import callback_store, logged_span, model_callbacks, request_context
from app.config import Settings
from app.context import HistoryExcerpt
from app.structured_output import invoke_validated_json

if TYPE_CHECKING:
    from app.pipeline import KitchenPipeline


_REMINDER_LIMIT = 100
_SAFE_REMINDER_TEMPLATES = {
    "expired": "有食材已过期，请检查后决定是否丢弃。",
    "expiring": "有食材将在3天内过期，请优先使用。",
    "stale": "有食材已超过14天未确认，请检查库存状态。",
    "restock": "有食材可能需要补货，请先确认库存再决定。",
}
_PRIORITY = ("expired", "expiring", "stale", "restock")
ReminderCategory = Literal["expired", "expiring", "stale", "restock"]


class ReminderResponse(BaseModel):
    category: ReminderCategory | None = None
    item_name: str | None = None


class VisitRepository:
    """Atomically record a visit and report whether its prior visit was due."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS user_visits (
                    user_id TEXT PRIMARY KEY,
                    last_visited_at TEXT NOT NULL
                )
                """
            )

    def check_and_touch(self, user_id: str, now: datetime) -> bool:
        if not user_id.strip():
            raise ValueError("user_id must be non-empty")
        reference = _as_utc(now)
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT last_visited_at FROM user_visits WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            connection.execute(
                """
                INSERT INTO user_visits(user_id, last_visited_at) VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE
                SET last_visited_at = excluded.last_visited_at
                """,
                (user_id, reference.isoformat()),
            )
        if row is None:
            return False
        previous = _as_utc(datetime.fromisoformat(str(row[0])))
        return reference - previous > timedelta(hours=24)


class ReminderGenerator:
    """Generate one constrained reminder, with a deterministic safe fallback."""

    def __init__(self, settings: Settings) -> None:
        self.logging_callbacks = model_callbacks(settings, "proactive_reminder")
        self.max_input_chars = settings.llm_max_input_chars
        self.model = (
            ChatOpenAI(**chat_model_kwargs(settings))
            if settings.llm_enabled
            else None
        )

    def generate(
        self,
        candidates: dict[str, list[dict[str, object]]],
        inventory: list[dict[str, object]],
        history: list[HistoryExcerpt],
    ) -> str | None:
        with logged_span(
            callback_store(self.logging_callbacks), "request", "proactive_reminder",
            prompt={"candidates": candidates, "inventory": inventory, "history": history},
        ) as query:
            result = self._generate(candidates, inventory, history)
            query["content"] = result
            return result

    def _generate(
        self,
        candidates: dict[str, list[dict[str, object]]],
        inventory: list[dict[str, object]],
        history: list[HistoryExcerpt],
    ) -> str | None:
        highest_priority = _highest_priority_selection(candidates)
        if self.model is not None:
            try:
                payload = json.dumps(
                    {
                        "candidates": candidates,
                        "inventory": inventory,
                        "recent_history": [
                            segment.model_dump(mode="json") for segment in history
                        ],
                    },
                    ensure_ascii=False,
                    default=str,
                )[: self.max_input_chars]
                prompt = (
                    "你是厨房库存提醒助手。按已过期、3天内临期、超过14天未确认、"
                    "补货建议的顺序选择重点，并原样返回对应的category和item_name；"
                    "前三类都为空时，可以根据完整库存和近期对话选择restock，但"
                    "item_name必须与库存中的一个食材名称完全一致。"
                    "不要生成面向用户的文本，应用会使用固定安全模板渲染提醒。"
                    "没有值得提醒的内容时，两个字段都返回null。\n"
                    f"数据：{payload}"
                )
                response = invoke_validated_json(
                    self.model,
                    ReminderResponse,
                    prompt,
                    callbacks=self.logging_callbacks,
                    component="proactive_reminder",
                )
                selection = _validated_model_selection(
                    response, highest_priority, inventory
                )
                if selection is not None:
                    # Selection data is never interpolated into user-visible text.
                    return _render_reminder(selection[0])
            except Exception:
                pass
        return _deterministic_fallback(candidates)


class ProactiveService:
    def __init__(
        self,
        visits: VisitRepository,
        generator: ReminderGenerator,
        pipeline: KitchenPipeline,
    ) -> None:
        self.visits = visits
        self.generator = generator
        self.pipeline = pipeline

    def check(self, user_id: str, *, now: datetime | None = None) -> str | None:
        reference = _as_utc(now or datetime.now(UTC))
        if not self.visits.check_and_touch(user_id, reference):
            return None
        candidates = self.pipeline.inventory.reminder_candidates(reference)
        with request_context(user_id, "", metadata={"trigger": "proactive_visit"}):
            return self.generator.generate(
                candidates,
                self.pipeline.inventory.list_instances(),
                self.pipeline.history.list_context_window(user_id),
            )


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime values must be timezone-aware")
    return value.astimezone(UTC)


def _highest_priority_selection(
    candidates: dict[str, list[dict[str, object]]],
) -> tuple[str, list[dict[str, object]]] | None:
    for category in _PRIORITY:
        items = candidates.get(category, [])
        if items:
            return category, items
    return None


def _validated_model_selection(
    response: ReminderResponse,
    highest_priority: tuple[str, list[dict[str, object]]] | None,
    inventory: list[dict[str, object]],
) -> tuple[str, str] | None:
    item_name = (
        response.item_name.strip() if isinstance(response.item_name, str) else ""
    )
    if highest_priority is None:
        inventory_names = {
            str(item.get("name", "")).strip() for item in inventory
        }
        if (
            response.category != "restock"
            or not item_name
            or item_name not in inventory_names
        ):
            return None
        return "restock", item_name
    category, items = highest_priority
    names = {str(item.get("name", "")).strip() for item in items}
    if response.category != category or not item_name or item_name not in names:
        return None
    return category, item_name


def _safe_message(message: str) -> bool:
    return (
        message in _SAFE_REMINDER_TEMPLATES.values()
        and len(message) <= _REMINDER_LIMIT
        and len(message.splitlines()) == 1
        and message.endswith(("。", "！", "？", ".", "!", "?"))
        and all(marker not in message.lower() for marker in ("act_", "<", "[", "*"))
    )


def _deterministic_fallback(
    candidates: dict[str, list[dict[str, object]]],
) -> str | None:
    for category in _PRIORITY:
        items = candidates.get(category, [])
        if not items:
            continue
        return _render_reminder(category)
    return None


def _render_reminder(category: str) -> str:
    message = _SAFE_REMINDER_TEMPLATES.get(category)
    if message is not None and _safe_message(message):
        return message
    return _SAFE_REMINDER_TEMPLATES["expired"]
