"""Visit-triggered, read-only inventory reminders."""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.config import Settings
from app.context import HistorySegment

if TYPE_CHECKING:
    from app.pipeline import KitchenPipeline


_REMINDER_LIMIT = 100
_MARKDOWN_LIST_PREFIX = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_MARKDOWN_SYNTAX = re.compile(
    r"[*_`~]|\[[^\]\r\n]*\]\([^)\r\n]*\)|(?:^|\s)#{1,6}\s|(?:^|\s)>\s"
)
_ONE_SENTENCE = re.compile(r"^[^。！？.!?]*[。！？.!?]$")
_ACTION_ID = re.compile(r"act_[0-9a-f]{32}", re.IGNORECASE)
_EXECUTED_INVENTORY_PATTERNS = (
    re.compile(
        r"(?:已经|已)(?:成功)?(?:为你)?(?:向|从)?(?:库存)?"
        r"(?:添加|加入|新增|删除|移除|更新|修改|设置|补充|补货|购买|采购|"
        r"入库|出库|清空)"
    ),
    re.compile(
        r"(?:已经|已)(?:成功)?(?:为你)?将.+?"
        r"(?:添加|加入|新增|删除|移除|补充)(?:到|进|至|出)?库存"
    ),
    re.compile(
        r"(?:库存|数量)(?:已经|已)(?:成功)?"
        r"(?:添加|加入|新增|删除|移除|更新|修改|设置|补充|补货)"
    ),
    re.compile(r"(?:库存)?操作(?:已经|已)(?:成功)?执行"),
    re.compile(
        r"\b(?:i(?:'ve| have)|we(?:'ve| have))\s+"
        r"(?:added|removed|updated|restocked|replenished)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:was|were|has been|have been)\s+"
        r"(?:added|removed|updated|restocked|replenished)\b",
        re.IGNORECASE,
    ),
)
_PRIORITY = ("expired", "expiring", "stale", "restock")
ReminderCategory = Literal["expired", "expiring", "stale", "restock"]


class ReminderResponse(BaseModel):
    message: str | None
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
        self.max_input_chars = settings.deepseek_max_input_chars
        self.model = (
            ChatOpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                model=settings.deepseek_model,
                temperature=0,
                timeout=settings.http_timeout_seconds,
                max_retries=1,
                max_tokens=settings.deepseek_max_output_tokens,
            )
            if settings.deepseek_api_key
            else None
        )

    def generate(
        self,
        candidates: dict[str, list[dict[str, object]]],
        inventory: list[dict[str, object]],
        history: list[HistorySegment],
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
                    "补货建议的顺序选择重点；较高优先级存在时必须在提醒中写出其中"
                    "一个食材名称，并原样返回对应的category和item_name。提醒必须以"
                    "该item_name开头。仅返回一句不超过100个Unicode字符的纯文本，"
                    "不得换行、使用Markdown、声称已经更改库存或创建待确认操作；"
                    "没有值得提醒的内容时返回null。\n"
                    f"数据：{payload}"
                )
                raw_response = self.model.with_structured_output(
                    ReminderResponse
                ).invoke(prompt)
                response = ReminderResponse.model_validate(raw_response)
                message = (
                    response.message.strip()
                    if isinstance(response.message, str)
                    else None
                )
                if message is not None and _valid_model_message(
                    message, response, highest_priority
                ):
                    return message
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
        return self.generator.generate(
            candidates,
            self.pipeline.inventory.list_instances(),
            self.pipeline.history.list_recent(user_id),
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


def _valid_model_message(
    message: str,
    response: ReminderResponse,
    highest_priority: tuple[str, list[dict[str, object]]] | None,
) -> bool:
    if not _safe_message(message) or highest_priority is None:
        return False
    category, items = highest_priority
    item_name = (
        response.item_name.strip() if isinstance(response.item_name, str) else ""
    )
    names = {str(item.get("name", "")).strip() for item in items}
    return (
        response.category == category
        and bool(item_name)
        and item_name in names
        and message.startswith(item_name)
    )


def _safe_message(message: str) -> bool:
    return (
        bool(message)
        and len(message) <= _REMINDER_LIMIT
        and "\r" not in message
        and "\n" not in message
        and len(message.splitlines()) == 1
        and _MARKDOWN_LIST_PREFIX.match(message) is None
        and _MARKDOWN_SYNTAX.search(message) is None
        and _ONE_SENTENCE.fullmatch(message) is not None
        and _ACTION_ID.search(message) is None
        and not any(
            pattern.search(message) for pattern in _EXECUTED_INVENTORY_PATTERNS
        )
    )


def _deterministic_fallback(
    candidates: dict[str, list[dict[str, object]]],
) -> str | None:
    for category in _PRIORITY:
        items = candidates.get(category, [])
        if not items:
            continue
        name = str(items[0].get("name", "")).strip()
        if category == "expired":
            suffix = "已过期，请检查后决定是否丢弃。"
            generic = "有食材已过期，请检查后决定是否丢弃。"
        elif category == "expiring":
            suffix = "将在3天内过期，请优先使用。"
            generic = "有食材将在3天内过期，请优先使用。"
        elif category == "stale":
            suffix = "已超过14天未确认，请检查库存状态。"
            generic = "有食材已超过14天未确认，请检查库存状态。"
        else:
            suffix = "可能需要补货，请先确认库存再决定。"
            generic = "有食材可能需要补货，请先确认库存再决定。"
        available = _REMINDER_LIMIT - len(suffix)
        named = f"{name[:available]}{suffix}" if name else generic
        return named if _safe_message(named) else generic
    return None
