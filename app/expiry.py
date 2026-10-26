"""Conservative, structured expiry inference for newly added food."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

from app.llm import ObservedChatOpenAI as ChatOpenAI
from pydantic import BaseModel

from app.llm_log import model_callbacks
from app.config import Settings
from app.llm import chat_model_kwargs
from app.structured_output import invoke_validated_json


class ExpiryPrediction(BaseModel):
    expires_at: datetime | None


class ExpiryInferer(Protocol):
    def infer(self, name: str, now: datetime) -> datetime | None: ...


class DeepSeekExpiryInferer:
    def __init__(self, settings: Settings) -> None:
        self.logging_callbacks = model_callbacks(settings, "expiry_inference")
        self.model = ChatOpenAI(**chat_model_kwargs(settings))

    def infer(self, name: str, now: datetime) -> datetime | None:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        utc_now = now.astimezone(UTC)
        prompt = (
            "请仅根据食物名称和当前 UTC 时间，推断保守的家庭储存过期时间。"
            "如果无法可靠推断，请返回 null。\n"
            f"食物名称：{name}\n"
            f"当前 UTC 时间：{utc_now.isoformat()}"
        )
        prediction = invoke_validated_json(
            self.model,
            ExpiryPrediction,
            prompt,
            callbacks=self.logging_callbacks,
            component="expiry_inference",
        )
        expires_at = prediction.expires_at
        if (
            expires_at is None
            or expires_at.tzinfo is None
            or expires_at.utcoffset() is None
            or expires_at <= utc_now
        ):
            return None
        return expires_at
