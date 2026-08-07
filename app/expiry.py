"""Conservative, structured expiry inference for newly added food."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.config import Settings


class ExpiryPrediction(BaseModel):
    expires_at: datetime | None


class ExpiryInferer(Protocol):
    def infer(self, name: str, now: datetime) -> datetime | None: ...


class DeepSeekExpiryInferer:
    def __init__(self, settings: Settings) -> None:
        self.model = ChatOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_model,
            temperature=0,
            timeout=settings.http_timeout_seconds,
            max_retries=1,
            max_tokens=settings.deepseek_max_output_tokens,
        )

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
        prediction = self.model.with_structured_output(ExpiryPrediction).invoke(prompt)
        expires_at = prediction.expires_at
        if (
            expires_at is None
            or expires_at.tzinfo is None
            or expires_at.utcoffset() is None
            or expires_at <= utc_now
        ):
            return None
        return expires_at
