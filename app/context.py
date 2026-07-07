"""Structured values passed into and out of the Agent."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ConfidenceStatus(StrEnum):
    ACCEPTED = "accepted"
    NEEDS_CONFIRMATION = "needs_confirmation"
    REJECTED = "rejected"


class Ingredient(BaseModel):
    name: str
    confidence: float = Field(ge=0.0, le=1.0)
    status: ConfidenceStatus


class AgentContext(BaseModel):
    user_id: str
    request_text: str
    intent: str = "unknown"
    ingredients: list[Ingredient] = Field(default_factory=list)
    allergens: list[str] = Field(default_factory=list)
    allergen_intolerances: list[str] = Field(default_factory=list)
    custom_allergens: list[str] = Field(default_factory=list)
    profile_markdown: str = ""
    history_summary: str = ""
    tool_history: list[dict[str, object]] = Field(default_factory=list)


class ExecutionTraceEvent(BaseModel):
    stage: str
    name: str
    status: str
    detail: str
    duration_ms: int = Field(default=0, ge=0)


class AgentResult(BaseModel):
    content: str
    blocked: bool = False
    audit_reason: str | None = None
    tool_history: list[dict[str, object]] = Field(default_factory=list)
    execution_trace: list[ExecutionTraceEvent] = Field(default_factory=list)
