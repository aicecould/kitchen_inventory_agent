"""Structured values passed into and out of the Agent."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from app.profile import ProfileProposal


class ConfidenceStatus(StrEnum):
    ACCEPTED = "accepted"
    NEEDS_CONFIRMATION = "needs_confirmation"
    REJECTED = "rejected"
    USER_CONFIRMED = "user_confirmed"


class Ingredient(BaseModel):
    name: str
    confidence: float = Field(ge=0.0, le=1.0)
    status: ConfidenceStatus


class HistoryExcerpt(BaseModel):
    request: str
    response: str


class CookingTaskContext(BaseModel):
    task_id: str = Field(pattern=r"^task_[0-9a-f]{32}$")
    query_text: str
    recipe_title: str
    servings: int = Field(gt=0)
    ingredients: list[str] = Field(default_factory=list)


class VisionConfirmationPrompt(BaseModel):
    confirmation_id: str
    candidates: list[Ingredient]


class AgentContext(BaseModel):
    user_id: str
    segment_id: str = ""
    request_text: str
    intent: str = "unknown"
    ingredients: list[Ingredient] = Field(default_factory=list)
    inventory_snapshot: list[dict[str, object]] | None = None
    allergens: list[str] = Field(default_factory=list)
    allergen_intolerances: list[str] = Field(default_factory=list)
    custom_allergens: list[str] = Field(default_factory=list)
    profile_markdown: str = ""
    history_segments: list[HistoryExcerpt] = Field(default_factory=list, max_length=10)
    cooking_task: CookingTaskContext | None = None
    inventory_source_task_id: str | None = Field(
        default=None, pattern=r"^task_[0-9a-f]{32}$"
    )
    vision_confirmation_result: dict[str, object] | None = None


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
    awaiting_vision_confirmation: bool = False
    vision_confirmation: VisionConfirmationPrompt | None = None
    profile_proposals: list[ProfileProposal] = Field(default_factory=list)
    segment_id: str = ""
    recipe_options: list[dict[str, object]] = Field(default_factory=list)

    def public_payload(self) -> dict[str, object]:
        payload = self.model_dump(mode="json", exclude={"tool_history", "audit_reason"})
        if self.awaiting_vision_confirmation:
            state = "awaiting_vision_confirmation"
        elif self.audit_reason in {"task_impossible", "retry_exhausted", "invalid_supervisor_output"}:
            state = "failed"
        elif self.blocked:
            state = "blocked"
        elif any(proposal.status == "pending" for proposal in self.profile_proposals) or any(
            entry.get("status") != "error" and isinstance(entry.get("result"), dict)
            and entry["result"].get("status") == "pending" for entry in self.tool_history
        ):
            state = "awaiting_confirmation"
        else:
            state = "completed"
        payload["state"] = state
        return payload
