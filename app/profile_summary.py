"""Structured autonomous profile inference from bounded history batches."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from app.memory import ConversationHistoryRepository, ConversationRecord
from app.profile import ProfileCategory, ProfileProposal
from app.profile_actions import (
    ProfileActionService,
    ProfileChangeOperation,
    ProfileChangeRequest,
)
from app.prompts import PROFILE_SUMMARY_SYSTEM_PROMPT
from app.structured_output import invoke_validated_json


class ProfileSummaryCandidate(BaseModel):
    operation: ProfileChangeOperation
    category: ProfileCategory
    key: str = Field(min_length=1, max_length=120)
    value: str | None = Field(default=None, max_length=300)
    reason: str = Field(min_length=1, max_length=500)
    evidence_ids: list[str] = Field(default_factory=list, max_length=5)

    def as_request(self) -> ProfileChangeRequest:
        return ProfileChangeRequest(
            operation=self.operation,
            category=self.category,
            key=self.key,
            value=self.value,
            reason=self.reason,
        )


class ProfileSummaryBatch(BaseModel):
    candidates: list[ProfileSummaryCandidate] = Field(default_factory=list, max_length=20)


class ProfileSummarizer:
    def __init__(
        self,
        model: Any,
        history: ConversationHistoryRepository,
        actions: ProfileActionService,
        *,
        callbacks: list[Any] | None = None,
    ) -> None:
        self.callbacks = callbacks or []
        self.model = model
        self.history = history
        self.actions = actions

    def run_if_due(self, user_id: str) -> list[ProfileProposal]:
        cursor = self.actions.repository.read().state.last_summarized_segment_id
        records = self.history.list_completed_after(
            user_id, cursor, limit=10
        )
        if len(records) < 10:
            return []
        try:
            batch = self._invoke(records)
            allowed_ids = {record.segment_id for record in records}
            valid = [
                candidate
                for candidate in batch.candidates
                if len(set(candidate.evidence_ids)) >= 2
                and set(candidate.evidence_ids).issubset(allowed_ids)
            ]
            return self.actions.apply_inferred_batch(
                user_id,
                [candidate.as_request() for candidate in valid],
                {candidate.key: candidate.evidence_ids for candidate in valid},
                records[-1].segment_id,
            )
        except Exception:
            return []

    def _invoke(self, records: list[ConversationRecord]) -> ProfileSummaryBatch:
        payload = [
            {
                "segment_id": record.segment_id,
                "request": record.request,
                "response": record.response,
            }
            for record in records
        ]
        return invoke_validated_json(
            self.model,
            ProfileSummaryBatch,
            [
                {"role": "system", "content": PROFILE_SUMMARY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ],
            callbacks=self.callbacks,
            component="profile_summary",
        )
