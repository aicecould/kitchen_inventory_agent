"""Audited action lifecycle for Markdown-backed profile mutations."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from app.audit import validate_profile_mutation_preview
from app.profile import (
    ProfileAuditEvent,
    ProfileCategory,
    ProfileFact,
    ProfileFactStatus,
    ProfileOrigin,
    ProfileProposal,
    ProfileProposalStatus,
    ProfileRepository,
    ProfileState,
)


class ProfileChangeOperation(StrEnum):
    UPSERT = "upsert"
    REPLACE = "replace"
    DELETE = "delete"


class ProfileChangeSource(StrEnum):
    EXPLICIT_USER_STATEMENT = "explicit_user_statement"
    AUTONOMOUS_SUMMARY = "autonomous_summary"


class ProfileChangeRequest(BaseModel):
    operation: ProfileChangeOperation
    category: ProfileCategory
    key: str = Field(min_length=1, max_length=120)
    value: str | None = Field(default=None, max_length=300)
    reason: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_request(self) -> "ProfileChangeRequest":
        self.key = _one_line(self.key, "profile key")
        self.reason = _one_line(self.reason, "profile reason")
        if self.operation != ProfileChangeOperation.DELETE:
            if self.value is None:
                raise ValueError("profile upserts require a value")
            self.value = _one_line(self.value, "profile value")
        return self


class ProfileActionService:
    def __init__(
        self,
        repository: ProfileRepository,
        *,
        evidence_exists: Callable[[str, str], bool],
        now: Callable[[], datetime] | None = None,
        proposal_ttl_days: int = 30,
    ) -> None:
        self.repository = repository
        self.evidence_exists = evidence_exists
        self.now = now or (lambda: datetime.now(UTC))
        self.proposal_ttl_days = proposal_ttl_days

    def propose(
        self,
        user_id: str,
        request: ProfileChangeRequest,
        evidence_ids: list[str],
        *,
        trusted_current_segment_id: str | None = None,
    ) -> ProfileProposal:
        self._validate_evidence(
            user_id, evidence_ids, trusted_current_segment_id=trusted_current_segment_id
        )
        profile = self.repository.read()
        state = profile.state.model_copy(deep=True)
        now = self._now()
        proposal = self._new_proposal(
            user_id,
            request,
            evidence_ids,
            state,
            now,
            ProfileChangeSource.EXPLICIT_USER_STATEMENT,
        )
        state.proposals.append(proposal)
        self._finish_state(
            state,
            operation="propose_profile_change",
            subject_id=proposal.proposal_id,
            now=now,
        )
        self._audit_preview(request, proposal.preview)
        saved = self.repository.replace(
            expected_revision=profile.state.revision,
            state=state,
            audit_passed=True,
        )
        return _proposal(saved.state, proposal.proposal_id)

    def confirm(self, proposal_id: str, user_id: str) -> ProfileProposal:
        profile = self.repository.read()
        state = profile.state.model_copy(deep=True)
        proposal = _proposal(state, proposal_id)
        self._authorize(proposal, user_id)
        if proposal.status == ProfileProposalStatus.CONFIRMED:
            return proposal
        if proposal.status != ProfileProposalStatus.PENDING:
            raise ValueError(f"Profile proposal cannot be confirmed from status: {proposal.status}")
        now = self._now()
        if proposal.expires_at <= now:
            proposal.status = ProfileProposalStatus.EXPIRED
            proposal.resolved_at = now
            self._finish_state(state, operation="expire_profile_proposals", subject_id=proposal_id, now=now)
            self.repository.replace(
                expected_revision=profile.state.revision, state=state, audit_passed=True
            )
            raise ValueError("Profile proposal has expired")
        current = _active_fact(state, proposal.category, proposal.key)
        if proposal.conflicting_fact_id is not None and (
            current is None or current.fact_id != proposal.conflicting_fact_id
        ):
            raise ValueError("Profile proposal preview is stale")
        if current is not None:
            current.status = ProfileFactStatus.SUPERSEDED
            current.updated_at = now
        if proposal.operation != "delete":
            fact = _new_fact(
                proposal.category,
                proposal.key,
                proposal.value or "",
                ProfileOrigin.USER_CONFIRMED,
                proposal.evidence_ids,
                proposal.reason,
                now,
            )
            if current is not None:
                current.replaced_by = fact.fact_id
            state.facts.append(fact)
        proposal.status = ProfileProposalStatus.CONFIRMED
        proposal.resolved_at = now
        proposal.resolved_revision = profile.state.revision + 1
        self._finish_state(state, operation="confirm_profile_change", subject_id=proposal_id, now=now)
        saved = self.repository.replace(
            expected_revision=profile.state.revision, state=state, audit_passed=True
        )
        return _proposal(saved.state, proposal_id)

    def reject(self, proposal_id: str, user_id: str) -> ProfileProposal:
        profile = self.repository.read()
        state = profile.state.model_copy(deep=True)
        proposal = _proposal(state, proposal_id)
        self._authorize(proposal, user_id)
        if proposal.status == ProfileProposalStatus.REJECTED:
            return proposal
        if proposal.status != ProfileProposalStatus.PENDING:
            raise ValueError(f"Profile proposal cannot be rejected from status: {proposal.status}")
        now = self._now()
        proposal.status = ProfileProposalStatus.REJECTED
        proposal.resolved_at = now
        proposal.resolved_revision = profile.state.revision + 1
        self._finish_state(state, operation="reject_profile_change", subject_id=proposal_id, now=now)
        saved = self.repository.replace(
            expected_revision=profile.state.revision, state=state, audit_passed=True
        )
        return _proposal(saved.state, proposal_id)

    def apply_inferred_batch(
        self,
        user_id: str,
        candidates: list[ProfileChangeRequest],
        evidence_by_key: dict[str, list[str]],
        summarized_through: str,
    ) -> list[ProfileProposal]:
        profile = self.repository.read()
        state = profile.state.model_copy(deep=True)
        now = self._now()
        created: list[str] = []
        for request in candidates:
            evidence = evidence_by_key.get(request.key, [])
            if len(set(evidence)) < 2:
                raise ValueError("Autonomous profile inference requires two distinct evidence segments")
            self._validate_evidence(user_id, evidence)
            current = _active_fact(state, request.category, request.key)
            requires_proposal = (
                request.category == ProfileCategory.REQUIREMENT
                or (current is not None and current.origin == ProfileOrigin.USER_CONFIRMED)
            )
            if requires_proposal:
                proposal = self._new_proposal(
                    user_id,
                    request,
                    evidence,
                    state,
                    now,
                    ProfileChangeSource.AUTONOMOUS_SUMMARY,
                )
                state.proposals.append(proposal)
                created.append(proposal.proposal_id)
                self._audit_preview(request, proposal.preview)
                continue
            if current is not None:
                current.status = ProfileFactStatus.SUPERSEDED
                current.updated_at = now
            if request.operation != ProfileChangeOperation.DELETE:
                self._ensure_capacity(state, request.category)
                fact = _new_fact(
                    request.category,
                    request.key,
                    request.value or "",
                    ProfileOrigin.INFERRED,
                    evidence,
                    request.reason,
                    now,
                )
                if current is not None:
                    current.replaced_by = fact.fact_id
                state.facts.append(fact)
        state.last_summarized_segment_id = summarized_through
        self._finish_state(state, operation="apply_inferred_profile_change", subject_id=summarized_through, now=now)
        saved = self.repository.replace(
            expected_revision=profile.state.revision, state=state, audit_passed=True
        )
        return [_proposal(saved.state, proposal_id) for proposal_id in created]

    def record_audit_only(self, user_id: str, operation: str) -> None:
        if not user_id:
            raise ValueError("user_id is required")
        profile = self.repository.read()
        state = profile.state.model_copy(deep=True)
        self._finish_state(state, operation=operation, subject_id=None, now=self._now())
        self.repository.replace(
            expected_revision=profile.state.revision, state=state, audit_passed=True
        )

    def _new_proposal(
        self,
        user_id: str,
        request: ProfileChangeRequest,
        evidence_ids: list[str],
        state: ProfileState,
        now: datetime,
        source: ProfileChangeSource,
    ) -> ProfileProposal:
        current = _active_fact(state, request.category, request.key)
        proposal_id = f"prof_{uuid4().hex}"
        preview: dict[str, object] = {
            "before_revision": state.revision,
            "after_revision": state.revision + 1,
            "proposal_id": proposal_id,
            "source": source.value,
            "evidence_ids": evidence_ids,
            "current": current.model_dump(mode="json") if current is not None else None,
            "proposed": {
                "category": request.category.value,
                "key": request.key,
                "value": request.value,
                "operation": request.operation.value,
            },
        }
        return ProfileProposal(
            proposal_id=proposal_id,
            user_id=user_id,
            operation=request.operation.value,
            category=request.category,
            key=request.key,
            value=request.value,
            conflicting_fact_id=current.fact_id if current is not None else None,
            source=source.value,
            evidence_ids=evidence_ids,
            reason=request.reason,
            base_revision=state.revision,
            created_at=now,
            expires_at=now + timedelta(days=self.proposal_ttl_days),
            preview=preview,
        )

    def _audit_preview(self, request: ProfileChangeRequest, preview: dict[str, object]) -> None:
        reason = validate_profile_mutation_preview(
            request.model_dump(mode="json"), preview
        )
        if reason is not None:
            raise ValueError(reason)

    def _validate_evidence(
        self,
        user_id: str,
        evidence_ids: list[str],
        *,
        trusted_current_segment_id: str | None = None,
    ) -> None:
        if not 1 <= len(evidence_ids) <= 5 or len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("Profile evidence must contain one to five unique IDs")
        for evidence_id in evidence_ids:
            if evidence_id == trusted_current_segment_id:
                continue
            if not self.evidence_exists(user_id, evidence_id):
                raise ValueError("Profile evidence does not exist for this user")

    def _finish_state(
        self,
        state: ProfileState,
        *,
        operation: str,
        subject_id: str | None,
        now: datetime,
    ) -> None:
        state.schema_version = 2
        state.revision += 1
        state.audit_events.append(
            ProfileAuditEvent(
                operation=operation,
                status="passed",
                reason="validated by existing audit rules",
                created_at=now,
                subject_id=subject_id,
            )
        )
        state.audit_events = state.audit_events[-200:]

    def _ensure_capacity(self, state: ProfileState, category: ProfileCategory) -> None:
        active = [
            fact
            for fact in state.facts
            if fact.category == category and fact.status == ProfileFactStatus.ACTIVE
        ]
        if len(active) < 20:
            return
        inferred = [fact for fact in active if fact.origin == ProfileOrigin.INFERRED]
        if not inferred:
            raise ValueError("Profile category contains 20 confirmed facts")
        oldest = min(inferred, key=lambda fact: (fact.updated_at, fact.fact_id))
        oldest.status = ProfileFactStatus.SUPERSEDED
        oldest.updated_at = self._now()

    def _authorize(self, proposal: ProfileProposal, user_id: str) -> None:
        if proposal.user_id != user_id:
            raise PermissionError("Profile proposal belongs to another user")

    def _now(self) -> datetime:
        value = self.now()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Profile action time must be timezone-aware")
        return value.astimezone(UTC)


def _proposal(state: ProfileState, proposal_id: str) -> ProfileProposal:
    for proposal in state.proposals:
        if proposal.proposal_id == proposal_id:
            return proposal
    raise KeyError("Profile proposal not found")


def _active_fact(
    state: ProfileState, category: ProfileCategory, key: str
) -> ProfileFact | None:
    for fact in state.facts:
        if (
            fact.category == category
            and fact.key.casefold() == key.casefold()
            and fact.status == ProfileFactStatus.ACTIVE
        ):
            return fact
    return None


def _new_fact(
    category: ProfileCategory,
    key: str,
    value: str,
    origin: ProfileOrigin,
    evidence_ids: list[str],
    reason: str,
    now: datetime,
) -> ProfileFact:
    return ProfileFact(
        fact_id=f"fact_{uuid4().hex}",
        category=category,
        key=key,
        value=value,
        origin=origin,
        evidence_ids=evidence_ids[-5:],
        reason=reason,
        created_at=now,
        updated_at=now,
    )


def _one_line(value: str, label: str) -> str:
    cleaned = value.strip()
    if not cleaned or "\n" in cleaned or "\r" in cleaned:
        raise ValueError(f"{label} must be one non-empty line")
    return cleaned
