from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.profile import ProfileFact, ProfileRepository, ProfileState, UserProfile
from app.profile_actions import (
    ProfileActionService,
    ProfileChangeRequest,
)


NOW = datetime(2026, 8, 10, 5, 0, tzinfo=UTC)


def _history_id(value: str) -> str:
    return f"hist_{value * 32}"


def _service(tmp_path: Path, *, now: datetime = NOW) -> ProfileActionService:
    known = {_history_id("1"), _history_id("2"), _history_id("3")}
    return ProfileActionService(
        ProfileRepository(tmp_path / "profile.md"),
        evidence_exists=lambda user_id, evidence_id: user_id == "u" and evidence_id in known,
        now=lambda: now,
    )


def _request(
    *,
    category: str = "preference",
    key: str = "ingredient.cilantro",
    value: str | None = "不喜欢香菜",
    operation: str = "upsert",
) -> ProfileChangeRequest:
    return ProfileChangeRequest.model_validate(
        {
            "category": category,
            "key": key,
            "value": value,
            "operation": operation,
            "reason": "用户明确表达长期偏好",
        }
    )


def test_explicit_change_stays_pending_until_confirmed(tmp_path: Path) -> None:
    service = _service(tmp_path)
    proposal = service.propose("u", _request(), [_history_id("1")])

    assert proposal.status == "pending"
    assert service.repository.read().preferences == []

    resolved = service.confirm(proposal.proposal_id, "u")
    profile = service.repository.read()

    assert resolved.status == "confirmed"
    assert profile.preferences == ["不喜欢香菜"]
    assert profile.state.facts[0].origin == "user_confirmed"
    assert service.confirm(proposal.proposal_id, "u").status == "confirmed"


def test_reject_is_idempotent_and_never_activates_fact(tmp_path: Path) -> None:
    service = _service(tmp_path)
    proposal = service.propose("u", _request(), [_history_id("1")])

    assert service.reject(proposal.proposal_id, "u").status == "rejected"
    assert service.reject(proposal.proposal_id, "u").status == "rejected"
    assert service.repository.read().preferences == []


def test_profile_actions_reject_foreign_or_missing_evidence(tmp_path: Path) -> None:
    service = _service(tmp_path)

    with pytest.raises(ValueError, match="evidence"):
        service.propose("other", _request(), [_history_id("1")])
    with pytest.raises(ValueError, match="evidence"):
        service.propose("u", _request(), [_history_id("9")])


def test_current_reserved_segment_is_the_only_unpersisted_evidence_allowed(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    reserved = _history_id("9")

    proposal = service.propose(
        "u",
        _request(),
        [reserved],
        trusted_current_segment_id=reserved,
    )

    assert proposal.evidence_ids == [reserved]
    with pytest.raises(ValueError, match="evidence"):
        service.propose(
            "u",
            _request(key="ingredient.onion"),
            [_history_id("8")],
            trusted_current_segment_id=reserved,
        )


def test_expired_proposal_cannot_be_confirmed(tmp_path: Path) -> None:
    service = _service(tmp_path)
    proposal = service.propose("u", _request(), [_history_id("1")])
    expired_service = _service(tmp_path, now=NOW + timedelta(days=31))

    with pytest.raises(ValueError, match="expired"):
        expired_service.confirm(proposal.proposal_id, "u")

    stored = expired_service.repository.read().state.proposals[0]
    assert stored.status == "expired"


def test_inference_replaces_only_inferred_fact(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.apply_inferred_batch(
        "u",
        [_request(value="最近喜欢土豆", key="ingredient.potato")],
        {"ingredient.potato": [_history_id("1"), _history_id("2")]},
        _history_id("2"),
    )

    proposals = service.apply_inferred_batch(
        "u",
        [_request(value="最近不喜欢土豆", key="ingredient.potato")],
        {"ingredient.potato": [_history_id("2"), _history_id("3")]},
        _history_id("3"),
    )

    profile = service.repository.read()
    assert proposals == []
    assert profile.preferences == ["最近不喜欢土豆"]
    assert [fact.status.value for fact in profile.state.facts] == [
        "superseded",
        "active",
    ]


def test_inference_conflicting_with_confirmed_fact_creates_proposal(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    explicit = service.propose("u", _request(), [_history_id("1")])
    service.confirm(explicit.proposal_id, "u")

    proposals = service.apply_inferred_batch(
        "u",
        [_request(value="最近喜欢香菜")],
        {"ingredient.cilantro": [_history_id("2"), _history_id("3")]},
        _history_id("3"),
    )

    assert len(proposals) == 1
    assert proposals[0].status == "pending"
    assert service.repository.read().preferences == ["不喜欢香菜"]


def test_inferred_requirement_always_creates_proposal(tmp_path: Path) -> None:
    service = _service(tmp_path)

    proposals = service.apply_inferred_batch(
        "u",
        [_request(category="requirement", key="allergen.custom.虾", value="对虾过敏")],
        {"allergen.custom.虾": [_history_id("1"), _history_id("2")]},
        _history_id("2"),
    )

    assert proposals[0].status == "pending"
    assert service.repository.read().requirements == []


def test_inference_requires_two_distinct_evidence_segments(tmp_path: Path) -> None:
    service = _service(tmp_path)

    with pytest.raises(ValueError, match="two distinct"):
        service.apply_inferred_batch(
            "u",
            [_request()],
            {"ingredient.cilantro": [_history_id("1")]},
            _history_id("1"),
        )


def test_audit_history_is_bounded_to_two_hundred(tmp_path: Path) -> None:
    service = _service(tmp_path)
    repository = service.repository
    state = ProfileState(
        revision=0,
        audit_events=[],
    )
    repository.replace(
        expected_revision=0,
        state=state.model_copy(update={"revision": 1}),
        audit_passed=True,
    )
    for index in range(205):
        service.record_audit_only("u", f"test.{index}")

    assert len(repository.read().state.audit_events) == 200
