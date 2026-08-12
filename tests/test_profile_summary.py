from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.memory import ConversationHistoryRepository
from app.profile import ProfileRepository
from app.profile_actions import ProfileActionService, ProfileChangeRequest
from app.profile_summary import (
    ProfileSummarizer,
    ProfileSummaryBatch,
    ProfileSummaryCandidate,
)


class FakeStructuredModel:
    def __init__(self, result: ProfileSummaryBatch | Exception) -> None:
        self.result = result
        self.calls = 0
        self.messages: list[object] = []
        self.structured_method: str | None = None

    def with_structured_output(
        self, schema: object, *, method: str
    ) -> "FakeStructuredModel":
        assert schema is ProfileSummaryBatch
        self.structured_method = method
        return self

    def invoke(self, messages: object, config: object = None) -> object:
        self.calls += 1
        self.messages.append(messages)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _stack(
    tmp_path: Path, result: ProfileSummaryBatch | Exception
) -> tuple[
    ConversationHistoryRepository,
    ProfileActionService,
    FakeStructuredModel,
    ProfileSummarizer,
]:
    history = ConversationHistoryRepository(tmp_path / "history.db")
    history.initialize()
    profile = ProfileRepository(tmp_path / "profile.md")
    actions = ProfileActionService(
        profile,
        evidence_exists=lambda user_id, segment_id: bool(
            history.get_evidence(user_id, [segment_id])
        ),
        now=lambda: datetime(2026, 8, 10, 6, 0, tzinfo=UTC),
    )
    model = FakeStructuredModel(result)
    summarizer = ProfileSummarizer(model, history, actions)
    return history, actions, model, summarizer


def _append(
    history: ConversationHistoryRepository, count: int, *, start: int = 0
) -> list[str]:
    ids: list[str] = []
    for index in range(start, start + count):
        record = history.append(
            user_id="u",
            request=f"第 {index} 次想吃土豆",
            response="推荐土豆料理",
            created_at=datetime(2026, 8, 1, tzinfo=UTC) + timedelta(minutes=index),
        )
        ids.append(record.segment_id)
    return ids


def test_summarizer_waits_for_ten_completed_records(tmp_path: Path) -> None:
    history, actions, model, summarizer = _stack(
        tmp_path, ProfileSummaryBatch(candidates=[])
    )
    ids = _append(history, 9)

    assert summarizer.run_if_due("u") == []
    assert model.calls == 0

    ids.extend(_append(history, 1, start=9))
    assert summarizer.run_if_due("u") == []
    assert model.calls == 1
    assert actions.repository.read().state.last_summarized_segment_id == ids[-1]


def test_summarizer_applies_supported_inference_with_real_evidence(
    tmp_path: Path,
) -> None:
    history, actions, model, summarizer = _stack(
        tmp_path, ProfileSummaryBatch(candidates=[])
    )
    ids = _append(history, 10)
    model.result = ProfileSummaryBatch(
        candidates=[
            ProfileSummaryCandidate(
                operation="upsert",
                category="preference",
                key="ingredient.potato",
                value="最近喜欢土豆",
                reason="多次选择土豆",
                evidence_ids=[ids[2], ids[7]],
            )
        ]
    )

    proposals = summarizer.run_if_due("u")

    assert proposals == []
    assert actions.repository.read().preferences == ["最近喜欢土豆"]


def test_summarizer_drops_fabricated_or_single_evidence_candidates(
    tmp_path: Path,
) -> None:
    history, actions, model, summarizer = _stack(
        tmp_path, ProfileSummaryBatch(candidates=[])
    )
    ids = _append(history, 10)
    model.result = ProfileSummaryBatch(
        candidates=[
            ProfileSummaryCandidate(
                operation="upsert",
                category="preference",
                key="ingredient.potato",
                value="最近喜欢土豆",
                reason="证据不足",
                evidence_ids=[ids[0]],
            ),
            ProfileSummaryCandidate(
                operation="upsert",
                category="preference",
                key="ingredient.onion",
                value="最近喜欢洋葱",
                reason="伪造证据",
                evidence_ids=[ids[1], "hist_" + "9" * 32],
            ),
        ]
    )

    assert summarizer.run_if_due("u") == []
    assert actions.repository.read().preferences == []
    assert actions.repository.read().state.last_summarized_segment_id == ids[-1]


def test_summarizer_failure_keeps_cursor_for_retry(tmp_path: Path) -> None:
    history, actions, model, summarizer = _stack(tmp_path, RuntimeError("offline"))
    _append(history, 10)

    assert summarizer.run_if_due("u") == []

    assert actions.repository.read().state.last_summarized_segment_id is None
    assert model.calls == 1


def test_summarizer_turns_requirement_into_pending_proposal(tmp_path: Path) -> None:
    history, _actions, model, summarizer = _stack(
        tmp_path, ProfileSummaryBatch(candidates=[])
    )
    ids = _append(history, 10)
    model.result = ProfileSummaryBatch(
        candidates=[
            ProfileSummaryCandidate(
                operation="upsert",
                category="requirement",
                key="allergen.custom.shrimp",
                value="对虾过敏",
                reason="对话中重复提及",
                evidence_ids=[ids[0], ids[9]],
            )
        ]
    )

    proposals = summarizer.run_if_due("u")

    assert len(proposals) == 1
    assert proposals[0].status == "pending"
