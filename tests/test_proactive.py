from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier

import pytest

from app.config import get_settings
from app.context import HistorySegment
from app.pipeline import KitchenPipeline, build_pipeline
from app.proactive import (
    ProactiveService,
    ReminderGenerator,
    ReminderResponse,
    VisitRepository,
)


class FakeChatOpenAI:
    response = ReminderResponse(message=None)
    failure: Exception | None = None
    instances: list["FakeChatOpenAI"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.schema: type[ReminderResponse] | None = None
        self.prompt: str | None = None
        self.instances.append(self)

    def with_structured_output(
        self, schema: type[ReminderResponse]
    ) -> "FakeChatOpenAI":
        self.schema = schema
        return self

    def invoke(self, prompt: str) -> ReminderResponse:
        self.prompt = prompt
        if self.failure is not None:
            raise self.failure
        return self.response


@pytest.fixture(autouse=True)
def reset_fake_model() -> None:
    FakeChatOpenAI.response = ReminderResponse(message=None)
    FakeChatOpenAI.failure = None
    FakeChatOpenAI.instances.clear()


def build_generator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> ReminderGenerator:
    monkeypatch.setattr("app.proactive.ChatOpenAI", FakeChatOpenAI)
    settings = replace(
        get_settings(),
        deepseek_api_key="test-key",
        deepseek_model="deepseek-test",
        deepseek_base_url="https://deepseek.invalid",
        inventory_db_path=tmp_path / "inventory.db",
        user_profile_path=tmp_path / "profile.md",
        http_timeout_seconds=7,
        deepseek_max_output_tokens=123,
        deepseek_max_input_chars=4567,
        food_classifier_enabled=False,
    )
    return ReminderGenerator(settings)


def build_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProactiveService, KitchenPipeline]:
    settings = replace(
        get_settings(),
        deepseek_api_key="",
        inventory_db_path=tmp_path / "inventory.db",
        user_profile_path=tmp_path / "profile.md",
        food_classifier_enabled=False,
    )
    pipeline = build_pipeline(settings)
    visits = VisitRepository(settings.inventory_db_path)
    visits.initialize()
    generator = build_generator(tmp_path, monkeypatch)
    return ProactiveService(visits, generator, pipeline), pipeline


def candidate(name: str) -> dict[str, object]:
    return {
        "instance_id": f"inv_{'a' * 32}",
        "name": name,
        "quantity_text": "1盒",
        "expires_at": "2026-08-07T08:59:59+00:00",
        "expiry_source": "user",
        "created_at": "2026-08-01T09:00:00+00:00",
        "updated_at": "2026-08-01T09:00:00+00:00",
        "last_confirmed_at": "2026-08-01T09:00:00+00:00",
    }


def empty_candidates() -> dict[str, list[dict[str, object]]]:
    return {"expired": [], "expiring": [], "stale": []}


def test_visit_check_uses_strict_24_hour_boundary_and_refresh_suppression(
    tmp_path: Path,
) -> None:
    repository = VisitRepository(tmp_path / "inventory.db")
    repository.initialize()
    now = datetime(2026, 8, 7, 9, tzinfo=UTC)

    assert repository.check_and_touch("u", now) is False
    assert repository.check_and_touch("u", now + timedelta(hours=24)) is False
    assert (
        repository.check_and_touch("u", now + timedelta(hours=48, seconds=1))
        is True
    )
    assert (
        repository.check_and_touch("u", now + timedelta(hours=48, seconds=2))
        is False
    )


def test_visit_check_is_atomic_and_isolated_per_user(tmp_path: Path) -> None:
    repository = VisitRepository(tmp_path / "inventory.db")
    repository.initialize()
    now = datetime(2026, 8, 7, 9, tzinfo=UTC)
    repository.check_and_touch("u1", now)
    repository.check_and_touch("u2", now + timedelta(hours=23))
    due = now + timedelta(hours=24, seconds=1)
    barrier = Barrier(2)

    def simultaneous_touch() -> bool:
        barrier.wait()
        return repository.check_and_touch("u1", due)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: simultaneous_touch(), range(2)))

    assert sorted(results) == [False, True]
    assert repository.check_and_touch("u2", due) is False


def test_valid_model_reminder_uses_existing_settings_and_recent_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeChatOpenAI.response = ReminderResponse(
        message="牛奶即将过期，请优先使用。",
        category="expiring",
        item_name="牛奶",
    )
    generator = build_generator(tmp_path, monkeypatch)

    message = generator.generate(
        {"expired": [], "expiring": [candidate("牛奶")], "stale": []},
        [candidate("牛奶")],
        [HistorySegment(request="昨天买了什么？", response="买了牛奶。")],
    )

    assert message == "牛奶即将过期，请优先使用。"
    model = FakeChatOpenAI.instances[0]
    assert model.schema is ReminderResponse
    assert model.kwargs == {
        "api_key": "test-key",
        "base_url": "https://deepseek.invalid",
        "model": "deepseek-test",
        "temperature": 0,
        "timeout": 7,
        "max_retries": 1,
        "max_tokens": 123,
    }
    assert model.prompt is not None
    assert "昨天买了什么？" in model.prompt
    assert "牛奶" in model.prompt


def test_model_reminder_accepts_exactly_100_unicode_characters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeChatOpenAI.response = ReminderResponse(
        message="鲜" * 99 + "。",
        category="restock",
        item_name="鲜",
    )
    generator = build_generator(tmp_path, monkeypatch)

    message = generator.generate(
        {**empty_candidates(), "restock": [candidate("鲜")]},
        [candidate("鲜")],
        [],
    )

    assert message == "鲜" * 99 + "。"
    assert len(message) == 100


@pytest.mark.parametrize(
    "invalid_message",
    [
        "鲜" * 100 + "。",
        "牛奶快过期。\n请优先使用。",
        "牛奶快过期，\u2028请优先使用。",
        "- 牛奶快过期，请优先使用。",
        "牛奶快过期。请优先使用。",
        "已为你向库存添加牛奶。",
        "已将牛奶加入库存。",
        f"请确认待办 act_{'b' * 32}。",
    ],
)
def test_invalid_model_reminder_falls_back_to_highest_priority_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_message: str,
) -> None:
    FakeChatOpenAI.response = ReminderResponse(
        message=invalid_message,
        category="expired",
        item_name="牛奶",
    )
    generator = build_generator(tmp_path, monkeypatch)

    message = generator.generate(
        {
            "expired": [candidate("牛奶")],
            "expiring": [candidate("酸奶")],
            "stale": [candidate("冷冻玉米")],
        },
        [candidate("牛奶"), candidate("酸奶"), candidate("冷冻玉米")],
        [],
    )

    assert message == "牛奶已过期，请检查后决定是否丢弃。"
    assert len(message) <= 100
    assert "\r" not in message and "\n" not in message


@pytest.mark.parametrize(
    ("name", "invalid_message"),
    [
        ("牛奶", "**牛奶**已过期。"),
        ("牛奶", f"牛奶待办act_{'c' * 32}待确认。"),
        ("牛奶", "牛奶已入库。"),
        ("Milk", "Milk was added to inventory."),
    ],
)
def test_inline_markdown_adjacent_ids_and_execution_claims_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    invalid_message: str,
) -> None:
    FakeChatOpenAI.response = ReminderResponse(
        message=invalid_message,
        category="expired",
        item_name=name,
    )
    generator = build_generator(tmp_path, monkeypatch)

    message = generator.generate(
        {"expired": [candidate(name)], "expiring": [], "stale": []},
        [candidate(name)],
        [],
    )

    assert message == f"{name}已过期，请检查后决定是否丢弃。"


def test_model_must_address_highest_priority_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeChatOpenAI.response = ReminderResponse(
        message="冷冻玉米很久未确认，请检查库存。",
        category="stale",
        item_name="冷冻玉米",
    )
    generator = build_generator(tmp_path, monkeypatch)

    message = generator.generate(
        {
            "expired": [candidate("牛奶")],
            "expiring": [],
            "stale": [candidate("冷冻玉米")],
        },
        [candidate("牛奶"), candidate("冷冻玉米")],
        [],
    )

    assert message == "牛奶已过期，请检查后决定是否丢弃。"


@pytest.mark.parametrize(
    ("category", "item_name"),
    [("stale", "玉米"), ("expired", "米")],
)
def test_candidate_selection_rejects_lower_priority_substring_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    category: str,
    item_name: str,
) -> None:
    FakeChatOpenAI.response = ReminderResponse.model_validate(
        {
            "category": category,
            "item_name": item_name,
            "message": "玉米已超过14天未确认，请检查库存。",
        }
    )
    generator = build_generator(tmp_path, monkeypatch)

    message = generator.generate(
        {
            "expired": [candidate("米")],
            "expiring": [],
            "stale": [candidate("玉米")],
            "restock": [],
        },
        [candidate("米"), candidate("玉米")],
        [],
    )

    assert message == "米已过期，请检查后决定是否丢弃。"


def test_provider_failure_without_deterministic_candidates_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeChatOpenAI.failure = RuntimeError("provider secret response")
    generator = build_generator(tmp_path, monkeypatch)

    assert generator.generate(empty_candidates(), [], []) is None


def test_restock_is_the_lowest_priority_and_has_a_deterministic_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeChatOpenAI.failure = RuntimeError("provider unavailable")
    generator = build_generator(tmp_path, monkeypatch)
    candidates = {
        "expired": [],
        "expiring": [],
        "stale": [],
        "restock": [candidate("大米")],
    }

    message = generator.generate(candidates, [candidate("大米")], [])

    assert message == "大米可能需要补货，请先确认库存再决定。"


@pytest.mark.parametrize(
    ("expires_delta", "confirmed_delta", "expected_fragment"),
    [
        (timedelta(seconds=-1), timedelta(0), "已过期"),
        (timedelta(days=3), timedelta(0), "3天内过期"),
        (timedelta(days=4), timedelta(0), None),
        (None, timedelta(days=-14, seconds=-1), "超过14天未确认"),
        (None, timedelta(days=-14), None),
    ],
)
def test_service_uses_real_candidate_boundaries_without_inventory_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    expires_delta: timedelta | None,
    confirmed_delta: timedelta,
    expected_fragment: str | None,
) -> None:
    FakeChatOpenAI.failure = RuntimeError("provider unavailable")
    service, pipeline = build_service(tmp_path, monkeypatch)
    scan_time = datetime(2026, 8, 7, 9, tzinfo=UTC)
    expires_at = None if expires_delta is None else scan_time + expires_delta
    item = pipeline.inventory.add_instance(
        "豆腐", "1盒", expires_at, "user" if expires_at else "unknown", now=scan_time
    )
    pipeline.inventory.confirm_instance(
        str(item["instance_id"]), now=scan_time + confirmed_delta
    )
    service.visits.check_and_touch("u", scan_time - timedelta(hours=24, seconds=1))
    before = pipeline.inventory.list_instances()

    message = service.check("u", now=scan_time)

    if expected_fragment is None:
        assert message is None
    else:
        assert message is not None
        assert expected_fragment in message
    assert pipeline.inventory.list_instances() == before
    assert pipeline.actions.repository.list_pending("u") == []
    assert pipeline.history.list_recent("u") == []
    assert service.check("u", now=scan_time + timedelta(seconds=1)) is None
