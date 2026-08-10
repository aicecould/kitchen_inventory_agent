from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier

import pytest

from app.config import get_settings
from app.context import HistoryExcerpt
from app.pipeline import KitchenPipeline, build_pipeline
from app.proactive import (
    ProactiveService,
    ReminderGenerator,
    ReminderResponse,
    VisitRepository,
)


class FakeChatOpenAI:
    response = ReminderResponse()
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

    def invoke(self, prompt: str) -> object:
        self.prompt = prompt
        if self.failure is not None:
            raise self.failure
        return self.response


@pytest.fixture(autouse=True)
def reset_fake_model() -> None:
    FakeChatOpenAI.response = ReminderResponse()
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
        "stocked_at": "2026-08-01T09:00:00+00:00",
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
        category="expiring",
        item_name="牛奶",
    )
    generator = build_generator(tmp_path, monkeypatch)

    message = generator.generate(
        {"expired": [], "expiring": [candidate("牛奶")], "stale": []},
        [candidate("牛奶")],
        [HistoryExcerpt(request="昨天买了什么？", response="买了牛奶。")],
    )

    assert message == "有食材将在3天内过期，请优先使用。"
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


def test_long_selected_item_cannot_expand_the_rendered_reminder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = "鲜" * 100
    FakeChatOpenAI.response = ReminderResponse(
        category="restock",
        item_name=name,
    )
    generator = build_generator(tmp_path, monkeypatch)

    message = generator.generate(
        {**empty_candidates(), "restock": [candidate(name)]},
        [candidate(name)],
        [],
    )

    assert message == "有食材可能需要补货，请先确认库存再决定。"
    assert len(message) <= 100


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
def test_model_supplied_free_text_is_never_rendered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_message: str,
) -> None:
    FakeChatOpenAI.response = {
        "message": invalid_message,
        "category": "expired",
        "item_name": "牛奶",
    }
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

    assert message == "有食材已过期，请检查后决定是否丢弃。"
    assert len(message) <= 100
    assert "\r" not in message and "\n" not in message


@pytest.mark.parametrize(
    ("name", "invalid_message"),
    [
        ("牛奶", "**牛奶**已过期。"),
        ("牛奶", f"牛奶待办act_{'c' * 32}待确认。"),
        ("牛奶", "牛奶已入库。"),
        ("Milk", "Milk was added to inventory."),
        ("Milk", "Milk is now in inventory."),
        ("牛奶", "牛奶<mailto:a@b>已过期。"),
    ],
)
def test_inline_markdown_adjacent_ids_and_execution_claims_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    invalid_message: str,
) -> None:
    FakeChatOpenAI.response = {
        "message": invalid_message,
        "category": "expired",
        "item_name": name,
    }
    generator = build_generator(tmp_path, monkeypatch)

    message = generator.generate(
        {"expired": [candidate(name)], "expiring": [], "stale": []},
        [candidate(name)],
        [],
    )

    assert message == "有食材已过期，请检查后决定是否丢弃。"


@pytest.mark.parametrize(
    "name",
    [
        "Milk is now in inventory",
        "Milk has entered inventory",
        "牛奶<mailto:a@b>",
        "牛奶<ftp://example.test>",
    ],
)
def test_untrusted_inventory_labels_never_reach_rendered_reminders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    FakeChatOpenAI.response = ReminderResponse(
        category="expired",
        item_name=name,
    )
    generator = build_generator(tmp_path, monkeypatch)

    message = generator.generate(
        {"expired": [candidate(name)], "expiring": [], "stale": []},
        [candidate(name)],
        [],
    )

    assert message == "有食材已过期，请检查后决定是否丢弃。"


def test_model_must_address_highest_priority_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeChatOpenAI.response = ReminderResponse(
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

    assert message == "有食材已过期，请检查后决定是否丢弃。"


def test_candidate_selection_rejects_lower_priority_substring_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeChatOpenAI.response = ReminderResponse(category="stale", item_name="玉米")
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

    assert message == "有食材已过期，请检查后决定是否丢弃。"


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

    assert message == "有食材可能需要补货，请先确认库存再决定。"


@pytest.mark.parametrize(
    ("selected_item", "expected"),
    [
        ("大米", "有食材可能需要补货，请先确认库存再决定。"),
        ("面粉", None),
    ],
)
def test_service_validates_model_restock_selection_against_real_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selected_item: str,
    expected: str | None,
) -> None:
    FakeChatOpenAI.response = ReminderResponse(
        category="restock",
        item_name=selected_item,
    )
    service, pipeline = build_service(tmp_path, monkeypatch)
    scan_time = datetime(2026, 8, 7, 9, tzinfo=UTC)
    pipeline.inventory.add_instance(
        "大米", "少量", None, "unknown", now=scan_time
    )
    pipeline.history.append(
        user_id="u",
        request="大米快吃完了",
        response="可以在下次检查库存时考虑补货。",
    )
    service.visits.check_and_touch("u", scan_time - timedelta(hours=25))
    inventory_before = pipeline.inventory.list_instances()
    history_before = pipeline.history.list_context_window("u")

    message = service.check("u", now=scan_time)

    assert message == expected
    assert pipeline.inventory.list_instances() == inventory_before
    assert pipeline.actions.repository.list_pending("u") == []
    assert pipeline.history.list_context_window("u") == history_before


def test_service_does_not_force_restock_when_provider_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeChatOpenAI.failure = RuntimeError("provider unavailable")
    service, pipeline = build_service(tmp_path, monkeypatch)
    scan_time = datetime(2026, 8, 7, 9, tzinfo=UTC)
    pipeline.inventory.add_instance(
        "大米", "少量", None, "unknown", now=scan_time
    )
    pipeline.history.append(
        user_id="u",
        request="大米快吃完了",
        response="可以在下次检查库存时考虑补货。",
    )
    service.visits.check_and_touch("u", scan_time - timedelta(hours=25))
    inventory_before = pipeline.inventory.list_instances()
    history_before = pipeline.history.list_context_window("u")

    assert pipeline.inventory.reminder_candidates(scan_time) == {
        "expired": [],
        "expiring": [],
        "stale": [],
    }
    assert service.check("u", now=scan_time) is None
    assert pipeline.inventory.list_instances() == inventory_before
    assert pipeline.actions.repository.list_pending("u") == []
    assert pipeline.history.list_context_window("u") == history_before


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
    assert pipeline.history.list_context_window("u") == []
    assert service.check("u", now=scan_time + timedelta(seconds=1)) is None
