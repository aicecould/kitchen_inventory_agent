from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest

from app.memory import ConversationHistoryRepository
from app.tasks import (
    CompletionReason,
    CookingTaskRepository,
    CookingTaskService,
    RecipeSnapshot,
    ServingsSource,
    TaskStatus,
)


def _recipe_result(recipe_id: str = "recipe-1") -> dict[str, object]:
    return {
        "id": recipe_id,
        "title": "Tomato Soup",
        "instructions": "Simmer the tomatoes.",
        "ingredients": ["2 tomatoes", "1 cup stock"],
        "source": "test",
        "source_url": "https://recipes.example/tomato-soup",
        "image_url": "https://recipes.example/tomato-soup.jpg",
    }


def _snapshot(recipe_id: str = "recipe-1") -> RecipeSnapshot:
    return RecipeSnapshot(
        **_recipe_result(recipe_id),
        history_segment_id="hist_0123456789abcdef0123456789abcdef",
        query_text="用番茄找菜谱",
    )


@dataclass
class TaskStack:
    repository: CookingTaskRepository
    history: ConversationHistoryRepository
    service: CookingTaskService

    def add_recipe_evidence(self, user_id: str = "u") -> str:
        record = self.history.append(
            user_id=user_id,
            request="用番茄找菜谱",
            response="找到番茄汤",
            tool_history=[
                {
                    "tool": "search_recipes",
                    "arguments": {"ingredients": ["tomato"]},
                    "result": [_recipe_result()],
                }
            ],
        )
        return record.segment_id

    def confirm(self, **overrides: object):
        values: dict[str, object] = {
            "user_id": "u",
            "segment_id": self.add_recipe_evidence(),
            "recipe_id": "recipe-1",
            "servings": None,
            "default_context": [],
            "as_new": False,
            "expected_version": None,
            "request_id": "confirm-1",
        }
        values.update(overrides)
        return self.service.confirm_recipe(**values)


@pytest.fixture
def stack(tmp_path: Path) -> TaskStack:
    database = tmp_path / "inventory.db"
    history = ConversationHistoryRepository(database)
    history.initialize()
    repository = CookingTaskRepository(database)
    repository.initialize()
    return TaskStack(repository, history, CookingTaskService(repository, history))


def test_repository_enforces_one_active_task_and_survives_restart(
    tmp_path: Path,
) -> None:
    database = tmp_path / "inventory.db"
    repository = CookingTaskRepository(database)
    repository.initialize()
    first = repository.create_active(
        user_id="u",
        query_text="番茄能做什么",
        recipe=_snapshot(),
        servings=1,
        servings_source=ServingsSource.FALLBACK,
        request_id="req-1",
    )

    with pytest.raises(sqlite3.IntegrityError):
        repository.create_active(
            user_id="u",
            query_text="土豆能做什么",
            recipe=_snapshot("recipe-2"),
            servings=1,
            servings_source=ServingsSource.FALLBACK,
            request_id="req-2",
        )

    restarted = CookingTaskRepository(database)
    restarted.initialize()
    assert restarted.get_active("u") == first


def test_confirm_recipe_uses_persisted_result_and_profile_servings(
    stack: TaskStack,
) -> None:
    segment_id = stack.add_recipe_evidence()
    task = stack.service.confirm_recipe(
        user_id="u",
        segment_id=segment_id,
        recipe_id="recipe-1",
        servings=None,
        default_context=["默认两人份"],
        as_new=False,
        expected_version=None,
        request_id="confirm-1",
    )

    assert task.recipe.title == "Tomato Soup"
    assert task.recipe.history_segment_id == segment_id
    assert task.recipe.query_text == "用番茄找菜谱"
    assert task.servings == 2
    assert task.servings_source == ServingsSource.PROFILE
    assert task.status == TaskStatus.ACTIVE


@pytest.mark.parametrize(
    ("explicit", "profile", "expected", "source"),
    [
        (3, ["默认两人份"], 3, ServingsSource.USER),
        (None, ["servings: 10"], 10, ServingsSource.PROFILE),
        (None, ["大概一家人"], 1, ServingsSource.FALLBACK),
        (None, [], 1, ServingsSource.FALLBACK),
    ],
)
def test_serving_resolution(
    stack: TaskStack,
    explicit: int | None,
    profile: list[str],
    expected: int,
    source: ServingsSource,
) -> None:
    task = stack.confirm(servings=explicit, default_context=profile)
    assert (task.servings, task.servings_source) == (expected, source)


def test_recipe_confirmation_rejects_cross_user_and_unknown_recipe(
    stack: TaskStack,
) -> None:
    segment_id = stack.add_recipe_evidence(user_id="other")
    with pytest.raises(KeyError, match="recipe evidence"):
        stack.service.confirm_recipe(
            user_id="u",
            segment_id=segment_id,
            recipe_id="recipe-1",
            servings=None,
            default_context=[],
            as_new=False,
            expected_version=None,
            request_id="confirm-1",
        )

    own_segment = stack.add_recipe_evidence()
    with pytest.raises(KeyError, match="recipe"):
        stack.service.confirm_recipe(
            user_id="u",
            segment_id=own_segment,
            recipe_id="missing",
            servings=None,
            default_context=[],
            as_new=False,
            expected_version=None,
            request_id="confirm-2",
        )


def test_confirm_updates_active_or_atomically_creates_new_task(
    stack: TaskStack,
) -> None:
    first = stack.confirm()
    updated = stack.confirm(
        expected_version=first.version,
        servings=2,
        request_id="confirm-2",
    )
    assert updated.task_id == first.task_id
    assert updated.version == first.version + 1
    assert updated.servings == 2

    second = stack.confirm(
        as_new=True,
        expected_version=updated.version,
        request_id="confirm-3",
    )
    assert second.task_id != first.task_id
    assert stack.repository.get(first.task_id).status == TaskStatus.PAUSED  # type: ignore[union-attr]
    assert stack.repository.get_active("u") == second


def test_switch_is_versioned_and_cooked_retry_is_idempotent(stack: TaskStack) -> None:
    first = stack.confirm()
    second = stack.confirm(
        as_new=True,
        expected_version=first.version,
        request_id="confirm-2",
    )
    switched = stack.service.activate(
        user_id="u",
        task_id=first.task_id,
        expected_version=stack.repository.get(first.task_id).version,  # type: ignore[union-attr]
        request_id="activate-1",
    )
    assert switched.status == TaskStatus.ACTIVE
    assert stack.repository.get(second.task_id).status == TaskStatus.PAUSED  # type: ignore[union-attr]

    with pytest.raises(ValueError, match="version"):
        stack.service.update_servings(
            user_id="u",
            task_id=first.task_id,
            servings=4,
            expected_version=first.version,
            request_id="servings-stale",
        )

    completed, changed = stack.service.complete_cooked(
        user_id="u",
        task_id=first.task_id,
        expected_version=switched.version,
        request_id="cooked-1",
    )
    repeated, repeated_changed = stack.service.complete_cooked(
        user_id="u",
        task_id=first.task_id,
        expected_version=switched.version,
        request_id="cooked-1",
    )
    assert changed is True
    assert repeated_changed is False
    assert repeated == completed
    assert completed.completion_reason == CompletionReason.COOKED
    assert completed.reconciliation_open is True

    open_task = stack.service.get_reconciliation_task("u", first.task_id)
    assert open_task == completed
    closed = stack.service.close_reconciliation(
        user_id="u",
        task_id=first.task_id,
        expected_version=completed.version,
        request_id="close-1",
    )
    assert closed.reconciliation_open is False


def test_clear_context_atomically_resets_window_and_completes_active_task(
    stack: TaskStack,
) -> None:
    task = stack.confirm()
    stack.history.append(user_id="u", request="before", response="before")

    cleared = stack.service.clear_context("u", request_id="clear-1")

    assert cleared is not None
    assert cleared.task_id == task.task_id
    assert cleared.status == TaskStatus.COMPLETED
    assert cleared.completion_reason == CompletionReason.CONTEXT_CLEARED
    assert cleared.reconciliation_open is False
    assert stack.repository.get_active("u") is None
    assert stack.history.list_context_window("u") == []
    assert len(stack.history.list_records("u")) == 2


def test_clear_context_without_active_task_is_idempotent(stack: TaskStack) -> None:
    stack.history.append(user_id="u", request="before", response="before")

    assert stack.service.clear_context("u", request_id="clear-1") is None
    assert stack.service.clear_context("u", request_id="clear-1") is None
    assert stack.history.list_context_window("u") == []
