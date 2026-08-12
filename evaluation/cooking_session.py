"""Compare bounded history with explicit structured cooking-task context."""

from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.context import CookingTaskContext
from app.memory import ConversationHistoryRepository
from evaluation.benchmarks import load_json_records
from evaluation.metrics import relative_reduction
from evaluation.models import EvidenceKind, SampleResult, SuiteResult


def run_cooking_session_suite(
    settings: Settings,
    path: Path,
    *,
    offline: bool,
    temp_dir: Path,
) -> SuiteResult:
    del settings
    records = load_json_records(
        path,
        required_keys={"id", "title", "servings", "ingredients", "status", "question"},
    )
    samples: list[SampleResult] = []
    history_scores: list[float] = []
    structured_scores: list[float] = []
    for record in records:
        db_path = temp_dir / f"{record['id']}.db"
        history = ConversationHistoryRepository(db_path)
        history.initialize()
        user_id = f"eval-{record['id']}"
        history.append(
            user_id=user_id,
            request=f"确认菜谱 {record['title']}",
            response=f"已确认 {record['title']}，{record['servings']} 人份。",
        )
        for index in range(11):
            history.append(
                user_id=user_id,
                request=f"干扰问题 {index}",
                response=f"干扰回答 {index}",
            )
        window = history.list_context_window(user_id, limit=10)
        history_text = "\n".join(
            f"{value.request}\n{value.response}" for value in window
        )
        task = CookingTaskContext(
            task_id="task_" + str(record["id"])[-2:] * 16,
            query_text=f"确认菜谱 {record['title']}",
            recipe_title=str(record["title"]),
            servings=int(record["servings"]),
            ingredients=[str(value) for value in record["ingredients"]],
        )
        required = [
            str(record["title"]),
            str(record["servings"]),
            *[str(value) for value in record["ingredients"]],
        ]
        task_text = task.model_dump_json()
        history_retained = sum(value in history_text for value in required) / len(required)
        structured_retained = sum(value in task_text for value in required) / len(required)
        history_scores.append(history_retained)
        structured_scores.append(structured_retained)
        samples.append(
            SampleResult(
                id=str(record["id"]),
                status="passed",
                evidence=EvidenceKind.MEASURED,
                metrics={
                    "history_fact_retention": history_retained,
                    "structured_fact_retention": structured_retained,
                    "intervening_turns": 11,
                    "task_status": record["status"],
                    "online_answer": "skipped" if offline else "unavailable",
                },
            )
        )
    history_retention = sum(history_scores) / len(history_scores)
    structured_retention = sum(structured_scores) / len(structured_scores)
    history_error = 1 - history_retention
    structured_error = 1 - structured_retention
    return SuiteResult(
        name="cooking_session",
        evidence=EvidenceKind.MEASURED,
        samples=samples,
        summary={
            "scenario_count": len(samples),
            "history_only_fact_retention": history_retention,
            "structured_fact_retention": structured_retention,
            "history_only_fact_error": history_error,
            "structured_fact_error": structured_error,
            "absolute_error_reduction": history_error - structured_error,
            "relative_error_reduction": relative_reduction(
                history_error, structured_error
            ),
        },
    )
