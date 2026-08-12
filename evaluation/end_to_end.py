"""Small isolated mixed-query pipeline-path benchmark."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from app.config import Settings
from app.intent import route_deterministically
from app.memory import ConversationHistoryRepository
from app.tools.inventory import InventoryRepository
from evaluation.benchmarks import load_json_records
from evaluation.metrics import percentile
from evaluation.models import EvidenceKind, SampleResult, SuiteResult


def run_end_to_end_suite(
    settings: Settings,
    path: Path,
    *,
    offline: bool,
    temp_dir: Path,
) -> SuiteResult:
    records = load_json_records(
        path, required_keys={"id", "text", "expected_route"}
    )
    db_path = temp_dir / "evaluation-state.db"
    inventory = InventoryRepository(db_path)
    inventory.initialize()
    history = ConversationHistoryRepository(db_path)
    history.initialize()
    samples: list[SampleResult] = []
    for record in records:
        started = perf_counter()
        route = route_deterministically(str(record["text"]))
        latency = (perf_counter() - started) * 1000
        history.append(
            user_id="evaluation-user",
            request=str(record["text"]),
            response=f"offline route: {route.kind}",
            is_relevant=route.kind != "irrelevant",
            blocked=route.kind == "irrelevant",
        )
        samples.append(
            SampleResult(
                id=str(record["id"]),
                status=(
                    "passed" if route.kind == record["expected_route"] else "wrong_route"
                ),
                evidence=EvidenceKind.MEASURED,
                latency_ms=[latency],
                metrics={
                    "text": record["text"],
                    "expected_route": record["expected_route"],
                    "actual_route": route.kind,
                    "online_agent": "skipped" if offline else "unavailable",
                },
                call_counts={"network": 0, "tools": 0},
            )
        )
    latencies = [value for sample in samples for value in sample.latency_ms]
    return SuiteResult(
        name="end_to_end_paths",
        evidence=EvidenceKind.MEASURED,
        samples=samples,
        summary={
            "sample_count": len(samples),
            "path_success_rate": sum(s.status == "passed" for s in samples) / len(samples),
            "blocked_count": sum(s.metrics["actual_route"] == "irrelevant" for s in samples),
            "latency_p50_ms": percentile(latencies, 0.5),
            "latency_p95_ms": percentile(latencies, 0.95),
            "latency_max_ms": max(latencies),
            "network_calls": 0,
            "temporary_db": str(db_path),
        },
    )
