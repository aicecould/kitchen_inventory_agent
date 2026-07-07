"""Request-local execution tracing with safe, compact summaries."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import TypeVar

from app.context import ExecutionTraceEvent

T = TypeVar("T")


class TraceRecorder:
    def __init__(self) -> None:
        self.events: list[ExecutionTraceEvent] = []

    def record(
        self,
        stage: str,
        name: str,
        status: str,
        detail: str,
        duration_ms: int = 0,
    ) -> None:
        self.events.append(
            ExecutionTraceEvent(
                stage=stage,
                name=name,
                status=status,
                detail=detail,
                duration_ms=max(0, duration_ms),
            )
        )

    def run(
        self,
        stage: str,
        name: str,
        operation: Callable[[], T],
        summarize: Callable[[T], str],
    ) -> T:
        started = perf_counter()
        try:
            result = operation()
        except Exception as exc:
            self.record(
                stage,
                name,
                "failed",
                f"{type(exc).__name__}",
                _elapsed_ms(started),
            )
            raise
        self.record(
            stage,
            name,
            "success",
            summarize(result),
            _elapsed_ms(started),
        )
        return result


def elapsed_ms(started: float) -> int:
    return _elapsed_ms(started)


def _elapsed_ms(started: float) -> int:
    return round((perf_counter() - started) * 1000)
