"""Request-local execution tracing with safe, compact summaries."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Any, TypeVar

from app.context import ExecutionTraceEvent
from app.llm_log import active_store, logged_span

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
        *,
        _persist: bool = True,
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
        if _persist:
            with logged_span(active_store(), "event", name,
                             parameters={"stage": stage}) as span:
                span["content"] = {"status": status, "detail": detail,
                                   "duration_ms": max(0, duration_ms)}

    def run(
        self,
        stage: str,
        name: str,
        operation: Callable[[], T],
        summarize: Callable[[T], str],
        *,
        inputs: Any = None,
        parameters: dict[str, Any] | None = None,
    ) -> T:
        started = perf_counter()
        try:
            with logged_span(active_store(), "component", name, prompt=inputs,
                             parameters={"stage": stage, **(parameters or {})}) as span:
                result = operation()
                span["content"] = result
        except Exception as exc:
            self.record(stage, name, "failed", type(exc).__name__,
                        _elapsed_ms(started), _persist=False)
            raise
        self.record(stage, name, "success", summarize(result),
                    _elapsed_ms(started), _persist=False)
        return result


def elapsed_ms(started: float) -> int:
    return _elapsed_ms(started)


def _elapsed_ms(started: float) -> int:
    return round((perf_counter() - started) * 1000)
