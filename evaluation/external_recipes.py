"""Counting and budget enforcement for external recipe clients."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from evaluation.budget import CallBudget


class BudgetExhausted(RuntimeError):
    pass


class CountingRecipeClient:
    def __init__(self, provider: str, client: Any, budget: CallBudget) -> None:
        self.provider = provider
        self.client = client
        self.budget = budget
        self.attempted = 0
        self.successful = 0
        self.empty = 0
        self.failed = 0
        self.budget_exhausted = 0
        self.latency_ms: list[float] = []

    def search(self, **kwargs: Any) -> Any:
        decision = self.budget.consume(self.provider)
        if not decision.allowed:
            self.budget_exhausted += 1
            raise BudgetExhausted(f"{self.provider} budget exhausted")
        self.attempted += 1
        started = perf_counter()
        try:
            result = self.client.search(**kwargs)
        except Exception:
            self.failed += 1
            raise
        finally:
            self.latency_ms.append((perf_counter() - started) * 1000)
        if result:
            self.successful += 1
        else:
            self.empty += 1
        return result
