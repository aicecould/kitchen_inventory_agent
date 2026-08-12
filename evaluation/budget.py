"""Run-level external-call budgets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BudgetDecision:
    allowed: bool
    reason: str | None = None


class CallBudget:
    def __init__(self, limits: dict[str, int]) -> None:
        if any(value < 0 for value in limits.values()):
            raise ValueError("call budget limits cannot be negative")
        self._limits = dict(limits)
        self._used = {provider: 0 for provider in limits}

    def consume(self, provider: str) -> BudgetDecision:
        if provider not in self._limits:
            raise KeyError(f"unknown provider budget: {provider}")
        if self._used[provider] >= self._limits[provider]:
            return BudgetDecision(False, "budget_exhausted")
        self._used[provider] += 1
        return BudgetDecision(True)

    def used(self, provider: str) -> int:
        if provider not in self._used:
            raise KeyError(f"unknown provider budget: {provider}")
        return self._used[provider]

    def limit(self, provider: str) -> int:
        if provider not in self._limits:
            raise KeyError(f"unknown provider budget: {provider}")
        return self._limits[provider]
