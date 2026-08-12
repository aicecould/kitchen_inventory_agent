"""Small deterministic metric helpers."""

from __future__ import annotations

from math import ceil


def confusion_counts(
    expected: list[bool], actual: list[bool]
) -> dict[str, int]:
    if len(expected) != len(actual):
        raise ValueError("expected and actual must have equal length")
    counts = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    for wanted, observed in zip(expected, actual, strict=True):
        if wanted and observed:
            counts["tp"] += 1
        elif wanted:
            counts["fn"] += 1
        elif observed:
            counts["fp"] += 1
        else:
            counts["tn"] += 1
    return counts


def percentile(values: list[float], q: float) -> float | None:
    if not 0 <= q <= 1:
        raise ValueError("q must be in [0, 1]")
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, ceil(q * len(ordered)) - 1)
    return ordered[index]


def recall_at_k(ranks: list[int | None], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not ranks:
        return 0.0
    return sum(rank is not None and rank <= k for rank in ranks) / len(ranks)


def mrr_at_k(ranks: list[int | None], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not ranks:
        return 0.0
    return sum(
        1.0 / rank for rank in ranks if rank is not None and rank <= k
    ) / len(ranks)


def relative_reduction(baseline: float, current: float) -> float | None:
    if baseline == 0:
        return None
    return (baseline - current) / baseline
