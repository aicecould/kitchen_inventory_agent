"""Output audit placeholders.

Only a minimal regular-expression stage is implemented. Tool-call review and
second-model review are intentionally left for later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


BLOCK_PATTERNS = (
    re.compile(r"忽略(?:之前|以上)(?:的)?指令"),
    re.compile(r"(?:已经|已)完成(?:付款|下单)"),
)


@dataclass(frozen=True, slots=True)
class AuditResult:
    passed: bool
    reason: str | None = None


def regex_audit(content: str) -> AuditResult:
    if not content.strip():
        return AuditResult(False, "Output is empty")
    for pattern in BLOCK_PATTERNS:
        if pattern.search(content):
            return AuditResult(False, f"Matched blocked pattern: {pattern.pattern}")
    return AuditResult(True)


def review_tool_calls() -> None:
    """TODO: Implement the second audit stage."""
    raise NotImplementedError


def review_with_secondary_model() -> None:
    """TODO: Implement the third audit stage."""
    raise NotImplementedError
