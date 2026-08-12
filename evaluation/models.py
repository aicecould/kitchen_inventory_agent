"""Typed result contracts shared by every evaluation suite."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EvidenceKind(StrEnum):
    MEASURED = "measured"
    PAIRED_ESTIMATE = "paired_estimate"
    FAULT_INJECTION = "fault_injection"
    UNAVAILABLE = "unavailable"
    SKIPPED = "skipped"


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class SampleResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    status: str = Field(min_length=1)
    evidence: EvidenceKind
    latency_ms: list[float] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    usage: TokenUsage | None = None
    call_counts: dict[str, int] = Field(default_factory=dict)
    error: str | None = None

    @field_validator("latency_ms")
    @classmethod
    def validate_latencies(cls, values: list[float]) -> list[float]:
        if any(value < 0 for value in values):
            raise ValueError("latency values cannot be negative")
        return values

    @field_validator("call_counts")
    @classmethod
    def validate_call_counts(cls, values: dict[str, int]) -> dict[str, int]:
        if any(value < 0 for value in values.values()):
            raise ValueError("call counts cannot be negative")
        return values


class SuiteResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    evidence: EvidenceKind
    samples: list[SampleResult] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class EvaluationRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: dict[str, Any]
    suites: list[SuiteResult]
    limitations: list[str] = Field(default_factory=list)
