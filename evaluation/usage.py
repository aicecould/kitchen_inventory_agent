"""Aggregate real provider token metadata without estimating from content."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from evaluation.models import TokenUsage


class UsageCollector(BaseCallbackHandler):
    def __init__(self) -> None:
        self.call_count = 0
        self.latency_ms: list[float] = []
        self._started: dict[str, float] = {}
        self._input = 0
        self._output = 0
        self._total = 0
        self._observed = False

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], *, run_id: Any, **kwargs: Any
    ) -> None:
        self._started[str(run_id)] = perf_counter()

    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: list[list[Any]], *, run_id: Any,
        **kwargs: Any,
    ) -> None:
        self._started[str(run_id)] = perf_counter()

    def on_llm_end(self, response: Any, *, run_id: Any = None, **kwargs: Any) -> None:
        self.call_count += 1
        started = self._started.pop(str(run_id), None)
        if started is not None:
            self.latency_ms.append((perf_counter() - started) * 1000)
        usage = _extract_usage(response)
        if usage is None:
            return
        self._observed = True
        self._input += usage[0]
        self._output += usage[1]
        self._total += usage[2]

    def snapshot(self) -> TokenUsage | None:
        if not self._observed:
            return None
        return TokenUsage(
            input_tokens=self._input,
            output_tokens=self._output,
            total_tokens=self._total,
        )


def _extract_usage(response: Any) -> tuple[int, int, int] | None:
    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, dict):
        value = llm_output.get("token_usage") or llm_output.get("usage")
        parsed = _parse_usage(value)
        if parsed is not None:
            return parsed
    for generation_group in getattr(response, "generations", []) or []:
        for generation in generation_group if isinstance(generation_group, list) else []:
            message = getattr(generation, "message", None)
            if message is None:
                continue
            parsed = _parse_usage(getattr(message, "usage_metadata", None))
            if parsed is not None:
                return parsed
            metadata = getattr(message, "response_metadata", None)
            if isinstance(metadata, dict):
                parsed = _parse_usage(
                    metadata.get("token_usage") or metadata.get("usage")
                )
                if parsed is not None:
                    return parsed
    return None


def _parse_usage(value: object) -> tuple[int, int, int] | None:
    if not isinstance(value, dict):
        return None
    input_value = value.get("input_tokens", value.get("prompt_tokens"))
    output_value = value.get("output_tokens", value.get("completion_tokens"))
    total_value = value.get("total_tokens")
    if not isinstance(input_value, int) or not isinstance(output_value, int):
        return None
    total = total_value if isinstance(total_value, int) else input_value + output_value
    if input_value < 0 or output_value < 0 or total < 0:
        return None
    return input_value, output_value, total
