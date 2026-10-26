"""DeepSeek-compatible native JSON Output plus local Pydantic validation."""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel

from app.llm_log import callback_store, logged_span


T = TypeVar("T", bound=BaseModel)


def invoke_validated_json(
    model: Any,
    schema: type[T],
    messages: str | list[dict[str, object]],
    *,
    callbacks: list[Any] | None = None,
    component: str | None = None,
    log_metadata: dict[str, Any] | None = None,
) -> T:
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    instruction = (
        "Return exactly one JSON object and no explanatory prose or markdown. "
        "Use the exact field names and value types shown in this JSON schema: "
        f"{schema_json}. JSON output is required."
    )
    if isinstance(messages, str):
        payload: list[dict[str, object]] = [
            {"role": "user", "content": f"{messages}\n\n{instruction}"}
        ]
    else:
        payload = [dict(message) for message in messages]
        if not payload:
            payload.append({"role": "user", "content": instruction})
        else:
            content = str(payload[-1].get("content", ""))
            payload[-1]["content"] = f"{content}\n\n{instruction}"
    config: dict[str, Any] = {}
    if callbacks:
        config["callbacks"] = callbacks
    if component or log_metadata:
        config["metadata"] = {}
        if component:
            config["metadata"]["log_component"] = component
        if log_metadata:
            config["metadata"]["log_metadata"] = log_metadata
    with logged_span(
        callback_store(callbacks), "component", component or schema.__name__,
        prompt=messages, parameters={"schema": schema.model_json_schema()},
        metadata=log_metadata,
    ) as detail:
        structured = model.with_structured_output(schema, method="json_mode")
        result = structured.invoke(payload, config=config or None)
        validated = result if isinstance(result, schema) else schema.model_validate(result)
        detail["content"] = validated
        return validated
