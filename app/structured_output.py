"""DeepSeek-compatible native JSON Output plus local Pydantic validation."""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


def invoke_validated_json(
    model: Any,
    schema: type[T],
    messages: str | list[dict[str, object]],
    *,
    callbacks: list[Any] | None = None,
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
    config = {"callbacks": callbacks} if callbacks else None
    structured = model.with_structured_output(schema, method="json_mode")
    result = structured.invoke(payload, config=config)
    if isinstance(result, schema):
        return result
    return schema.model_validate(result)
