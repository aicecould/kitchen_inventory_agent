"""Translate Qwen's native tool-call tags into OpenAI message objects."""

import json
import re
from uuid import uuid4

_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def assistant_message(text: str, tools: list[dict] | None = None) -> dict:
    names = {tool["function"]["name"] for tool in tools or []}
    calls = []
    for match in _TOOL_CALL.finditer(text):
        call = json.loads(match.group(1))
        if (not isinstance(call, dict) or not isinstance(call.get("name"), str)
                or call["name"] not in names):
            raise ValueError("Model returned an unknown tool name")
        arguments = call.get("arguments")
        if not isinstance(arguments, dict):
            raise ValueError("Model tool arguments must be a JSON object")
        calls.append({
            "id": "call_" + uuid4().hex,
            "type": "function",
            "function": {"name": call["name"],
                         "arguments": json.dumps(arguments, ensure_ascii=False)},
        })
    content = _TOOL_CALL.sub("", text).strip()
    if "<tool_call>" in content or "</tool_call>" in content:
        raise ValueError("Model returned an incomplete tool call")
    message = {"role": "assistant", "content": content or None}
    if calls:
        message["tool_calls"] = calls
    return message
