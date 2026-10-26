"""Non-streaming chat, native Qwen tools and JSON mode on the local GPU."""

import argparse
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from llm.qwen3_1_7b.runtime import MODEL_ID, QwenRuntime
from llm.qwen3_1_7b.protocol import assistant_message


class FunctionCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    arguments: str


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str | None = None
    parameters: dict = Field(default_factory=lambda: {"type": "object", "properties": {}})
    strict: bool | None = None


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["function"] = "function"
    function: FunctionDefinition


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict] | None = Field(default=None, max_length=60000)
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None

    @model_validator(mode="after")
    def validate_tool_message(self):
        if isinstance(self.content, list):
            if all(block.get("type") == "text" and isinstance(block.get("text"), str)
                   for block in self.content):
                self.content = "\n".join(block["text"] for block in self.content)
            elif self.role == "tool":
                # LangChain preserves list-of-dict tool outputs as message content.
                self.content = json.dumps(self.content, ensure_ascii=False)
            else:
                raise ValueError("Only text content blocks are supported")
            if len(self.content) > 60000:
                raise ValueError("Message content exceeds 60000 characters")
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("Tool results require tool_call_id")
        if self.tool_calls and self.role != "assistant":
            raise ValueError("Only assistant messages may contain tool_calls")
        for call in self.tool_calls or []:
            if not isinstance(json.loads(call.function.arguments), dict):
                raise ValueError("Tool arguments must encode a JSON object")
        return self


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = MODEL_ID
    messages: list[Message] = Field(min_length=1, max_length=128)
    max_tokens: int = Field(default=256, ge=1, le=4096,
                            validation_alias=AliasChoices("max_tokens", "max_completion_tokens"))
    temperature: float = Field(default=0.7, ge=0, le=2)
    stream: Literal[False] = False
    tools: list[ToolDefinition] | None = Field(default=None, max_length=64)
    tool_choice: Literal["auto", "none", "required"] | dict | None = None
    parallel_tool_calls: bool = True
    response_format: dict | None = None

    @model_validator(mode="after")
    def validate_options(self):
        if self.response_format not in (None, {"type": "text"}, {"type": "json_object"}):
            raise ValueError("Only text and json_object response formats are supported")
        names = {tool.function.name for tool in self.tools or []}
        if self.tool_choice == "required" and not names:
            raise ValueError("Required tool choice needs tools")
        if isinstance(self.tool_choice, dict):
            function = self.tool_choice.get("function")
            name = function.get("name") if isinstance(function, dict) else None
            if (self.tool_choice.get("type") != "function"
                    or not isinstance(name, str) or name not in names):
                raise ValueError("Unknown tool choice")
        return self


@asynccontextmanager
async def lifespan(app):
    app.state.runtime = QwenRuntime()
    yield


app = FastAPI(title="Local Qwen3-1.7B GPU inference", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": str(app.state.runtime.device)}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [
        {"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "local"}
    ]}


@app.post("/v1/chat/completions")
def chat(request: ChatRequest):
    if request.model != MODEL_ID:
        raise HTTPException(404, "Model not found")
    messages = [message.model_dump(exclude_none=True) for message in request.messages]
    tools = [tool.model_dump(exclude_none=True) for tool in request.tools or []]
    required_name = None
    instructions = []
    if request.tool_choice == "none":
        tools = []
    elif isinstance(request.tool_choice, dict):
        required_name = request.tool_choice["function"]["name"]
        tools = [tool for tool in tools if tool["function"]["name"] == required_name]
        instructions.append(f"You must call the function {required_name}.")
    elif request.tool_choice == "required":
        instructions.append("You must call one of the provided functions.")
    if not request.parallel_tool_calls:
        instructions.append("Call at most one function in this response.")
    json_mode = request.response_format == {"type": "json_object"}
    if json_mode:
        instructions.append("Return exactly one valid JSON object without markdown or explanations.")
    if instructions:
        instruction = "\n".join(instructions)
        if messages[0]["role"] == "system":
            messages[0]["content"] = (messages[0].get("content") or "") + "\n" + instruction
        else:
            messages.insert(0, {"role": "system", "content": instruction})
    try:
        content, prompt, completion, finish = app.state.runtime.generate(
            messages, request.max_tokens, request.temperature, tools=tools or None,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    try:
        message = assistant_message(content, tools)
        calls = message.get("tool_calls", [])
        if (request.tool_choice == "required" or required_name) and not calls:
            raise ValueError("Model did not produce the required tool call")
        if not request.parallel_tool_calls and len(calls) > 1:
            raise ValueError("Model returned parallel tool calls when disabled")
        if json_mode and not calls and not isinstance(json.loads(message["content"] or ""), dict):
            raise ValueError("Model did not produce a JSON object")
        if calls:
            finish = "tool_calls"
    except ValueError as exc:
        raise HTTPException(502, str(exc)) from exc
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion", "created": int(time.time()), "model": MODEL_ID,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": finish}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                  "total_tokens": prompt + completion},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port)
