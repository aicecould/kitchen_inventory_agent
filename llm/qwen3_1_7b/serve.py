"""Minimal non-streaming OpenAI-style chat API, bound to loopback by default."""

import argparse
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from llm.qwen3_1_7b.runtime import MODEL_ID, QwenRuntime


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=16000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = MODEL_ID
    messages: list[Message] = Field(min_length=1, max_length=32)
    max_tokens: int = Field(default=256, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0, le=2)
    stream: Literal[False] = False


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
    try:
        content, prompt, completion, finish = app.state.runtime.generate(
            [message.model_dump() for message in request.messages],
            request.max_tokens, request.temperature,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion", "created": int(time.time()), "model": MODEL_ID,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": finish}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                  "total_tokens": prompt + completion},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port)
