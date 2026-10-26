"""Select an OpenAI-compatible model without importing local model runtimes."""

from typing import Any

import httpx

from app.config import Settings


def chat_model_kwargs(settings: Settings, *, disable_thinking: bool = False) -> dict[str, Any]:
    if settings.llm_provider == "deepseek":
        settings.require("deepseek_api_key", "deepseek_model", "deepseek_base_url")
        options = {
            "api_key": settings.deepseek_api_key,
            "base_url": settings.deepseek_base_url,
            "model": settings.deepseek_model,
            "temperature": 0,
            "timeout": settings.http_timeout_seconds,
            "max_retries": 1,
            "max_tokens": settings.deepseek_max_output_tokens,
        }
        if disable_thinking:
            options["extra_body"] = {"thinking": {"type": "disabled"}}
        return options
    settings.require("qwen_base_url", "qwen_model")
    return {
        "api_key": settings.qwen_api_key or "local",
        "base_url": settings.qwen_base_url,
        "model": settings.qwen_model,
        "temperature": 0,
        "timeout": settings.qwen_timeout_seconds,
        "max_retries": 0,
        "max_tokens": settings.qwen_max_output_tokens,
        "use_responses_api": False,
        # Local inference must not be routed through the user's proxy.
        "http_client": httpx.Client(trust_env=False),
        "http_async_client": httpx.AsyncClient(trust_env=False),
    }
