"""DeepSeek-powered LangChain tool-calling Agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from app.audit import regex_audit
from app.config import Settings
from app.context import AgentContext, AgentResult
from app.prompts import SYSTEM_PROMPT
from app.tools.inventory import InventoryRepository
from app.tools.recipe import RecipeRouter


@dataclass(slots=True)
class KitchenAgent:
    model: ChatOpenAI
    inventory: InventoryRepository
    recipes: RecipeRouter
    recursion_limit: int = 10

    def run(self, context: AgentContext, target_language: str = "zh") -> AgentResult:
        tools = self._create_tools(context, target_language)
        agent = create_agent(
            model=self.model,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
        )
        response = agent.invoke(
            {"messages": [{"role": "user", "content": self._user_message(context, target_language)}]},
            config={"recursion_limit": self.recursion_limit},
        )
        messages = response.get("messages", [])
        content = self._last_text(messages)
        history = self._tool_history(messages)
        audit = regex_audit(content)
        if not audit.passed:
            return AgentResult(
                content="输出未通过安全检查。",
                blocked=True,
                audit_reason=audit.reason,
                tool_history=history,
            )
        return AgentResult(content=content, tool_history=history)

    def _create_tools(self, context: AgentContext, target_language: str) -> list[Any]:
        repository = self.inventory
        recipe_router = self.recipes

        @tool
        def list_inventory() -> list[dict[str, object]]:
            """List every ingredient currently stored in the user's inventory."""
            return repository.list_items()

        @tool
        def get_inventory_item(name: str) -> dict[str, object] | None:
            """Get one inventory item by its exact ingredient name."""
            return repository.get_item(name)

        @tool
        def add_inventory_item(name: str, quantity: float, unit: str) -> dict[str, object]:
            """Add a positive quantity of an ingredient to inventory."""
            return repository.add_item(name, quantity, unit)

        @tool
        def update_inventory_item(name: str, quantity: float, unit: str) -> dict[str, object]:
            """Set the exact quantity and unit of an existing inventory item."""
            return repository.update_item(name, quantity, unit)

        @tool
        def remove_inventory_item(
            name: str, quantity: float | None = None
        ) -> dict[str, object]:
            """Remove a quantity of an ingredient, or remove it entirely when quantity is omitted."""
            return repository.remove_item(name, quantity)

        @tool
        def search_recipes(ingredients: list[str]) -> list[dict[str, object]]:
            """Search Spoonacular and TheMealDB for safe recipes using the given ingredients."""
            return recipe_router.search(
                ingredients=ingredients,
                preferences=self._profile_values(context, "preferences"),
                allergens=context.allergens,
                target_language=target_language,
            )

        return [
            list_inventory,
            get_inventory_item,
            add_inventory_item,
            update_inventory_item,
            remove_inventory_item,
            search_recipes,
        ]

    @staticmethod
    def _profile_values(context: AgentContext, field: str) -> list[str]:
        if field == "preferences":
            values = []
            active = False
            for line in context.profile_markdown.splitlines():
                if line.strip() == "## 饮食偏好":
                    active = True
                elif line.startswith("## "):
                    active = False
                elif active and line.strip().startswith("- "):
                    value = line.strip()[2:].strip()
                    if value not in {"", "待填写", "暂无"}:
                        values.append(value)
            return values
        return []

    @staticmethod
    def _user_message(context: AgentContext, target_language: str) -> str:
        payload = {
            "request": context.request_text,
            "intent_hint": context.intent,
            "ingredients": [item.model_dump(mode="json") for item in context.ingredients],
            "allergens": context.allergens,
            "user_profile_markdown": context.profile_markdown,
            "history_summary": context.history_summary,
            "target_language": target_language,
        }
        return (
            "请根据以下结构化上下文完成用户请求。最终回答使用 target_language 指定的语言。\n"
            + json.dumps(payload, ensure_ascii=False)
        )

    @staticmethod
    def _last_text(messages: list[Any]) -> str:
        for message in reversed(messages):
            if isinstance(message, ToolMessage):
                continue
            content = getattr(message, "content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                texts = [
                    str(block.get("text", ""))
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                if any(texts):
                    return "\n".join(texts).strip()
        raise RuntimeError("Agent returned no final text")

    @staticmethod
    def _tool_history(messages: list[Any]) -> list[dict[str, object]]:
        history: list[dict[str, object]] = []
        for message in messages:
            if isinstance(message, ToolMessage):
                history.append(
                    {
                        "tool": message.name or "unknown",
                        "tool_call_id": message.tool_call_id,
                        "result": message.content,
                    }
                )
        return history


def build_agent(
    settings: Settings,
    inventory: InventoryRepository,
    recipes: RecipeRouter,
) -> KitchenAgent:
    settings.require("deepseek_api_key", "deepseek_model", "deepseek_base_url")
    model = ChatOpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_model,
        temperature=0,
        timeout=settings.http_timeout_seconds,
        max_retries=2,
    )
    return KitchenAgent(
        model=model,
        inventory=inventory,
        recipes=recipes,
        recursion_limit=settings.agent_recursion_limit,
    )


def run_agent(
    agent: KitchenAgent,
    context: AgentContext,
    target_language: str = "zh",
) -> AgentResult:
    return agent.run(context, target_language)
