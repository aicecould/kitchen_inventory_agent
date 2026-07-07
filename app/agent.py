"""DeepSeek-powered LangChain tool-calling Agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from app.actions import InventoryActionService
from app.allergens import AllergenIntolerance
from app.audit import regex_audit
from app.config import Settings
from app.context import AgentContext, AgentResult
from app.prompts import SYSTEM_PROMPT
from app.tools.inventory import InventoryRepository
from app.tools.recipe import RecipeRouter
from app.trace import TraceRecorder, elapsed_ms


@dataclass(slots=True)
class KitchenAgent:
    model: ChatOpenAI
    inventory: InventoryRepository
    recipes: RecipeRouter
    actions: InventoryActionService
    recursion_limit: int = 10
    max_input_chars: int = 12000

    def run(self, context: AgentContext, target_language: str = "zh") -> AgentResult:
        trace = TraceRecorder()
        user_message = self._user_message(context, target_language)
        if len(user_message) > self.max_input_chars:
            trace.record("agent", "input_limit", "blocked", "input too long")
            raise ValueError(
                f"Agent input is too long: {len(user_message)} characters; "
                f"limit is {self.max_input_chars}"
            )
        trace.record("agent", "input_limit", "passed", f"{len(user_message)} character(s)")
        tools = self._create_tools(context, trace)
        agent = create_agent(
            model=self.model,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
        )
        started = perf_counter()
        try:
            response = agent.invoke(
                {"messages": [{"role": "user", "content": user_message}]},
                config={"recursion_limit": self.recursion_limit},
            )
        except Exception as exc:
            trace.record(
                "model",
                "deepseek_agent",
                "failed",
                type(exc).__name__,
                elapsed_ms(started),
            )
            raise
        trace.record(
            "model",
            "deepseek_agent",
            "success",
            "response completed",
            elapsed_ms(started),
        )
        messages = response.get("messages", [])
        content = self._last_text(messages)
        history = self._tool_history(messages)
        audit = regex_audit(content)
        trace.record(
            "output",
            "regex_audit",
            "passed" if audit.passed else "blocked",
            audit.reason or "passed",
        )
        if not audit.passed:
            return AgentResult(
                content="输出未通过安全检查。",
                blocked=True,
                audit_reason=audit.reason,
                tool_history=history,
                execution_trace=trace.events,
            )
        return AgentResult(
            content=content,
            tool_history=history,
            execution_trace=trace.events,
        )

    def _create_tools(
        self, context: AgentContext, trace: TraceRecorder
    ) -> list[Any]:
        repository = self.inventory
        recipe_router = self.recipes
        action_service = self.actions

        @tool
        def list_inventory() -> list[dict[str, object]]:
            """List every ingredient currently stored in the user's inventory."""
            return trace.run(
                "database",
                "list_inventory",
                repository.list_items,
                lambda values: f"{len(values)} item(s)",
            )

        @tool
        def get_inventory_item(name: str) -> dict[str, object] | None:
            """Get one inventory item by its exact ingredient name."""
            return trace.run(
                "database",
                "get_inventory_item",
                lambda: repository.get_item(name),
                lambda value: "found" if value else "not found",
            )

        @tool
        def propose_add_inventory_item(
            name: str, quantity: float, unit: str
        ) -> dict[str, object]:
            """Propose adding an ingredient. This never writes until the user confirms."""
            return trace.run(
                "database",
                "propose_add_inventory_item",
                lambda: action_service.propose(
                    context.user_id,
                    "inventory.add",
                    {"name": name, "quantity": quantity, "unit": unit},
                ).model_dump(mode="json"),
                lambda value: f"status={value['status']}",
            )

        @tool
        def propose_update_inventory_item(
            name: str, quantity: float, unit: str
        ) -> dict[str, object]:
            """Propose setting inventory quantity. This never writes until confirmation."""
            return trace.run(
                "database",
                "propose_update_inventory_item",
                lambda: action_service.propose(
                    context.user_id,
                    "inventory.update",
                    {"name": name, "quantity": quantity, "unit": unit},
                ).model_dump(mode="json"),
                lambda value: f"status={value['status']}",
            )

        @tool
        def propose_remove_inventory_item(
            name: str, quantity: float | None = None
        ) -> dict[str, object]:
            """Propose reducing or deleting inventory. Confirmation is always required."""
            return trace.run(
                "database",
                "propose_remove_inventory_item",
                lambda: action_service.propose(
                    context.user_id,
                    "inventory.remove",
                    {"name": name, "quantity": quantity},
                ).model_dump(mode="json"),
                lambda value: f"status={value['status']}",
            )

        @tool
        def search_recipes(
            ingredients: list[str],
            intolerances: list[AllergenIntolerance],
            exclude_ingredients: list[str],
            diet: str | None = None,
            cuisine: str | None = None,
            meal_type: str | None = None,
            max_ready_time: int | None = None,
            ranking: Literal[
                "max-used-ingredients", "min-missing-ingredients"
            ] = "max-used-ingredients",
        ) -> list[dict[str, object]]:
            """Search safe recipes. Use official intolerance categories plus concrete excluded ingredients."""
            if not ingredients:
                raise ValueError("At least one ingredient is required")
            if context.allergens and not intolerances and not exclude_ingredients:
                raise ValueError(
                    "User allergens exist; intolerances or exclude_ingredients must be provided"
                )
            missing_broad = set(context.allergen_intolerances) - set(intolerances)
            missing_custom = {
                value.casefold() for value in context.custom_allergens
            } - {value.casefold() for value in exclude_ingredients}
            if missing_broad or missing_custom:
                raise ValueError(
                    "Recipe search must include every configured allergen category "
                    "and custom excluded ingredient"
                )
            if max_ready_time is not None and max_ready_time <= 0:
                raise ValueError("max_ready_time must be positive")
            return recipe_router.search(
                ingredients=ingredients,
                intolerances=intolerances,
                exclude_ingredients=exclude_ingredients,
                diet=diet,
                cuisine=cuisine,
                meal_type=meal_type,
                max_ready_time=max_ready_time,
                ranking=ranking,
                trace=trace.record,
            )

        return [
            list_inventory,
            get_inventory_item,
            propose_add_inventory_item,
            propose_update_inventory_item,
            propose_remove_inventory_item,
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
            "allergen_intolerances": context.allergen_intolerances,
            "custom_allergens": context.custom_allergens,
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
    actions: InventoryActionService,
) -> KitchenAgent:
    settings.require("deepseek_api_key", "deepseek_model", "deepseek_base_url")
    model = ChatOpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_model,
        temperature=0,
        timeout=settings.http_timeout_seconds,
        max_retries=1,
        max_tokens=settings.deepseek_max_output_tokens,
    )
    return KitchenAgent(
        model=model,
        inventory=inventory,
        recipes=recipes,
        actions=actions,
        recursion_limit=settings.agent_recursion_limit,
        max_input_chars=settings.deepseek_max_input_chars,
    )


def run_agent(
    agent: KitchenAgent,
    context: AgentContext,
    target_language: str = "zh",
) -> AgentResult:
    return agent.run(context, target_language)
