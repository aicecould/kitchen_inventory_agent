from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.actions import InventoryActionService, PendingActionRepository
from app.adapters.recipe_api import TheMealDbClient
from app.agent import KitchenAgent
from app.context import AgentContext, CookingTaskContext, HistoryExcerpt
from app.profile import ProfileRepository
from app.profile_actions import ProfileActionService
from app.prompts import SYSTEM_PROMPT
from app.tools.inventory import InventoryRepository
from app.tools.recipe import RecipeRouter
from app.trace import TraceRecorder


class FakeRecipeRouter:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    def search(self, **kwargs: object) -> list[dict[str, object]]:
        self.kwargs = kwargs
        return []


class FixedExpiryInferer:
    def infer(self, name: str, now: datetime) -> datetime:
        assert name == "牛奶"
        return datetime(2099, 1, 2, tzinfo=UTC)


def recipe_tool(
    context: AgentContext, recipes: FakeRecipeRouter
) -> object:
    agent = KitchenAgent(
        model=None,  # type: ignore[arg-type]
        inventory=None,  # type: ignore[arg-type]
        recipes=recipes,  # type: ignore[arg-type]
        actions=None,  # type: ignore[arg-type]
    )
    return next(
        tool
        for tool in agent._create_tools(context, TraceRecorder())
        if tool.name == "search_recipes"
    )


def inventory_tools(
    tmp_path: Path, context: AgentContext | None = None
) -> tuple[
    InventoryRepository,
    InventoryActionService,
    dict[str, object],
]:
    inventory = InventoryRepository(tmp_path / "inventory.db")
    inventory.initialize()
    repository = PendingActionRepository(tmp_path / "inventory.db")
    repository.initialize()
    actions = InventoryActionService(
        repository,
        inventory,
        expiry_inferer=FixedExpiryInferer(),
    )
    agent = KitchenAgent(
        model=None,  # type: ignore[arg-type]
        inventory=inventory,
        recipes=RecipeRouter(
            None,
            TheMealDbClient("https://example.invalid", timeout=1),
        ),
        actions=actions,
    )
    context = context or AgentContext(user_id="test", request_text="管理库存")
    created_tools = agent._create_tools(context, TraceRecorder())
    tools = {tool.name: tool for tool in created_tools}
    return inventory, actions, tools


def test_profile_tool_binds_trusted_user_and_current_segment(tmp_path: Path) -> None:
    inventory, actions, _ = inventory_tools(tmp_path)
    profile_actions = ProfileActionService(
        ProfileRepository(tmp_path / "profile.md"),
        evidence_exists=lambda _user_id, _segment_id: False,
        now=lambda: datetime(2026, 8, 10, tzinfo=UTC),
    )
    agent = KitchenAgent(
        model=None,  # type: ignore[arg-type]
        inventory=inventory,
        recipes=RecipeRouter(
            None,
            TheMealDbClient("https://example.invalid", timeout=1),
        ),
        actions=actions,
        profile_actions=profile_actions,
    )
    segment_id = "hist_" + "1" * 32
    context = AgentContext(
        user_id="test",
        segment_id=segment_id,
        request_text="以后都不吃香菜",
    )
    tools = {
        tool.name: tool
        for tool in agent._create_tools(context, TraceRecorder())
    }

    result = tools["propose_profile_change"].invoke(  # type: ignore[attr-defined]
        {
            "category": "preference",
            "key": "ingredient.cilantro",
            "operation": "upsert",
            "value": "不喜欢香菜",
            "reason": "用户明确表达永久偏好",
        }
    )

    assert result["user_id"] == "test"
    assert result["evidence_ids"] == [segment_id]
    assert result["status"] == "pending"


def test_agent_prompt_requires_instance_lookup_and_ambiguity_clarification() -> None:
    assert "instance_id" in SYSTEM_PROMPT
    assert "先调用" in SYSTEM_PROMPT and "查询" in SYSTEM_PROMPT
    assert "多个" in SYSTEM_PROMPT and "澄清" in SYSTEM_PROMPT
    assert "完整的 quantity_text" in SYSTEM_PROMPT
    assert "完整 instance_id" in SYSTEM_PROMPT
    assert "短前缀" in SYSTEM_PROMPT and "不能" in SYSTEM_PROMPT


def test_user_message_injects_task_separately_from_global_history() -> None:
    context = AgentContext(
        user_id="u",
        request_text="继续",
        history_segments=[HistoryExcerpt(request="global", response="history")],
        cooking_task=CookingTaskContext(
            task_id="task_0123456789abcdef0123456789abcdef",
            query_text="做汤",
            recipe_title="Tomato Soup",
            servings=2,
            ingredients=["2 tomatoes"],
        ),
    )

    message = KitchenAgent._user_message(
        context,
        "zh",
        current_time=datetime(2026, 8, 10, tzinfo=UTC),
    )
    import json

    payload = json.loads(message.split("\n", 1)[1])
    assert payload["recent_history"] == [
        {"request": "global", "response": "history"}
    ]
    assert payload["cooking_task"] == {
        "task_id": "task_0123456789abcdef0123456789abcdef",
        "query_text": "做汤",
        "recipe_title": "Tomato Soup",
        "servings": 2,
        "ingredients": ["2 tomatoes"],
    }


def test_inventory_write_tools_bind_trusted_reconciliation_task(
    tmp_path: Path,
) -> None:
    task_id = "task_0123456789abcdef0123456789abcdef"
    context = AgentContext(
        user_id="test",
        request_text="番茄用完了",
        inventory_source_task_id=task_id,
    )
    inventory, actions, tools = inventory_tools(tmp_path, context)
    item = inventory.add_instance("番茄", "2个", None, "unknown")
    tools["get_inventory_instance"].invoke(  # type: ignore[attr-defined]
        {"instance_id": item["instance_id"]}
    )

    result = tools["propose_remove_inventory_instance"].invoke(  # type: ignore[attr-defined]
        {"instance_id": item["instance_id"]}
    )

    assert result["source_task_id"] == task_id
    assert actions.repository.get(result["action_id"]).source_task_id == task_id  # type: ignore[union-attr]


def test_inventory_read_tools_return_real_instance_snapshots(tmp_path: Path) -> None:
    inventory, _, tools = inventory_tools(tmp_path)
    first = inventory.add_instance("番茄", "2个", None, "unknown")
    second = inventory.add_instance("番茄", "500g", None, "unknown")

    listed = tools["list_inventory_instances"].invoke({})  # type: ignore[attr-defined]
    found = tools["find_inventory_instances"].invoke(  # type: ignore[attr-defined]
        {"name": "番茄"}
    )
    fetched = tools["get_inventory_instance"].invoke(  # type: ignore[attr-defined]
        {"instance_id": first["instance_id"]}
    )

    assert listed == [first, second]
    assert found == [first, second]
    assert fetched == first


def test_inventory_write_tools_require_lookup_and_create_only_pending_actions(
    tmp_path: Path,
) -> None:
    inventory, actions, tools = inventory_tools(tmp_path)
    first = inventory.add_instance("番茄", "2个", None, "unknown")
    second = inventory.add_instance("番茄", "500g", None, "unknown")
    unique = inventory.add_instance("土豆", "1袋", None, "unknown")
    first_id = str(first["instance_id"])
    unique_id = str(unique["instance_id"])

    with pytest.raises(ValueError, match="INSTANCE_LOOKUP_REQUIRED"):
        tools["propose_remove_inventory_instance"].invoke(  # type: ignore[attr-defined]
            {"instance_id": first_id}
        )

    tools["find_inventory_instances"].invoke({"name": "番茄"})  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="AMBIGUOUS_INSTANCE_MATCH"):
        tools["propose_remove_inventory_instance"].invoke(  # type: ignore[attr-defined]
            {"instance_id": first_id}
        )

    tools["get_inventory_instance"].invoke(  # type: ignore[attr-defined]
        {"instance_id": first_id}
    )
    with pytest.raises(ValueError, match="AMBIGUOUS_INSTANCE_MATCH"):
        tools["propose_remove_inventory_instance"].invoke(  # type: ignore[attr-defined]
            {"instance_id": first_id}
        )

    tools["find_inventory_instances"].invoke({"name": "土豆"})  # type: ignore[attr-defined]
    add = tools["propose_add_inventory_instance"].invoke(  # type: ignore[attr-defined]
        {"name": "牛奶", "quantity_text": "大约一盒"}
    )
    replace = tools["propose_replace_inventory_instance"].invoke(  # type: ignore[attr-defined]
        {
            "instance_id": unique_id,
            "quantity_text": "1个",
            "expires_at": None,
            "expiry_source": "unknown",
        }
    )
    remove = tools["propose_remove_inventory_instance"].invoke(  # type: ignore[attr-defined]
        {"instance_id": unique_id}
    )
    remove_all = tools["propose_remove_all_inventory_instances"].invoke(  # type: ignore[attr-defined]
        {"name": "番茄"}
    )
    confirm = tools["propose_confirm_inventory_instance"].invoke(  # type: ignore[attr-defined]
        {"instance_id": unique_id}
    )

    assert inventory.list_instances() == sorted(
        [first, second, unique],
        key=lambda item: (
            str(item["name"]),
            str(item["stocked_at"]),
            str(item["instance_id"]),
        ),
    )
    assert [value["status"] for value in (add, replace, remove, remove_all, confirm)] == [
        "pending",
        "pending",
        "pending",
        "pending",
        "pending",
    ]
    assert add["arguments"] == {
        "name": "牛奶",
        "quantity_text": "大约一盒",
        "expires_at": "2099-01-02T00:00:00Z",
        "expiry_source": "llm",
    }
    assert replace["arguments"] == {
        "instance_id": unique_id,
        "quantity_text": "1个",
        "expires_at": None,
        "expiry_source": "unknown",
    }
    assert len(actions.repository.list_pending("test")) == 5


def test_exact_get_cannot_choose_duplicate_from_history_without_user_id_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = iter(
        [
            "abcd1111111111111111111111111111",
            "abcd2222222222222222222222222222",
        ]
    )
    monkeypatch.setattr(
        "app.tools.inventory.uuid4", lambda: SimpleNamespace(hex=next(values))
    )
    inventory, actions, _ = inventory_tools(tmp_path)
    first = inventory.add_instance("番茄", "2个", None, "unknown")
    second = inventory.add_instance("番茄", "500g", None, "unknown")
    history = HistoryExcerpt(
        request="查看番茄还有多少",
        response=(
            f"番茄 [{first['instance_id']}]：2个\n"
            f"番茄 [{second['instance_id']}]：500g"
        ),
    )
    agent = KitchenAgent(
        model=None,  # type: ignore[arg-type]
        inventory=inventory,
        recipes=RecipeRouter(
            None,
            TheMealDbClient("https://example.invalid", timeout=1),
        ),
        actions=actions,
    )
    ambiguous_context = AgentContext(
        user_id="test",
        request_text="删掉一个番茄",
        history_segments=[history],
    )
    ambiguous_tools = {
        tool.name: tool
        for tool in agent._create_tools(ambiguous_context, TraceRecorder())
    }
    first_id = str(first["instance_id"])

    ambiguous_tools["get_inventory_instance"].invoke(  # type: ignore[attr-defined]
        {"instance_id": first_id}
    )

    with pytest.raises(ValueError, match="AMBIGUOUS_INSTANCE_MATCH"):
        ambiguous_tools["propose_remove_inventory_instance"].invoke(  # type: ignore[attr-defined]
            {"instance_id": first_id}
        )
    assert actions.repository.list_pending("test") == []

    prefix_context = AgentContext(
        user_id="test",
        request_text=f"删除 {first_id[:8]}",
        history_segments=[history],
    )
    prefix_tools = {
        tool.name: tool
        for tool in agent._create_tools(prefix_context, TraceRecorder())
    }
    prefix_tools["get_inventory_instance"].invoke(  # type: ignore[attr-defined]
        {"instance_id": first_id}
    )
    with pytest.raises(ValueError, match="AMBIGUOUS_INSTANCE_MATCH"):
        prefix_tools["propose_remove_inventory_instance"].invoke(  # type: ignore[attr-defined]
            {"instance_id": first_id}
        )
    assert actions.repository.list_pending("test") == []

    selected_context = AgentContext(
        user_id="test",
        request_text=f"删除 {first_id}",
        history_segments=[history],
    )
    selected_tools = {
        tool.name: tool
        for tool in agent._create_tools(selected_context, TraceRecorder())
    }
    selected_tools["get_inventory_instance"].invoke(  # type: ignore[attr-defined]
        {"instance_id": first_id}
    )
    proposal = selected_tools["propose_remove_inventory_instance"].invoke(  # type: ignore[attr-defined]
        {"instance_id": first_id}
    )

    assert proposal["status"] == "pending"
    assert inventory.list_instances() == [first, second]


def test_recipe_tool_requires_all_configured_allergens() -> None:
    context = AgentContext(
        user_id="test",
        request_text="推荐菜谱",
        allergens=["Peanut", "腰果"],
        allergen_intolerances=["Peanut"],
        custom_allergens=["腰果"],
    )
    recipes = FakeRecipeRouter()
    tool = recipe_tool(context, recipes)

    with pytest.raises(ValueError, match="MISSING_ALLERGEN_FILTER"):
        tool.invoke(  # type: ignore[attr-defined]
            {
                "ingredients": ["tomato"],
                "intolerances": ["Peanut"],
                "exclude_ingredients": [],
                "custom_allergen_mapping": [],
            }
        )


def test_recipe_tool_passes_broad_and_custom_allergens_to_router() -> None:
    context = AgentContext(
        user_id="test",
        request_text="推荐菜谱",
        allergens=["Peanut", "腰果"],
        allergen_intolerances=["Peanut"],
        custom_allergens=["腰果"],
    )
    recipes = FakeRecipeRouter()
    tool = recipe_tool(context, recipes)

    tool.invoke(  # type: ignore[attr-defined]
        {
            "ingredients": ["tomato"],
            "intolerances": ["Peanut"],
            "exclude_ingredients": ["onion"],
            "custom_allergen_mapping": [
                {"original": "腰果", "api_term": "cashew"}
            ],
        }
    )

    assert recipes.kwargs is not None
    assert recipes.kwargs["intolerances"] == ["Peanut"]
    assert recipes.kwargs["exclude_ingredients"] == ["onion", "cashew"]


def test_recipe_tool_rejects_non_english_api_term() -> None:
    context = AgentContext(
        user_id="test",
        request_text="推荐菜谱",
        allergens=["腰果"],
        custom_allergens=["腰果"],
    )
    tool = recipe_tool(context, FakeRecipeRouter())

    with pytest.raises(ValueError, match="INVALID_ENGLISH_API_TERM"):
        tool.invoke(  # type: ignore[attr-defined]
            {
                "ingredients": ["tomato"],
                "intolerances": [],
                "exclude_ingredients": [],
                "custom_allergen_mapping": [
                    {"original": "腰果", "api_term": "腰果"}
                ],
            }
        )
