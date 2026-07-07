import pytest

from app.agent import KitchenAgent
from app.context import AgentContext
from app.trace import TraceRecorder


class FakeRecipeRouter:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    def search(self, **kwargs: object) -> list[dict[str, object]]:
        self.kwargs = kwargs
        return []


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
