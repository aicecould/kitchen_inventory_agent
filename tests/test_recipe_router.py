from app.adapters.recipe_api import Recipe
from app.tools.recipe import RecipeRouter
from recommender.models import IngredientTranslation


class FakeSpoonacular:
    def __init__(self, recipes: list[Recipe] | None = None, fail: bool = False) -> None:
        self.recipes = recipes or []
        self.fail = fail
        self.calls = 0

    def search(self, **kwargs: object) -> list[Recipe]:
        self.calls += 1
        assert kwargs["ingredients"] == ["tomato"]
        assert kwargs["intolerances"] == ["Peanut"]
        assert kwargs["exclude_ingredients"] == ["cashew"]
        if self.fail:
            raise RuntimeError("quota exhausted")
        return self.recipes


class FakeMealDb:
    def __init__(self, recipes: list[Recipe], fail: bool = False) -> None:
        self.recipes = recipes
        self.fail = fail
        self.calls = 0

    def search(self, ingredients: list[str], limit: int = 3) -> list[Recipe]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("meal db unavailable")
        return self.recipes


class FakeWebSearch:
    def __init__(self, recipes: list[Recipe], fail: bool = False) -> None:
        self.recipes = recipes
        self.fail = fail
        self.calls = 0

    def search(self, **kwargs: object) -> list[Recipe]:
        self.calls += 1
        assert kwargs["ingredients"] == ["tomato"]
        assert kwargs["intolerances"] == ["Peanut"]
        assert kwargs["exclude_ingredients"] == ["cashew"]
        if self.fail:
            raise RuntimeError("search unavailable")
        return self.recipes


def recipe(recipe_id: str, source: str = "spoonacular") -> Recipe:
    return Recipe(
        id=recipe_id,
        title=f"Recipe {recipe_id}",
        instructions="Mix tomato and salt.",
        ingredients=["tomato", "salt"],
        source=source,
    )


SAFE_RECIPES = [recipe("s1"), recipe("s2"), recipe("s3")]


def search(router: RecipeRouter) -> list[dict[str, object]]:
    return router.search(
        ingredients=["tomato"],
        ingredient_mapping=[
            IngredientTranslation(original="腰果", api_term="cashew"),
        ],
        allergen_categories=["Peanut"],
        custom_allergens=["腰果"],
        exclude_ingredients=[],
    )


def test_primary_result_skips_themealdb() -> None:
    spoonacular = FakeSpoonacular(SAFE_RECIPES)
    themealdb = FakeMealDb([recipe("m1", "themealdb")])
    results = search(RecipeRouter(None, spoonacular, themealdb))  # type: ignore[arg-type]

    assert [item["source"] for item in results] == ["spoonacular"] * 3
    assert spoonacular.calls == 1
    assert themealdb.calls == 0


def test_themealdb_is_used_when_spoonacular_fails() -> None:
    fallback = [recipe("m1", "themealdb"), recipe("m2", "themealdb"), recipe("m3", "themealdb")]
    spoonacular = FakeSpoonacular(fail=True)
    themealdb = FakeMealDb(fallback)
    results = search(RecipeRouter(None, spoonacular, themealdb))  # type: ignore[arg-type]

    assert [item["source"] for item in results] == ["themealdb"] * 3
    assert themealdb.calls == 1


def test_primary_filtered_to_empty_uses_safe_fallback() -> None:
    unsafe = recipe("s2")
    unsafe.ingredients = ["cashew", "tomato"]
    unsafe.instructions = "Add cashew."
    unsafe.title = "Cashew tomato salad"
    fallback = [recipe("m2", "themealdb"), recipe("m3", "themealdb"), recipe("m4", "themealdb")]
    themealdb = FakeMealDb(fallback)
    results = search(RecipeRouter(None, FakeSpoonacular([unsafe]), themealdb))  # type: ignore[arg-type]

    assert [item["source"] for item in results] == ["themealdb"] * 3
    assert themealdb.calls == 1


def test_fallback_trace_uses_safe_error_summary() -> None:
    fallback = [recipe("m3", "themealdb"), recipe("m4", "themealdb"), recipe("m5", "themealdb")]
    events: list[tuple[str, str, str, str, int]] = []
    router = RecipeRouter(None, FakeSpoonacular(fail=True), FakeMealDb(fallback))  # type: ignore[arg-type]

    results = router.search(
        ingredients=["tomato"],
        ingredient_mapping=[
            IngredientTranslation(original="腰果", api_term="cashew"),
        ],
        allergen_categories=["Peanut"],
        custom_allergens=["腰果"],
        exclude_ingredients=[],
        trace=lambda *event: events.append(event),
    )

    assert [item["source"] for item in results] == ["themealdb"] * 3
    assert [(event[0], event[1]) for event in events if event[0] == "external_api"] == [
        ("external_api", "spoonacular"),
        ("external_api", "themealdb"),
    ]
    assert events[1][2] == "failed"
    assert events[2][2] == "success"
    assert "quota exhausted" not in str(events)


def test_deepseek_web_search_is_last_resort_after_empty_recipe_apis() -> None:
    web_recipe = recipe("web-1", "deepseek_web_search")
    web_recipe.source_url = "https://recipes.example/tomato-noodles"
    web = FakeWebSearch([web_recipe])
    events: list[tuple[str, str, str, str, int]] = []
    router = RecipeRouter(None, FakeSpoonacular([]), FakeMealDb([]), web)  # type: ignore[arg-type]

    results = router.search(
        ingredients=["tomato"],
        ingredient_mapping=[
            IngredientTranslation(original="腰果", api_term="cashew"),
        ],
        allergen_categories=["Peanut"],
        custom_allergens=["腰果"],
        exclude_ingredients=[],
        trace=lambda *event: events.append(event),
    )

    assert results == [web_recipe.as_dict()]
    assert web.calls == 1
    assert [event[1] for event in events if event[0] == "external_api"] == [
        "spoonacular",
        "themealdb",
        "deepseek_web_search",
    ]


def test_deepseek_web_search_results_still_apply_allergen_filter() -> None:
    unsafe = recipe("web-unsafe", "deepseek_web_search")
    unsafe.ingredients = ["cashew", "tomato"]
    unsafe.instructions = "Add cashew."
    unsafe.title = "Cashew tomato noodles"
    web = FakeWebSearch([unsafe])
    router = RecipeRouter(None, FakeSpoonacular([]), FakeMealDb([]), web)  # type: ignore[arg-type]

    assert search(router) == []
    assert web.calls == 1


def test_deepseek_web_search_is_used_after_both_recipe_apis_fail() -> None:
    web_recipe = recipe("web-2", "deepseek_web_search")
    web_recipe.source_url = "https://recipes.example/tomato-stew"
    events: list[tuple[str, str, str, str, int]] = []
    router = RecipeRouter(
        None,
        FakeSpoonacular(fail=True),
        FakeMealDb([], fail=True),
        FakeWebSearch([web_recipe]),
    )  # type: ignore[arg-type]

    results = router.search(
        ingredients=["tomato"],
        ingredient_mapping=[
            IngredientTranslation(original="腰果", api_term="cashew"),
        ],
        allergen_categories=["Peanut"],
        custom_allergens=["腰果"],
        exclude_ingredients=[],
        trace=lambda *event: events.append(event),
    )

    assert results == [web_recipe.as_dict()]
    assert [event[1] for event in events if event[0] == "external_api"] == [
        "spoonacular",
        "themealdb",
        "deepseek_web_search",
    ]
