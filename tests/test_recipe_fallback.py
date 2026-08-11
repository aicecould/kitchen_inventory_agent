from app.adapters.recipe_api import Recipe
from app.tools.recipe import RecipeRouter
from recommender.models import IngredientTranslation


REQUEST = {
    "ingredients": ["tomato"],
    "ingredient_mapping": [
        IngredientTranslation(original="腰果", api_term="cashew"),
    ],
    "allergen_categories": ["Peanut"],
    "custom_allergens": ["腰果"],
    "exclude_ingredients": [],
}


def recipe(recipe_id: str) -> Recipe:
    return Recipe(
        id=recipe_id,
        title=f"Recipe {recipe_id}",
        instructions="Cook tomato.",
        ingredients=["tomato"],
        source="fake",
    )


def local_payload(recipe_item: Recipe) -> dict[str, object]:
    payload = recipe_item.as_dict()
    payload.update(
        {
            "used_ingredient_count": 1,
            "missing_ingredient_count": 0,
            "rrf_score": 0.01,
            "fts_rank": 1,
            "semantic_rank": None,
        }
    )
    return payload


class FakeLocal:
    def __init__(self, recipes: list[Recipe], fail: bool = False) -> None:
        self.recipes = recipes
        self.fail = fail
        self.calls = 0
        self.requests: list[object] = []

    def search(self, request: object, trace: object = None) -> list[dict[str, object]]:
        self.calls += 1
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("local unavailable")
        return [local_payload(item) for item in self.recipes]


class FakeSpoonacular:
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
            raise RuntimeError("quota exhausted")
        return self.recipes


class FakeMealDb:
    def __init__(self, recipes: list[Recipe], fail: bool = False) -> None:
        self.recipes = recipes
        self.fail = fail
        self.calls = 0

    def search(self, ingredients: list[str], limit: int = 3) -> list[Recipe]:
        self.calls += 1
        assert ingredients == ["tomato"]
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


def router(
    local: object,
    spoonacular: object,
    themealdb: object,
    web: object,
) -> RecipeRouter:
    return RecipeRouter(local, spoonacular, themealdb, web)  # type: ignore[arg-type]


def test_three_local_results_skip_all_external_sources() -> None:
    local = FakeLocal([recipe("l1"), recipe("l2"), recipe("l3")])
    spoonacular = FakeSpoonacular([recipe("s1")])
    themealdb = FakeMealDb([recipe("m1")])
    web = FakeWebSearch([recipe("w1")])

    result = router(local, spoonacular, themealdb, web).search(**REQUEST)

    assert [item["id"] for item in result] == ["l1", "l2", "l3"]
    assert spoonacular.calls == themealdb.calls == web.calls == 0
    assert local.calls == 1


def test_two_local_results_are_kept_and_spoonacular_fills_to_three() -> None:
    local = FakeLocal([recipe("l1"), recipe("l2")])
    spoonacular = FakeSpoonacular([recipe("s1")])
    themealdb = FakeMealDb([recipe("m1")])
    web = FakeWebSearch([recipe("w1")])

    result = router(local, spoonacular, themealdb, web).search(**REQUEST)

    assert [item["id"] for item in result] == ["l1", "l2", "s1"]
    assert spoonacular.calls == 1
    assert themealdb.calls == web.calls == 0


def test_one_local_result_uses_themealdb_after_empty_spoonacular() -> None:
    local = FakeLocal([recipe("l1")])
    spoonacular = FakeSpoonacular([])
    themealdb = FakeMealDb([recipe("m1"), recipe("m2")])
    web = FakeWebSearch([recipe("w1")])

    result = router(local, spoonacular, themealdb, web).search(**REQUEST)

    assert [item["id"] for item in result] == ["l1", "m1", "m2"]
    assert spoonacular.calls == themealdb.calls == 1
    assert web.calls == 0


def test_web_search_fills_after_empty_apis() -> None:
    local = FakeLocal([recipe("l1")])
    spoonacular = FakeSpoonacular([])
    themealdb = FakeMealDb([])
    web = FakeWebSearch([recipe("w1"), recipe("w2")])

    result = router(local, spoonacular, themealdb, web).search(**REQUEST)

    assert [item["id"] for item in result] == ["l1", "w1", "w2"]
    assert spoonacular.calls == themealdb.calls == web.calls == 1


def test_local_failure_continues_from_spoonacular() -> None:
    local = FakeLocal([], fail=True)
    spoonacular = FakeSpoonacular([recipe("s1"), recipe("s2"), recipe("s3")])
    themealdb = FakeMealDb([recipe("m1")])
    web = FakeWebSearch([recipe("w1")])

    result = router(local, spoonacular, themealdb, web).search(**REQUEST)

    assert [item["id"] for item in result] == ["s1", "s2", "s3"]
    assert spoonacular.calls == 1
    assert themealdb.calls == web.calls == 0


def test_all_sources_below_minimum_returns_real_results_and_degrades() -> None:
    local = FakeLocal([recipe("l1")])
    spoonacular = FakeSpoonacular([])
    themealdb = FakeMealDb([])
    web = FakeWebSearch([])
    events: list[tuple[str, str, str, str, int]] = []

    result = router(local, spoonacular, themealdb, web).search(
        **REQUEST,
        trace=lambda *event: events.append(event),
    )

    assert [item["id"] for item in result] == ["l1"]
    assert ("external_api", "deepseek_web_search", "empty") in [
        event[:3] for event in events
    ]
    assert any(event[2] == "degraded" for event in events)


def test_cross_source_duplicates_are_removed_by_title() -> None:
    local = FakeLocal([recipe("l1")])
    duplicate = recipe("s1")
    duplicate.title = "Recipe l1"
    spoonacular = FakeSpoonacular([duplicate, recipe("s2"), recipe("s3")])
    themealdb = FakeMealDb([recipe("m1")])
    web = FakeWebSearch([recipe("w1")])

    result = router(local, spoonacular, themealdb, web).search(**REQUEST)

    assert [item["id"] for item in result] == ["l1", "s2", "s3"]
    assert themealdb.calls == 0
