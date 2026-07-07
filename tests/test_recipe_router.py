from app.adapters.recipe_api import Recipe
from app.tools.recipe import RecipeRouter


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
    def __init__(self, recipes: list[Recipe]) -> None:
        self.recipes = recipes
        self.calls = 0

    def search(self, ingredients: list[str], limit: int = 3) -> list[Recipe]:
        self.calls += 1
        return self.recipes


SAFE_RECIPE = Recipe(
    id="s1",
    title="Tomato salad",
    instructions="Mix tomato and salt.",
    ingredients=["tomato", "salt"],
    source="spoonacular",
)


def search(router: RecipeRouter) -> list[dict[str, object]]:
    return router.search(
        ingredients=["tomato"],
        intolerances=["Peanut"],
        exclude_ingredients=["cashew"],
    )


def test_primary_result_skips_themealdb() -> None:
    spoonacular = FakeSpoonacular([SAFE_RECIPE])
    themealdb = FakeMealDb([])
    results = search(RecipeRouter(spoonacular, themealdb))  # type: ignore[arg-type]

    assert results[0]["source"] == "spoonacular"
    assert spoonacular.calls == 1
    assert themealdb.calls == 0


def test_themealdb_is_used_when_spoonacular_fails() -> None:
    fallback = Recipe(
        id="m1",
        title="Tomato soup",
        instructions="Cook tomato.",
        ingredients=["tomato"],
        source="themealdb",
    )
    spoonacular = FakeSpoonacular(fail=True)
    themealdb = FakeMealDb([fallback])
    results = search(RecipeRouter(spoonacular, themealdb))  # type: ignore[arg-type]

    assert results[0]["source"] == "themealdb"
    assert themealdb.calls == 1


def test_primary_filtered_to_empty_uses_safe_fallback() -> None:
    unsafe = Recipe(
        id="s2",
        title="Cashew tomato salad",
        instructions="Add cashew.",
        ingredients=["cashew", "tomato"],
        source="spoonacular",
    )
    fallback = Recipe(
        id="m2",
        title="Tomato rice",
        instructions="Cook tomato and rice.",
        ingredients=["tomato", "rice"],
        source="themealdb",
    )
    themealdb = FakeMealDb([fallback])
    results = search(RecipeRouter(FakeSpoonacular([unsafe]), themealdb))  # type: ignore[arg-type]

    assert results[0]["source"] == "themealdb"
    assert themealdb.calls == 1


def test_fallback_trace_uses_safe_error_summary() -> None:
    fallback = Recipe(
        id="m3",
        title="Tomato soup",
        instructions="Cook tomato.",
        ingredients=["tomato"],
        source="themealdb",
    )
    events: list[tuple[str, str, str, str, int]] = []
    router = RecipeRouter(FakeSpoonacular(fail=True), FakeMealDb([fallback]))  # type: ignore[arg-type]

    results = router.search(
        ingredients=["tomato"],
        intolerances=["Peanut"],
        exclude_ingredients=["cashew"],
        trace=lambda *event: events.append(event),
    )

    assert results[0]["source"] == "themealdb"
    assert [event[1:3] for event in events] == [
        ("spoonacular", "failed"),
        ("themealdb", "success"),
    ]
    assert events[0][3] == "RuntimeError; using fallback"
    assert "quota exhausted" not in str(events)
