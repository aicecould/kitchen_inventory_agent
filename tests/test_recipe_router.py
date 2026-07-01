from app.adapters.recipe_api import Recipe
from app.tools.recipe import RecipeRouter


class FakeSpoonacular:
    def search(self, ingredients: list[str], limit: int = 3) -> list[Recipe]:
        assert ingredients == ["tomato"]
        return [
            Recipe(
                id="s1",
                title="Tomato salad",
                instructions="Mix tomato and salt.",
                ingredients=["tomato", "salt"],
                source="spoonacular",
            )
        ]


class FakeMealDb:
    def search(self, ingredients: list[str], limit: int = 3) -> list[Recipe]:
        return [
            Recipe(
                id="m1",
                title="Peanut tomato noodles",
                instructions="Mix peanut and tomato.",
                ingredients=["peanut", "tomato"],
                source="themealdb",
            )
        ]


class FakeTranslator:
    translations = {
        ("番茄", "en"): "tomato",
        ("花生", "en"): "peanut",
        ("Tomato salad", "zh"): "番茄沙拉",
        ("Mix tomato and salt.", "zh"): "混合番茄和盐。",
        ("tomato", "zh"): "番茄",
        ("salt", "zh"): "盐",
    }

    def translate(self, text: str, target: str, source: str = "auto") -> str:
        return self.translations.get((text, target), text)


def test_dual_route_filters_translated_allergen_and_localizes() -> None:
    router = RecipeRouter(FakeSpoonacular(), FakeMealDb(), FakeTranslator())  # type: ignore[arg-type]
    results = router.search(
        ingredients=["番茄"],
        preferences=[],
        allergens=["花生"],
        target_language="zh",
    )
    assert len(results) == 1
    assert results[0]["title"] == "番茄沙拉"
    assert results[0]["source"] == "spoonacular"
