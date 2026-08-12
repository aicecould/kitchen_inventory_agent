from types import SimpleNamespace

import pytest

from evaluation.budget import CallBudget
from evaluation.external_recipes import BudgetExhausted, CountingRecipeClient


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    def search(self, **kwargs):
        self.calls += 1
        return [SimpleNamespace(id="recipe")]


def test_spoonacular_proxy_stops_before_sixth_call() -> None:
    client = FakeClient()
    proxy = CountingRecipeClient(
        "spoonacular", client, CallBudget({"spoonacular": 5})
    )
    for _ in range(5):
        proxy.search(ingredients=["egg"])
    with pytest.raises(BudgetExhausted):
        proxy.search(ingredients=["egg"])
    assert client.calls == 5
    assert proxy.budget_exhausted == 1
