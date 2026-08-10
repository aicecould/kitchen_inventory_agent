from typing import Any

from app.adapters.recipe_api import DeepSeekWebSearchClient


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {
            "stop_reason": "end_turn",
            "content": [
                {
                    "type": "text",
                    "text": (
                        '{"recipes":[{"title":"Tomato Noodles",'
                        '"instructions":"Cook tomato with noodles.",'
                        '"ingredients":["tomato","noodles"],'
                        '"source_url":"https://recipes.example/tomato-noodles",'
                        '"image_url":null}]}'
                    ),
                }
            ],
        }


def test_deepseek_client_uses_official_anthropic_web_search_tool(
    monkeypatch: Any,
) -> None:
    captured: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("app.adapters.recipe_api.httpx.post", fake_post)
    client = DeepSeekWebSearchClient(
        "https://api.deepseek.com", "secret", "deepseek-v4-flash"
    )

    recipes = client.search(
        ingredients=["tomato"],
        intolerances=["Peanut"],
        exclude_ingredients=["cashew"],
        limit=3,
    )

    assert captured["url"] == "https://api.deepseek.com/anthropic/v1/messages"
    assert captured["headers"] == {
        "content-type": "application/json",
        "x-api-key": "secret",
        "anthropic-version": "2023-06-01",
    }
    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload["tools"] == [
        {"type": "web_search_20250305", "name": "web_search", "max_uses": 1}
    ]
    assert recipes[0].source == "deepseek_web_search"
    assert recipes[0].source_url == "https://recipes.example/tomato-noodles"
