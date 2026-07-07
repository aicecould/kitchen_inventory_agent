from time import monotonic
from typing import Any

from app.adapters.vision_api import VisionApiClient


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {
            "result_num": 1,
            "result": [{"name": "土豆", "score": 0.91}],
        }


def test_ingredient_api_uses_top_num_and_name_field(monkeypatch: Any) -> None:
    captured: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        captured["url"] = url
        captured["data"] = kwargs["data"]
        return FakeResponse()

    monkeypatch.setattr("app.adapters.vision_api.httpx.post", fake_post)
    client = VisionApiClient("https://example.test/ingredient", "key", "secret")
    client._access_token = "token"
    client._token_expires_at = monotonic() + 60

    detections = client.recognize(b"image")

    assert captured["url"] == "https://example.test/ingredient"
    assert captured["data"]["top_num"] == 5  # type: ignore[index]
    assert detections[0].name == "土豆"
    assert detections[0].confidence == 0.91
