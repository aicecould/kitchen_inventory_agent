from dataclasses import replace
from datetime import UTC, datetime

import pytest

from app.config import get_settings
from app.expiry import DeepSeekExpiryInferer, ExpiryPrediction


class FakeChatOpenAI:
    response = ExpiryPrediction(expires_at=None)
    instances: list["FakeChatOpenAI"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.schema: type[ExpiryPrediction] | None = None
        self.structured_method: str | None = None
        self.prompt: str | None = None
        self.instances.append(self)

    def with_structured_output(
        self, schema: type[ExpiryPrediction], *, method: str
    ) -> "FakeChatOpenAI":
        self.schema = schema
        self.structured_method = method
        return self

    def invoke(self, messages: object, config: object = None) -> object:
        self.prompt = str(messages)
        return self.response


def build_inferer(monkeypatch: pytest.MonkeyPatch) -> DeepSeekExpiryInferer:
    FakeChatOpenAI.instances.clear()
    monkeypatch.setattr("app.expiry.ChatOpenAI", FakeChatOpenAI)
    settings = replace(
        get_settings(),
        deepseek_api_key="test-key",
        deepseek_model="deepseek-test",
        deepseek_base_url="https://deepseek.invalid",
        http_timeout_seconds=7,
        deepseek_max_output_tokens=123,
    )
    return DeepSeekExpiryInferer(settings)


def test_infer_returns_future_aware_structured_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeChatOpenAI.response = ExpiryPrediction(
        expires_at=datetime(2026, 8, 12, 9, tzinfo=UTC)
    )
    inferer = build_inferer(monkeypatch)

    result = inferer.infer("牛奶", datetime(2026, 8, 7, 9, tzinfo=UTC))

    assert result == datetime(2026, 8, 12, 9, tzinfo=UTC)
    model = FakeChatOpenAI.instances[0]
    assert model.schema is ExpiryPrediction
    assert model.structured_method == "json_mode"
    assert model.kwargs == {
        "api_key": "test-key",
        "base_url": "https://deepseek.invalid",
        "model": "deepseek-test",
        "temperature": 0,
        "timeout": 7,
        "max_retries": 1,
        "use_responses_api": False,
        "max_tokens": 123,
    }
    assert model.prompt is not None
    assert "牛奶" in model.prompt
    assert "2026-08-07T09:00:00+00:00" in model.prompt
    assert "保守" in model.prompt


@pytest.mark.parametrize(
    "predicted",
    [
        None,
        datetime(2026, 8, 7, 8, tzinfo=UTC),
        datetime(2026, 8, 12, 9),
    ],
)
def test_infer_returns_none_for_missing_past_or_naive_expiry(
    monkeypatch: pytest.MonkeyPatch, predicted: datetime | None
) -> None:
    FakeChatOpenAI.response = ExpiryPrediction(expires_at=predicted)
    inferer = build_inferer(monkeypatch)

    result = inferer.infer("牛奶", datetime(2026, 8, 7, 9, tzinfo=UTC))

    assert result is None
