import pytest
import json
from pydantic import BaseModel, ConfigDict

from app.structured_output import invoke_validated_json


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int


class FakeModel:
    def __init__(self, content: object) -> None:
        self.content = content
        self.method: str | None = None
        self.schema: object = None
        self.calls: list[tuple[object, object]] = []

    def with_structured_output(
        self, schema: object, *, method: str
    ) -> "FakeModel":
        self.schema = schema
        self.method = method
        return self

    def invoke(self, messages: object, config: object = None) -> object:
        self.calls.append((messages, config))
        if isinstance(self.content, Exception):
            raise self.content
        if isinstance(self.content, str):
            import json
            return json.loads(self.content.removeprefix("```json\n").removesuffix("\n```"))
        return self.content


def test_adapter_parses_fenced_json_and_validates_schema() -> None:
    model = FakeModel("```json\n{\"value\": 3}\n```")
    result = invoke_validated_json(
        model, Schema, [{"role": "user", "content": "return value"}]
    )
    assert result == Schema(value=3)
    assert model.method == "json_mode"
    messages = model.calls[0][0]
    assert "JSON" in messages[-1]["content"]
    assert "value" in messages[-1]["content"]


def test_adapter_rejects_prose_or_extra_fields() -> None:
    with pytest.raises(Exception):
        invoke_validated_json(FakeModel("value is 3"), Schema, "return value")
    with pytest.raises(ValueError):
        invoke_validated_json(
            FakeModel('{"value": 3, "untrusted": true}'), Schema, "return value"
        )


def test_adapter_forwards_callbacks_without_response_format_or_tools() -> None:
    model = FakeModel('{"value": 3}')
    callback = object()
    invoke_validated_json(model, Schema, "return value", callbacks=[callback])
    assert model.calls[0][1] == {"callbacks": [callback]}


def test_adapter_embeds_schema_as_valid_json() -> None:
    model = FakeModel('{"value": 3}')

    invoke_validated_json(model, Schema, "return value")

    content = model.calls[0][0][-1]["content"]
    encoded_schema = content.split("JSON schema: ", 1)[1].split(
        ". JSON output is required.", 1
    )[0]
    assert json.loads(encoded_schema) == Schema.model_json_schema()
