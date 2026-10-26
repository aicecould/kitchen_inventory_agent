import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import ToolMessage
from langchain_openai.chat_models.base import _convert_message_to_dict

from llm.qwen3_1_7b import serve

TOOL = {"type": "function", "function": {"name": "list_inventory_instances",
        "parameters": {"type": "object", "properties": {}}}}


@pytest.fixture
def qwen_api(monkeypatch):
    runtime = SimpleNamespace(device="xpu:0", output="你好", received=None)

    def generate(messages, max_tokens, temperature, tools=None):
        runtime.received = (messages, max_tokens, tools)
        return runtime.output, 20, 10, "stop"

    runtime.generate = generate
    monkeypatch.setattr(serve, "QwenRuntime", lambda: runtime)
    with TestClient(serve.app) as client:
        yield client, runtime


def test_tool_call_and_tool_result_roundtrip(qwen_api):
    client, runtime = qwen_api
    runtime.output = '<tool_call>{"name":"list_inventory_instances","arguments":{}}</tool_call>'
    response = client.post("/v1/chat/completions", json={
        "model": serve.MODEL_ID, "messages": [{"role": "user", "content": "查看库存"}],
        "tools": [TOOL], "max_completion_tokens": 128,
    })
    assert response.status_code == 200, response.text
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == {}
    assert runtime.received[1:] == (128, [TOOL])
    runtime.output = "库存为空。"
    response = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "查看库存"}, choice["message"],
                     {"role": "tool", "tool_call_id": call["id"], "content": "[]"}],
        "tools": [TOOL],
    })
    assert response.status_code == 200, response.text
    assert runtime.received[0][-1]["tool_call_id"] == call["id"]
    assert response.json()["choices"][0]["message"]["content"] == "库存为空。"


@pytest.mark.parametrize("output", [
    '<tool_call>{"name":"delete_everything","arguments":{}}</tool_call>',
    '<tool_call>{"name":"list_inventory_instances","arguments":',
    '<tool_call>{"name":"list_inventory_instances","arguments":"bad"}</tool_call>',
])
def test_invalid_generated_tools_are_rejected(qwen_api, output):
    client, runtime = qwen_api
    runtime.output = output
    response = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "查看库存"}], "tools": [TOOL],
    })
    assert response.status_code == 502


def test_json_mode_validates_output(qwen_api):
    client, runtime = qwen_api
    request = {"messages": [{"role": "user", "content": "返回 JSON"}],
               "response_format": {"type": "json_object"}}
    runtime.output = '{"ok":true}'
    assert client.post("/v1/chat/completions", json=request).status_code == 200
    assert "JSON object" in runtime.received[0][0]["content"]
    runtime.output = "不是JSON"
    assert client.post("/v1/chat/completions", json=request).status_code == 502


def test_langchain_list_tool_results_are_preserved(qwen_api):
    client, runtime = qwen_api
    inventory = [{"instance_id": "example", "name": "鸡蛋", "quantity_text": "一盒"}]
    message = _convert_message_to_dict(ToolMessage(content=inventory, tool_call_id="call_1"))
    response = client.post("/v1/chat/completions", json={"messages": [message]})
    assert response.status_code == 200, response.text
    assert json.loads(runtime.received[0][0]["content"]) == inventory


def test_required_tool_cannot_silently_return_plain_text(qwen_api):
    client, _ = qwen_api
    response = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "查看库存"}],
        "tools": [TOOL], "tool_choice": "required",
    })
    assert response.status_code == 502
