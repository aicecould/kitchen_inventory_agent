import json
from uuid import uuid4

import httpx
import pytest

from app.llm import ObservedChatOpenAI
from app.llm_log import LLMLogStore, PersistentLLMCallback, logged_span
from app.log_migration import migrate
from tests.test_llm_log import queries


@pytest.mark.parametrize("reported", ["served-model-revision", None])
def test_wire_response_identity_survives_sdk_aggregation(tmp_path, reported):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        body = {"id": "completion-1", "object": "chat.completion", "created": 0,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                             "finish_reason": "stop"}]}
        if reported:
            body["model"] = reported
        return httpx.Response(200, json=body)

    store = LLMLogStore(tmp_path / "identity.db")
    callback = PersistentLLMCallback(store, "agent", "configured-alias")
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = ObservedChatOpenAI(model="client-alias", api_key="test", http_client=client,
                                   base_url="https://example.invalid/v1", max_retries=0)
        with logged_span(store, "request", "test"):
            model.invoke("hi", model="request-override", config={"callbacks": [callback]})
    record, = queries(store.path)
    step, = record["steps"]
    identity = step["model_identity"]
    assert requests[0]["model"] == identity["requested_model"] == "request-override"
    assert identity["configured_model"] == "configured-alias"
    assert identity["response_model"] == step["model"] == reported
    assert identity["source"] == ("response_body" if reported else "unknown")
    assert record["models"] == ([reported] if reported else [])
    assert identity["independently_verified"] is False


def test_failed_request_never_reports_configured_model_as_actual(tmp_path):
    store = LLMLogStore(tmp_path / "failure.db")
    callback = PersistentLLMCallback(store, "agent", "configured")
    with logged_span(store, "request", "test"):
        run_id = uuid4()
        callback.on_chat_model_start({}, [], run_id=run_id,
                                     invocation_params={"model": "requested"})
        callback.on_llm_error(TimeoutError(), run_id=run_id)
    record, = queries(store.path)
    assert record["models"] == []
    assert record["steps"][0]["model_identity"]["requested_model"] == "requested"
    assert record["steps"][0]["model_identity"]["response_model"] is None


def test_legacy_migration_uses_preserved_metadata_and_keeps_backup(tmp_path):
    import sqlite3

    path = tmp_path / "legacy.db"
    store = LLMLogStore(path)
    with logged_span(store, "request", "test"):
        pass
    record, = queries(path)
    record["schema_version"] = 1
    record["steps"] = [{"kind": "llm", "model": "wrong-alias", "parameters": {"model": "requested"},
                        "content": [{"response_metadata": {"model_name": "reported"}}]}]
    record["models"] = ["wrong-alias"]
    with sqlite3.connect(path) as db:
        db.execute("UPDATE query_logs SET record_json=?", (json.dumps(record),))
    backup, count = migrate(path)
    assert count == 1
    assert queries(backup)[0]["models"] == ["wrong-alias"]
    migrated, = queries(path)
    assert migrated["models"] == ["reported"]
    assert migrated["steps"][0]["legacy_model"] == "wrong-alias"
    assert migrated["steps"][0]["model_identity"]["requested_model"] == "requested"
    assert migrate(path)[1] == 0
