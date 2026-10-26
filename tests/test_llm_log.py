from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import sqlite3
import subprocess
import sys
from uuid import uuid4

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from app.config import get_settings
from app.llm_log import (
    LLMLogStore, PersistentLLMCallback, fingerprint,
    logged_span, logging_context, log_store, request_context,
)


def queries(path):
    with sqlite3.connect(path) as db:
        # One returned JSON document corresponds to exactly one actual SQLite row.
        return [json.loads(row[0]) for row in db.execute(
            "SELECT record_json FROM query_logs ORDER BY started_at")]


class ToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class StructuredToolModel(ToolModel):
    def with_structured_output(self, schema, *, method):
        from langchain_core.output_parsers import PydanticOutputParser
        return self | PydanticOutputParser(pydantic_object=schema)


def test_agent_tool_round_trip_produces_exactly_one_query_row(tmp_path):
    store = LLMLogStore(tmp_path / "queries.db")
    callback = PersistentLLMCallback(store, "kitchen_agent", "deepseek-test")

    @tool
    def lookup(name: str) -> str:
        """Look up food."""
        return f"库存：{name}"

    model = ToolModel(responses=[
        AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"name": "鸡蛋"}, "id": "call-1"}],
                  usage_metadata={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13}),
        AIMessage(content="有鸡蛋", response_metadata={"model_name": "deepseek-test"}, usage_metadata={"input_tokens": 15, "output_tokens": 4, "total_tokens": 19}),
    ])
    with request_context("alice", "segment-1"), logged_span(
        store, "request", "test", prompt={"text": "有鸡蛋吗"}
    ) as query:
        result = create_agent(model, tools=[lookup]).invoke(
            {"messages": [{"role": "user", "content": "有鸡蛋吗"}]},
            config={"callbacks": [callback]},
        )
        query["content"] = result["messages"][-1].content
        # The Query already exists while its components run.
        assert len(queries(store.path)) == 1
        assert queries(store.path)[0]["result"]["status"] == "running"
    records = queries(store.path)
    assert len(records) == 1
    record = records[0]
    assert record["user_id"] == "alice" and record["segment_id"] == "segment-1"
    assert record["result"] == {"content": "有鸡蛋", "status": "success", "error_type": None}
    assert record["tokens"]["total_tokens"] == 32 and record["tokens"]["complete"]
    assert record["models"] == ["deepseek-test"]
    calls = [s for s in record["steps"] if s["kind"] == "llm"]
    tools = [s for s in record["steps"] if s["kind"] == "tool"]
    assert len(calls) == 2 and len(tools) == 1
    assert calls[0]["tool_calls"][0]["id"] == tools[0]["content"]["tool_call_id"] == "call-1"
    assert "有鸡蛋吗" in json.dumps(calls[0]["prompt"], ensure_ascii=False)
    assert "鸡蛋" in json.dumps(tools[0]["prompt"], ensure_ascii=False)
    assert all(s["kind"] != "chain" for s in record["steps"])
    ids = {record["id"], *(s["id"] for s in record["steps"])}
    assert all(s["parent_id"] in ids for s in record["steps"])


def test_parallel_tools_and_requests_never_split_or_lose_details(tmp_path):
    store = LLMLogStore(tmp_path / "parallel.db")
    callback = PersistentLLMCallback(store, "agent", "m")

    def component(index):
        # Exercise the actual LangChain callback with parallel worker contexts.
        run_id = uuid4()
        callback.on_tool_start({"name": f"tool-{index}"}, str(index), run_id=run_id)
        callback.on_tool_end({"value": index}, run_id=run_id)

    def request(index):
        from contextvars import copy_context
        with request_context(str(index), f"s-{index}"), logged_span(store, "request", "test"):
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(copy_context().run, component, i) for i in range(6)]
                for future in futures:
                    future.result()

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(request, range(8)))
    records = queries(store.path)
    assert len(records) == 8
    assert len({r["id"] for r in records}) == 8
    for record in records:
        assert record["segment_id"] == f's-{record["user_id"]}'
        assert len(record["steps"]) == 6
        assert {s["content"]["value"] for s in record["steps"]} == set(range(6))
        assert all(s["status"] == "success" for s in record["steps"])


def test_errors_retries_and_interruption_stay_in_one_record(tmp_path):
    store = LLMLogStore(tmp_path / "failure.db")
    callback = PersistentLLMCallback(store, "test", "model")
    with pytest.raises(ValueError):
        with request_context("u", "s"), logged_span(store, "request", "test"):
            run_id = uuid4()
            callback.on_chat_model_start({}, [], run_id=run_id)
            callback.on_llm_error(TimeoutError("private credential"), run_id=run_id)
            with logged_span(store, "llm", "retry") as detail:
                detail["content"] = "retry completed"
            store.start("component", "interrupted")
            raise ValueError("private credential")
    record, = queries(store.path)
    assert record["result"]["status"] == "failed"
    assert record["result"]["error_type"] == "ValueError"
    assert record["steps"][0]["status"] == "failed"
    assert record["steps"][0]["error_type"] == "TimeoutError"
    assert record["steps"][-1]["status"] == "running"
    assert record["steps"][-1]["time"]["ended_at"] is None
    assert record["tokens"]["total_tokens"] is None
    assert not record["tokens"]["complete"]
    assert "private credential" not in json.dumps(record)


def test_reopen_preserves_query_and_partial_progress(tmp_path):
    store = LLMLogStore(tmp_path / "progress.db")
    with request_context("u", "s"), logged_span(store, "request", "test"):
        with logged_span(store, "component", "first") as detail:
            detail["content"] = {"decision": True}
        reopened = LLMLogStore(store.path)
        record, = queries(reopened.path)
        assert record["result"]["status"] == "running"
        assert record["steps"][0]["content"] == {"decision": True}
    assert queries(store.path)[0]["result"]["status"] == "success"


def test_cli_always_exports_whole_queries_and_preserves_missing_usage(tmp_path):
    store = LLMLogStore(tmp_path / "export.db")
    with logged_span(store, "request", "test", prompt={"api_key": "private"}) as query:
        for usage in (None, {"prompt_tokens": 12, "completion_tokens": 3, "prompt_cache_hit_tokens": 8}):
            with logged_span(store, "llm", "agent", model="m") as detail:
                detail["usage"] = usage
                detail["model"] = "m"
        query["content"] = "中文"
    command = [sys.executable, "-m", "app.log_analysis", "--db", str(store.path)]
    exported = [json.loads(line) for line in subprocess.check_output(
        [*command, "--kind", "llm", "--model", "m"], text=True).splitlines()]
    assert len(exported) == 1 and len(exported[0]["steps"]) == 2
    assert exported[0]["result"]["content"] == "中文"
    assert exported[0]["steps"][1]["usage"]["prompt_cache_hit_tokens"] == 8
    assert "private" not in json.dumps(exported)
    summary = json.loads(subprocess.check_output([*command, "--summary"], text=True))
    assert summary["queries"] == 1 and summary["model_calls"] == 2
    assert summary["calls_with_usage"] == 1 and summary["total_tokens"] == 15
    assert not summary["usage_complete"]


def test_disabled_logging_and_storage_failure_do_not_break_business(tmp_path, monkeypatch, caplog):
    settings = replace(get_settings(), llm_logging_enabled=False,
                       llm_log_db_path=tmp_path / "disabled.db")
    assert log_store(settings) is None and not settings.llm_log_db_path.exists()
    store = LLMLogStore(tmp_path / "failed.db")

    def fail(*args, **kwargs):
        raise OSError("sensitive error")

    monkeypatch.setattr(store, "begin_query", fail)
    with logged_span(store, "request", "test") as result:
        result["content"] = "business completed"
    assert "LLM log persistence failed (OSError)" in caplog.text
    assert "sensitive error" not in caplog.text


def test_runtime_and_provenance_are_saved_once_per_query(tmp_path):
    settings = replace(get_settings(), deepseek_api_key="private-key",
                       deepseek_base_url="https://user:password@example.invalid/v1?key=private-key")
    store = log_store(settings)
    with logging_context(source="evaluation", run_id="run-1", sample_id="case-1"):
        with request_context("u", "s"), logged_span(store, "request", "test"):
            callback = PersistentLLMCallback(store, "kitchen_agent", "m")
            model = ToolModel(responses=[AIMessage(content="ok")])
            model.invoke("hi", config={"callbacks": [callback],
                                      "metadata": {"log_component": "answer_rewrite"}})
    record, = queries(store.path)
    assert record["source"] == "evaluation"
    assert record["metadata"]["sample_id"] == "case-1"
    assert record["metadata"]["run_id"] == "run-1"
    assert record["runtime"]["config"]["deepseek_model"] == settings.deepseek_model
    assert record["runtime"]["packages"]["langchain-core"]
    assert record["runtime"]["artifacts"]["food_classifier_model_path"]["sha256"]
    assert record["steps"][0]["name"] == "answer_rewrite"
    assert "runtime" not in record["steps"][0]["metadata"]
    assert "private-key" not in json.dumps(record) and "password" not in json.dumps(record)
    assert fingerprint({"b": 2, "a": 1}) == fingerprint({"a": 1, "b": 2})


def test_classifier_facts_survive_query_failure(tmp_path):
    from classifier.model import ClassificationResult
    from app.trace import TraceRecorder

    store = LLMLogStore(tmp_path / "facts.db")
    with pytest.raises(RuntimeError):
        with request_context("u", "s"), logged_span(store, "request", "test"):
            TraceRecorder().run("routing", "food_query_classifier",
                                lambda: ClassificationResult(True, 0.93, 0.1),
                                lambda value: "related", inputs={"text": "土豆怎么做"})
            raise RuntimeError("later failure")
    record, = queries(store.path)
    assert record["result"]["status"] == "failed"
    step, = record["steps"]
    assert step["content"] == {"is_related": True, "positive_score": 0.93, "threshold": 0.1}
    assert step["prompt"] == {"text": "土豆怎么做"} and step["status"] == "success"


def test_uploaded_image_is_durable_and_deduplicated(tmp_path):
    store = LLMLogStore(tmp_path / "image.db")
    image = b"image input bytes"
    reference = store.capture_image(image)
    assert store.capture_image(image) == reference
    assert Path(reference["path"]).read_bytes() == image
    assert reference["size_bytes"] == len(image)
    assert len(list(Path(reference["path"]).parent.iterdir())) == 1


def test_profile_summarizer_keeps_model_details_in_one_query(tmp_path, monkeypatch):
    from app.pipeline import build_pipeline

    store = LLMLogStore(tmp_path / "summary.db")
    callback = PersistentLLMCallback(store, "kitchen_agent", "m")
    fake_agent = SimpleNamespace(
        model=StructuredToolModel(responses=[AIMessage(content='{"candidates": []}')]),
        callbacks=[callback],
    )
    monkeypatch.setattr("app.pipeline.build_agent", lambda *args: fake_agent)
    settings = replace(get_settings(), deepseek_api_key="offline",
                       inventory_db_path=tmp_path / "inventory.db",
                       user_profile_path=tmp_path / "profile.md",
                       food_classifier_enabled=False, recommender_enabled=False)
    pipeline = build_pipeline(settings)
    for index in range(10):
        pipeline.history.append(user_id="u", request=f"request-{index}", response="response")
    assert pipeline.profile_summarizer.run_if_due("u") == []
    record, = queries(store.path)
    calls = [s for s in record["steps"] if s["kind"] == "llm"]
    assert len(calls) == 1 and calls[0]["name"] == "profile_summary"
    assert "request-0" in json.dumps(calls[0]["prompt"])


def test_repair_and_retry_components_share_the_same_query(tmp_path):
    from app.agent import KitchenAgent
    from app.audit import AuditCode, AuditResult
    from app.context import AgentContext
    from app.supervisor import RetryMode, SupervisorRequest
    from app.trace import TraceRecorder

    store = LLMLogStore(tmp_path / "roles.db")
    model = StructuredToolModel(responses=[
        AIMessage(content='{"action":"fallback","reason_code":"task_impossible"}'),
        AIMessage(content="重写回答"), AIMessage(content="菜谱恢复回答"),
    ])
    agent = KitchenAgent(model=model, inventory=SimpleNamespace(), recipes=SimpleNamespace(),
                         actions=SimpleNamespace(),
                         logging_callbacks=[PersistentLLMCallback(store, "kitchen_agent", "m")])
    with request_context("u", "s"), logged_span(store, "request", "test"):
        agent._run_repair_subagent(SupervisorRequest(content="old"),
                                  AuditResult(False, "failure", AuditCode.TASK_IMPOSSIBLE))
        for mode in (RetryMode.ANSWER_REWRITE, RetryMode.RECIPE_RECOVERY):
            agent._run_main_retry(context=AgentContext(user_id="u", request_text="做什么菜"),
                                  target_language="zh", mode=mode, feedback="retry",
                                  immutable_history=[], trace=TraceRecorder())
    record, = queries(store.path)
    calls = [s for s in record["steps"] if s["kind"] == "llm"]
    assert [s["name"] for s in calls] == ["response_repair", "answer_rewrite", "recipe_recovery"]


def test_deepseek_search_pause_turn_is_one_query(tmp_path, monkeypatch):
    from app.adapters.recipe_api import DeepSeekWebSearchClient

    settings = replace(get_settings(), llm_log_db_path=tmp_path / "http.db")
    client = DeepSeekWebSearchClient("https://example.invalid", "private-key", "model",
                                    logging_settings=settings)
    responses = iter([
        {"stop_reason": "pause_turn", "content": [{"type": "server_tool_use", "id": "web-1"}],
         "usage": {"input_tokens": 3, "output_tokens": 2}},
        {"stop_reason": "end_turn", "content": [{"type": "text", "text": '{"recipes":[]}'}],
         "usage": {"input_tokens": 5, "output_tokens": 4}},
    ])

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return next(responses)

    monkeypatch.setattr("app.adapters.recipe_api.httpx.post", lambda *args, **kwargs: Response())
    assert client.search(ingredients=["tomato"], intolerances=[], exclude_ingredients=[]) == []
    record, = queries(settings.llm_log_db_path)
    assert len(record["steps"]) == 2 and record["tokens"]["total_tokens"] == 14
    assert record["steps"][0]["content"]["stop_reason"] == "pause_turn"
    assert "private-key" not in json.dumps(record)


def test_real_pipeline_query_produces_one_row_including_routing(tmp_path):
    from app.pipeline import build_pipeline

    settings = replace(get_settings(), deepseek_api_key="",
                       inventory_db_path=tmp_path / "inventory.db",
                       user_profile_path=tmp_path / "profile.md",
                       food_classifier_enabled=False, recommender_enabled=False)
    pipeline = build_pipeline(settings)
    result = pipeline.process_request(user_id="u", text="查看库存")
    record, = queries(settings.llm_log_db_path)
    assert record["segment_id"] == result.segment_id
    assert record["request"]["text"] == "查看库存"
    assert record["result"]["content"] == result.content
    assert any(s["name"] == "deterministic_route" for s in record["steps"])
    assert record["tokens"]["model_calls"] == 0
