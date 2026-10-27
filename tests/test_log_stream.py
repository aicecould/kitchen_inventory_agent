import asyncio
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.config import get_settings
from app.context import AgentResult
from app.llm_log import LLMLogStore, logged_span, request_context, query_scope
from app.log_stream import QueryLogReader, query_stream


def test_reader_emits_only_current_query_changed_steps(tmp_path):
    store = LLMLogStore(tmp_path / "logs.db")
    with query_scope("current"), request_context("user", "segment"), logged_span(store, "request", "query"):
        reader = QueryLogReader(store.path, "current", "user")
        other_user = QueryLogReader(store.path, "current", "another-user")
        assert other_user.poll() is None
        assert reader.poll()["status"] == "running"
        assert reader.poll() is None
        with logged_span(store, "component", "first") as span:
            update = reader.poll()
            assert update["steps"][0]["status"] == "running"
            span["content"] = {"answer": "<script>unsafe</script>"}
        assert reader.poll()["steps"][0]["content"] == {"answer": "<script>unsafe</script>"}
        assert reader.poll() is None
        with request_context("other", "other-segment"), logged_span(LLMLogStore(tmp_path / "other.db"), "request", "other"):
            pass
        # A different query in the same DB must not leak or resend steps.
        with query_scope("other"), request_context("other", "other-segment"), logged_span(store, "request", "other"):
            pass
        assert reader.poll() is None
    assert reader.poll()["status"] == "success"
    reader.close()
    other_user.close()


def test_stream_delivers_running_step_before_operation_finishes(tmp_path):
    async def scenario():
        store = LLMLogStore(tmp_path / "logs.db")
        release = asyncio.Event()

        async def operation():
            with request_context("user", "segment"), logged_span(store, "request", "query"):
                with logged_span(store, "component", "slow-step") as span:
                    await release.wait()
                    span["content"] = "done"
            return {"content": "answer"}

        response = query_stream(operation, SimpleNamespace(llm_log_db_path=store.path, llm_logging_enabled=True), "user")
        frames = []
        async for frame in response.body_iterator:
            frames.append(frame)
            if '"name": "slow-step"' in frame and '"status": "running"' in frame:
                release.set()
        text = "".join(frames)
        assert release.is_set()
        assert '"content": "done"' in text
        assert text.index('"content": "done"') < text.index("event: result")
        assert '"status": "success"' in text

    asyncio.run(asyncio.wait_for(scenario(), timeout=5))


def test_stream_http_result_disabled_and_error(monkeypatch, tmp_path):
    from app import web
    settings = replace(get_settings(), llm_log_db_path=tmp_path / "logs.db", llm_logging_enabled=False)
    monkeypatch.setattr(web, "get_settings", lambda: settings)
    monkeypatch.setattr(web, "get_pipeline", lambda: SimpleNamespace(process_request=lambda **kwargs: AgentResult(content="answer")))
    client = TestClient(web.app)
    response = client.post("/api/process/stream", data={"text": "查看库存"})
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "日志记录已关闭" in response.text
    assert '"content": "answer"' in response.text
    response = client.post("/api/process/stream", data={"text": "x" * 100_000})
    assert "event: error" in response.text
    assert "event: result" not in response.text
    assert not settings.llm_log_db_path.exists()


def test_browser_parser_handles_split_unicode_frames_and_connection_loss():
    script = r'''
const { readQueryStream } = require('./frontend/logs.js');
const assert = require('node:assert/strict');
const bytes = new TextEncoder().encode('event: steps\ndata: {"steps":[{"name":"中文"}]}\n\nevent: result\ndata: {"content":"完成"}\n\n');
const make = (data) => ({ok:true, body:new ReadableStream({start(c) {for (const byte of data) c.enqueue(new Uint8Array([byte])); c.close();}})});
(async () => {
  const events = [];
  const result = await readQueryStream(make(bytes), (...args) => events.push(args));
  assert.equal(result.content, '完成');
  assert.equal(events[0][1].steps[0].name, '中文');
  await assert.rejects(() => readQueryStream(make(bytes.slice(0, 15)), () => {}), /连接已中断/);
  const err = new TextEncoder().encode('event: error\ndata: {"detail":"failed"}\n\n');
  await assert.rejects(() => readQueryStream(make(err), () => {}), /failed/);
})().catch(e => { console.error(e); process.exitCode = 1; });
'''
    subprocess.run(["node", "-e", script], cwd=Path(__file__).resolve().parents[1], check=True)


def test_http_stream_reads_the_pipeline_query_database(monkeypatch, tmp_path):
    from app import web
    from app.pipeline import build_pipeline

    settings = replace(get_settings(), llm_log_db_path=tmp_path / "logs.db",
                       inventory_db_path=tmp_path / "inventory.db",
                       user_profile_path=tmp_path / "profile.md", deepseek_api_key="",
                       food_classifier_enabled=False, recommender_enabled=False)
    pipeline = build_pipeline(settings)
    monkeypatch.setattr(web, "get_settings", lambda: settings)
    monkeypatch.setattr(web, "get_pipeline", lambda: pipeline)
    response = TestClient(web.app).post("/api/process/stream", data={"text": "查看库存"})
    frames = [frame for frame in response.text.split("\n\n") if frame.startswith("event:")]
    updates = [json.loads(frame.split("data: ", 1)[1]) for frame in frames if frame.startswith("event: steps")]
    result = json.loads(next(frame for frame in frames if frame.startswith("event: result")).split("data: ", 1)[1])
    assert updates[-1]["status"] == "success"
    assert len({update["query_id"] for update in updates}) == 1
    assert any(step["name"] == "deterministic_route" for update in updates for step in update["steps"])
    import sqlite3
    with sqlite3.connect(settings.llm_log_db_path) as db:
        record, = db.execute("SELECT record_json FROM query_logs").fetchone()
    assert json.loads(record)["segment_id"] == result["segment_id"]


def test_browser_log_view_keeps_step_nodes_and_lazily_displays_full_content():
    script = r'''
const { createLogView } = require('./frontend/logs.js');
const assert = require('node:assert/strict');
const element = () => ({children: [], textContent: '', open: false,
  append(...nodes) {this.children.push(...nodes);}, replaceChildren() {this.children = [];}});
global.document = {createElement: element};
const container = element(), status = element();
const view = createLogView(container, status);
view.reset();
const step = {id: 'step', kind: 'llm', name: '<script>x</script>', status: 'running', prompt: 'x'.repeat(10000), time: {}};
view.update('steps', {query_id:'q', status:'running', steps:[step]});
const card = container.children[0];
assert.equal(card.children[1].textContent, '');
assert.ok(card.children[0].textContent.includes("模型调用"));
assert.ok(!card.children[0].textContent.includes(step.name));
card.open = true; card.ontoggle();
assert.ok(card.children[1].textContent.includes(step.prompt));
view.update('steps', {query_id:'q', status:'success', steps:[{...step, status:'success', content:'answer'}]});
assert.equal(container.children.length, 1);
assert.equal(card.open, true);
assert.ok(card.children[1].textContent.includes('answer'));
assert.ok(card.children[1].textContent.includes(step.name));
view.update('steps', {query_id:'q', status:'running', steps:[{...step, name:'kitchen_agent', status:'failed', error_type:'ValidationError'}]});
assert.ok(card.children[0].textContent.includes('主助手生成回答'));
assert.ok(card.children[0].textContent.includes('失败：返回数据校验未通过'));
assert.ok(status.textContent.includes('1 步失败'));
view.update('steps', {query_id:'q', status:'success', steps:[{...step, kind:'event', name:'qwen_agent', status:'success', content:{status:'failed', detail:'OutputParserException', duration_ms:12}}]});
assert.ok(card.children[0].textContent.includes('本地模型生成回答'));
assert.ok(card.children[0].textContent.includes('失败：模型返回内容解析失败'));
assert.ok(card.children[0].textContent.includes('12 毫秒'));
assert.equal(card.className, 'trace-entry trace-failed');
assert.ok(!card.children[0].textContent.includes('OutputParserException'));
view.reset();
assert.equal(container.children.length, 0);
'''
    subprocess.run(["node", "-e", script], cwd=Path(__file__).resolve().parents[1], check=True)


def test_stream_reads_uploaded_image_during_response(monkeypatch, tmp_path):
    from io import BytesIO
    from PIL import Image
    from app import web

    settings = replace(get_settings(), llm_log_db_path=tmp_path / "logs.db", llm_logging_enabled=False)
    monkeypatch.setattr(web, "get_settings", lambda: settings)
    image = BytesIO()
    Image.new("RGB", (16, 16)).save(image, format="PNG")
    seen = []

    def process(**kwargs):
        seen.append(kwargs["image_bytes"])
        return AgentResult(content="image accepted")

    monkeypatch.setattr(web, "get_pipeline", lambda: SimpleNamespace(process_request=process))
    response = TestClient(web.app).post("/api/process/stream", data={"text": "图片"},
                                      files={"image": ("image.png", image.getvalue(), "image/png")})
    assert "event: result" in response.text
    assert seen == [image.getvalue()]


def test_vision_continuation_stream_has_its_own_query(monkeypatch, tmp_path):
    from app import web

    settings = replace(get_settings(), llm_log_db_path=tmp_path / "logs.db")
    monkeypatch.setattr(web, "get_settings", lambda: settings)
    store = LLMLogStore(settings.llm_log_db_path)

    def resolve(**kwargs):
        with logged_span(store, "component", "continue-image") as span:
            span["content"] = {"selections": kwargs["selections"]}
        return AgentResult(content="confirmed")

    monkeypatch.setattr(web, "get_pipeline", lambda: SimpleNamespace(resolve_vision_confirmation=resolve))
    response = TestClient(web.app).post("/api/vision-confirmations/confirmation-id/resolve/stream",
                                      json={"selections": []})
    assert '"name": "continue-image"' in response.text
    assert '"content": "confirmed"' in response.text
    assert '"status": "success"' in response.text
