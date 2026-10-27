"""One durable JSON document per Query, with embedded component and model details."""
from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
import logging
import sqlite3
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Iterator
from uuid import UUID, uuid4
from urllib.parse import urlsplit

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)
_context: ContextVar[dict[str, Any] | None] = ContextVar("llm_log_context", default=None)
_SECRET_KEYS = {"api_key", "authorization", "x-api-key", "password", "secret", "access_token"}


def json_value(value: Any) -> Any:
    """Keep structured message content; never serialize arbitrary objects via repr."""
    if is_dataclass(value) and not isinstance(value, type):
        return json_value(asdict(value))
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return json_value(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(k): "[REDACTED]" if str(k).lower() in _SECRET_KEYS else json_value(v)
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"type": type(value).__name__}


def _json(value: Any) -> str:
    return json.dumps(json_value(value), ensure_ascii=False)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def fingerprint(value: Any) -> str:
    payload = json.dumps(json_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def runtime_metadata() -> dict[str, Any]:
    """Identify the running source, including local edits, without invoking Git."""
    root = Path(__file__).resolve().parent.parent
    files = {}
    for folder in ("app", "classifier", "recommender", "llm", "evaluation"):
        for path in sorted((root / folder).rglob("*.py")):
            files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    packages = {}
    for name in ("langchain", "langchain-core", "langchain-openai", "pydantic"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"code_version": fingerprint(files), "source_files": files,
            "python": platform.python_version(), "packages": packages}


@lru_cache(maxsize=32)
def _file_digest(path: str, mtime_ns: int, size: int) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def settings_metadata(settings: Any) -> dict[str, Any]:
    # Never serialize Settings wholesale: it also contains provider credentials.
    names = ("application_timezone", "http_timeout_seconds", "agent_recursion_limit",
             "deepseek_model", "deepseek_max_output_tokens", "deepseek_max_input_chars",
             "pending_action_ttl_minutes", "food_classifier_enabled", "recommender_enabled",
             "recommender_model_revision", "llm_provider", "llm_model")
    config = {name: getattr(settings, name) for name in names}
    artifacts = {}
    for name in ("food_classifier_model_path", "food_classifier_metadata_path", "recommender_metadata_path"):
        path = Path(getattr(settings, name))
        digest = None
        if path.is_file():
            stat = path.stat()
            digest = _file_digest(str(path.resolve()), stat.st_mtime_ns, stat.st_size)
        artifacts[name] = {"path": str(path), "sha256": digest}
    endpoint = urlsplit(settings.llm_base_url)
    return {"config": config, "artifacts": artifacts,
            "provider_endpoint": {"scheme": endpoint.scheme, "host": endpoint.hostname,
                                  "path": endpoint.path}}


def active_store() -> LLMLogStore | None:
    return (_context.get() or {}).get("store")


class LLMLogStore:
    """Exactly one SQLite row per Query; all execution details live in its JSON."""

    def __init__(self, path: Path | str, *, source: str = "application",
                 metadata: dict[str, Any] | None = None) -> None:
        self.path = Path(path)
        self.source = source
        self.metadata = metadata or {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS query_logs (
                id TEXT PRIMARY KEY, user_id TEXT, segment_id TEXT,
                source TEXT NOT NULL, status TEXT NOT NULL,
                started_at TEXT NOT NULL, ended_at TEXT,
                record_json TEXT NOT NULL
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS query_logs_time ON query_logs(started_at)")
            db.execute("CREATE INDEX IF NOT EXISTS query_logs_segment ON query_logs(segment_id)")
            db.execute("CREATE INDEX IF NOT EXISTS query_logs_source ON query_logs(source, status)")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def capture_image(self, data: bytes | None) -> dict[str, Any] | None:
        if data is None:
            return None
        digest = hashlib.sha256(data).hexdigest()
        path = self.path.resolve().with_suffix(".assets") / f"{digest}.bin"

        def save() -> dict[str, Any]:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                temporary = path.with_suffix(f".{uuid4().hex}.tmp")
                try:
                    temporary.write_bytes(data)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
            return {"sha256": digest, "size_bytes": len(data), "path": str(path)}

        return _safe(save)

    def begin_query(self, name: str, prompt: Any, *,
                    metadata: dict[str, Any] | None = None) -> str:
        context = _context.get() or {}
        query_id = context.get("trace_id") or str(uuid4())
        info = {**context.get("metadata", {}), **(metadata or {})}
        source = info.pop("source", self.source)
        record = {
            "id": query_id, "schema_version": 2, "name": name, "source": source,
            "user_id": context.get("user_id") or info.pop("user_id", None),
            "segment_id": context.get("segment_id") or info.pop("segment_id", None),
            "request": json_value(prompt), "prompt_hash": fingerprint(prompt),
            "runtime": json_value({**self.metadata, **runtime_metadata()}),
            "metadata": json_value(info), "steps": [], "models": [],
            "result": {"status": "running", "content": None, "error_type": None},
            "time": {"started_at": _now(), "ended_at": None, "duration_ms": None},
            "tokens": _total_usage([]),
        }
        with self.connect() as db:
            db.execute("""INSERT INTO query_logs
                (id, user_id, segment_id, source, status, started_at, record_json)
                VALUES (?, ?, ?, ?, 'running', ?, ?)""",
                (query_id, record["user_id"], record["segment_id"], source,
                 record["time"]["started_at"], _json(record)))
        return query_id

    def _update(self, query_id: str | None, operation: Any) -> None:
        if query_id is None:
            return
        with self.connect() as db:
            # Serialize read-modify-write, including callbacks from parallel tools.
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT record_json FROM query_logs WHERE id=?", (query_id,)).fetchone()
            if row is None:
                return
            record = json.loads(row[0])
            operation(record)
            calls = [step for step in record["steps"] if step["kind"] == "llm"]
            record["tokens"] = _total_usage(calls)
            record["models"] = sorted({step["model"] for step in calls if step.get("model")})
            db.execute("""UPDATE query_logs SET status=?, ended_at=?, record_json=? WHERE id=?""",
                       (record["result"]["status"], record["time"]["ended_at"], _json(record), query_id))

    def start(self, kind: str, name: str, *, id: str | None = None,
              parent_id: str | None = None, model: str | None = None,
              prompt: Any = None, parameters: Any = None,
              metadata: dict[str, Any] | None = None) -> str | None:
        context = _context.get() or {}
        query_id = context.get("query_id")
        if query_id is None:
            return None
        step_id = id or str(uuid4())
        step = {
            "id": step_id, "parent_id": parent_id or context.get("parent_id") or query_id,
            "kind": kind, "name": name, "model": model,
            "prompt": json_value(prompt), "prompt_hash": fingerprint(prompt),
            "parameters": json_value(parameters), "metadata": json_value(metadata or {}),
            "content": None, "tool_calls": [], "usage": None, "tokens": _parse_usage(None),
            "status": "running", "error_type": None,
            "time": {"started_at": _now(), "ended_at": None, "duration_ms": None},
        }
        if kind == "llm":
            step["model_identity"] = model_identity(
                model, (parameters or {}).get("model") or (parameters or {}).get("model_name"), None, None
            )
            step["model"] = None

        def append(record: dict[str, Any]) -> None:
            # A nested model may receive both inherited and explicit callbacks.
            if not any(existing["id"] == step_id for existing in record["steps"]):
                record["steps"].append(step)

        self._update(query_id, append)
        return step_id

    def finish(self, step_id: str | None, *, content: Any = None, tool_calls: Any = None,
               usage: dict[str, Any] | None = None, model: str | None = None,
               identity: dict[str, Any] | None = None,
               error: BaseException | None = None, query_id: str | None = None) -> None:
        if step_id is None:
            return

        def update(record: dict[str, Any]) -> None:
            step = next((step for step in record["steps"] if step["id"] == step_id), None)
            if step is None:
                return
            step.update(content=json_value(content), tool_calls=json_value(tool_calls or []),
                        usage=json_value(usage), tokens=_parse_usage(usage),
                        status="failed" if error else "success",
                        error_type=type(error).__name__ if error else None)
            if model:
                step["model"] = model
            if step["kind"] == "llm":
                step["model_identity"] = identity or model_identity(
                    step["model_identity"]["configured_model"],
                    step["model_identity"]["requested_model"], model,
                    "response_body" if model else None,
                )
                step["model"] = step["model_identity"]["response_model"]
            _end_time(step["time"])

        self._update(query_id or (_context.get() or {}).get("query_id"), update)

    def finish_query(self, query_id: str | None, *, content: Any = None,
                     error: BaseException | None = None) -> None:
        def update(record: dict[str, Any]) -> None:
            payload = json_value(content)
            record["result"] = {
                **(payload if isinstance(payload, dict) else {"content": payload}),
                "status": "failed" if error else "success",
                "error_type": type(error).__name__ if error else None,
            }
            _end_time(record["time"])

        self._update(query_id, update)


def _parse_usage(usage: dict[str, Any] | None) -> dict[str, int | None]:
    usage = usage or {}
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    total = usage.get("total_tokens")
    if total is None and isinstance(input_tokens, int) and isinstance(output_tokens, int):
        total = input_tokens + output_tokens
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total}


def model_identity(configured: str | None, requested: str | None,
                   reported: str | None, source: str | None) -> dict[str, Any]:
    return {"configured_model": configured, "requested_model": requested,
            "response_model": reported, "source": source or "unknown",
            "independently_verified": False}


def response_model_identity(configured: str | None, requested: str | None,
                            messages: list[Any], output: dict[str, Any]) -> dict[str, Any]:
    # ChatOpenAI may fill llm_output.model_name from the request. Message response
    # metadata carries the per-response model, and therefore takes precedence.
    reported = []
    stamped = False
    for message in messages:
        metadata = (message.get("response_metadata", {}) if isinstance(message, dict)
                    else getattr(message, "response_metadata", {}))
        evidence = metadata.get("model_identity_response")
        if isinstance(evidence, dict):
            stamped = True
            name = evidence.get("model")
        else:
            name = metadata.get("model_name") or metadata.get("model")
        if isinstance(name, str) and name.strip():
            reported.append(name)
    unique = sorted(set(reported))
    identity = model_identity(configured, requested,
                              unique[0] if len(unique) == 1 else None,
                              ("response_body" if stamped else "message_response_metadata") if len(unique) == 1 else None)
    identity["message_models"] = unique
    identity["sdk_model"] = output.get("model_name") or output.get("model")
    if len(unique) > 1:
        identity["source"] = "conflicting_response_models"
    return identity


def _total_usage(calls: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"model_calls": len(calls)}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        values = [call["tokens"][key] for call in calls if call["tokens"][key] is not None]
        result[key] = sum(values) if values or not calls else None
    result["calls_with_usage"] = sum(call["tokens"]["total_tokens"] is not None for call in calls)
    result["complete"] = all(
        all(value is not None for value in call["tokens"].values()) for call in calls
    )
    return result


def _end_time(time: dict[str, Any]) -> None:
    time["ended_at"] = _now()
    time["duration_ms"] = max(0, (
        datetime.fromisoformat(time["ended_at"]) - datetime.fromisoformat(time["started_at"])
    ).total_seconds() * 1000)


def callback_store(callbacks: list[Any] | None) -> LLMLogStore | None:
    store = active_store()
    if store:
        return store
    return next((callback.store for callback in callbacks or []
                 if isinstance(callback, PersistentLLMCallback)), None)


@contextmanager
def logged_span(store: LLMLogStore | None, kind: str, name: str,
                **kwargs: Any) -> Iterator[dict[str, Any]]:
    """Update a Query's embedded details; only the outer scope inserts a row."""
    result: dict[str, Any] = {}
    context = dict(_context.get() or {})
    store = active_store() or store
    if store is None:
        yield result
        return
    owns_query = context.get("query_id") is None
    query_id = context.get("query_id")
    if owns_query:
        query_id = _safe(lambda: store.begin_query(name, kwargs.get("prompt"),
                                                  metadata=kwargs.get("metadata")))
    context.update(query_id=query_id, store=store)
    token = _context.set(context)
    step_id = None
    if kind != "request":
        step_id = _safe(lambda: store.start(kind, name, **kwargs))
        context = {**context, "parent_id": step_id}
        _context.set(context)
    try:
        yield result
    except BaseException as exc:
        if step_id:
            _safe(lambda: store.finish(step_id, error=exc))
        if owns_query:
            _safe(lambda: store.finish_query(query_id, error=exc))
        raise
    else:
        if step_id:
            _safe(lambda: store.finish(step_id, **result))
        if owns_query:
            _safe(lambda: store.finish_query(query_id, content=result.get("content")))
    finally:
        _context.reset(token)


@contextmanager
def query_scope(query_id: str) -> Iterator[None]:
    """Associate a web stream with its durable query across worker threads."""
    token = _context.set({**(_context.get() or {}), "trace_id": query_id})
    try:
        yield
    finally:
        _context.reset(token)


@contextmanager
def request_context(user_id: str, segment_id: str, *,
                    metadata: dict[str, Any] | None = None) -> Iterator[None]:
    token = _context.set({"trace_id": (_context.get() or {}).get("trace_id") or str(uuid4()), "user_id": user_id,
                          "segment_id": segment_id,
                          "metadata": {**(_context.get() or {}).get("metadata", {}), **(metadata or {})}})
    try:
        yield
    finally:
        _context.reset(token)


@contextmanager
def logging_context(**metadata: Any) -> Iterator[None]:
    """Carry observation provenance without changing application behavior."""
    context = dict(_context.get() or {})
    context["metadata"] = {**context.get("metadata", {}), **metadata}
    token = _context.set(context)
    try:
        yield
    finally:
        _context.reset(token)


def _safe(operation: Any) -> Any:
    try:
        return operation()
    except Exception as exc:
        # Avoid exception messages: provider errors may embed credentials or payloads.
        logger.error("LLM log persistence failed (%s)", type(exc).__name__)
        return None


class PersistentLLMCallback(BaseCallbackHandler):
    run_inline = True

    def __init__(self, store: LLMLogStore, component: str, model: str) -> None:
        self.store, self.component, self.model = store, component, model
        self._runs: dict[str, tuple[LLMLogStore, str]] = {}
        self._requested: dict[str, str | None] = {}

    def _start(self, kind: str, serialized: Any, run_id: Any, parent_run_id: Any,
               prompt: Any = None, parameters: Any = None,
               metadata: dict[str, Any] | None = None) -> None:
        context = _context.get() or {}
        if not context.get("query_id"):
            return
        store = active_store() or self.store
        self._runs[str(run_id)] = (store, context["query_id"])
        name = (serialized or {}).get("name") or self.component
        _safe(lambda: store.start(kind, name, id=str(run_id),
            parent_id=context.get("parent_id"),
            model=self.model if kind == "llm" else None, prompt=prompt,
            parameters=parameters, metadata=metadata))

    def _finish(self, run_id: Any, **kwargs: Any) -> None:
        run = self._runs.pop(str(run_id), None)
        if run:
            store, query_id = run
            _safe(lambda: store.finish(str(run_id), query_id=query_id, **kwargs))

    def on_chat_model_start(self, serialized: Any, messages: Any, *, run_id: Any,
                            parent_run_id: Any = None, **kwargs: Any) -> None:
        params = kwargs.get("invocation_params", {})
        self._requested[str(run_id)] = params.get("model") or params.get("model_name")
        # Explicit allowlist excludes client objects, headers and credentials.
        params = {k: v for k, v in params.items() if k in {
            "model", "model_name", "temperature", "max_tokens", "stop", "tools",
            "tool_choice", "response_format", "extra_body", "thinking"}}
        metadata = kwargs.get("metadata") or {}
        system_prompts = [getattr(message, "content", None)
                          for group in messages for message in group
                          if getattr(message, "type", None) == "system"]
        component = metadata.get("log_component", self.component)
        self._start("llm", {"name": component}, run_id, parent_run_id, messages, params,
                    metadata={**metadata.get("log_metadata", {}), "component": component,
                              "system_prompt_hash": fingerprint(system_prompts),
                              "parameters_hash": fingerprint(params)})

    def on_llm_end(self, response: Any, *, run_id: Any, **kwargs: Any) -> None:
        messages = [g.message for group in response.generations for g in group if hasattr(g, "message")]
        output = response.llm_output or {}
        usage = output.get("token_usage") or output.get("usage")
        if not usage and messages:
            usage = getattr(messages[0], "usage_metadata", None)
            metadata = getattr(messages[0], "response_metadata", {})
            usage = usage or metadata.get("token_usage") or metadata.get("usage")
        calls = [call for m in messages for call in (getattr(m, "tool_calls", None) or [])]
        identity = response_model_identity(
            self.model, self._requested.pop(str(run_id), None), messages, output
        )
        self._finish(run_id, content=messages, tool_calls=calls,
                     usage=usage, identity=identity)

    def on_llm_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        self._requested.pop(str(run_id), None)
        self._finish(run_id, error=error)

    def on_tool_start(self, serialized: Any, input_str: str, *, run_id: Any,
                      parent_run_id: Any = None, **kwargs: Any) -> None:
        self._start("tool", serialized, run_id, parent_run_id, kwargs.get("inputs") or input_str)

    def on_tool_end(self, output: Any, *, run_id: Any, **kwargs: Any) -> None:
        self._finish(run_id, content=output)

    def on_tool_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        self._finish(run_id, error=error)


def log_store(settings: Any) -> LLMLogStore | None:
    if not settings.llm_logging_enabled:
        return None
    return _safe(lambda: LLMLogStore(settings.llm_log_db_path,
        source=settings.llm_log_source, metadata=settings_metadata(settings)))


def model_callbacks(settings: Any, component: str) -> list[Any]:
    store = log_store(settings)
    return [PersistentLLMCallback(store, component, settings.llm_model)] if store else []
