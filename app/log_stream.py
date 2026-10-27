"""Stream current-query steps from committed SQLite records, without model work."""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from app.llm_log import query_scope

logger = logging.getLogger(__name__)


class QueryLogReader:
    """One read-only connection; data_version avoids reloading unchanged JSON."""

    def __init__(self, path: Path, query_id: str, user_id: str):
        self.path, self.query_id, self.user_id = path, query_id, user_id
        self.db = None
        self.version = None
        self.steps: dict[str, dict[str, Any]] = {}
        self.summary = None

    def poll(self, *, force: bool = False) -> dict[str, Any] | None:
        if self.db is None:
            if not self.path.is_file():
                return None
            self.db = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro",
                                      uri=True, timeout=0.1, check_same_thread=False)
        version = self.db.execute("PRAGMA data_version").fetchone()[0]
        if not force and version == self.version:
            return None
        self.version = version
        row = self.db.execute(
            "SELECT record_json FROM query_logs WHERE id=? AND user_id=?",
            (self.query_id, self.user_id),
        ).fetchone()
        if row is None:
            return None
        record = json.loads(row[0])
        summary = {"query_id": record["id"], "status": record["result"]["status"],
                   "time": record["time"], "tokens": record["tokens"]}
        changed = [step for step in record["steps"] if self.steps.get(step["id"]) != step]
        if not changed and summary == self.summary:
            return None
        self.steps = {step["id"]: step for step in record["steps"]}
        self.summary = summary
        return {**summary, "steps": changed}

    def close(self):
        if self.db is not None:
            self.db.close()


def query_stream(operation, settings, user_id: str) -> StreamingResponse:
    query_id = str(uuid4())

    async def execute():
        with query_scope(query_id):
            return await operation()

    def event(kind, data):
        return f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    async def events():
        reader = QueryLogReader(settings.llm_log_db_path, query_id, user_id)
        task = asyncio.create_task(execute())
        heartbeat = 0
        try:
            yield event("log_status", {"message": "等待本轮日志写入…" if settings.llm_logging_enabled
                                      else "日志记录已关闭。"})
            while True:
                done = task.done()
                if settings.llm_logging_enabled:
                    try:
                        update = await run_in_threadpool(reader.poll, force=done)
                        if update:
                            yield event("steps", update)
                    except (sqlite3.Error, ValueError, KeyError):
                        logger.warning("Current-query log read failed")
                        yield event("log_status", {"message": "日志暂时无法读取。"})
                if done:
                    break
                # Heartbeat also keeps proxy connections alive during slow steps.
                if heartbeat % 60 == 0:
                    yield ": heartbeat\n\n"
                heartbeat += 1
                await asyncio.sleep(0.25)
            if settings.llm_logging_enabled and reader.summary is None:
                yield event("log_status", {"message": "本轮暂无可读取的数据库日志。"})
            try:
                yield event("result", task.result())
            except HTTPException as exc:
                yield event("error", {"detail": exc.detail})
            except Exception:
                yield event("error", {"detail": "请求处理失败，请检查服务日志。"})
        finally:
            # A disconnected browser must not interrupt an inventory mutation.
            # Retrieve background errors even when there is no longer a subscriber.
            task.add_done_callback(lambda future: future.exception() if not future.cancelled() else None)
            await run_in_threadpool(reader.close)

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
