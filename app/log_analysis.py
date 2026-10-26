"""Read-only Query log inspection and JSONL export: python -m app.log_analysis."""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

from app.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=get_settings().llm_log_db_path)
    parser.add_argument("--query-id", "--trace-id", dest="query_id")
    parser.add_argument("--segment-id")
    parser.add_argument("--model")
    parser.add_argument("--source")
    parser.add_argument("--name")
    parser.add_argument("--kind", choices=["llm", "tool", "component", "event"])
    parser.add_argument("--status", choices=["running", "success", "failed"])
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    if not args.db.is_file():
        parser.error(f"Log database does not exist: {args.db}")
    filters = {key: getattr(args, key) for key in ("segment_id", "source", "status")
               if getattr(args, key) is not None}
    if args.query_id:
        filters["id"] = args.query_id
    where = " WHERE " + " AND ".join(f"{key}=?" for key in filters) if filters else ""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    exported = 0
    with sqlite3.connect(args.db.resolve().as_uri() + "?mode=ro", uri=True) as db:
        for row in db.execute("SELECT record_json FROM query_logs" + where +
                              " ORDER BY started_at DESC", tuple(filters.values())):
            record = json.loads(row[0])
            matching_steps = [step for step in record["steps"] if
                              (not args.kind or step["kind"] == args.kind) and
                              (not args.model or step.get("model") == args.model) and
                              (not args.name or step["name"] == args.name)]
            if (args.kind or args.model or args.name) and not matching_steps:
                if args.kind or args.model or record["name"] != args.name:
                    continue
            if args.summary:
                groups[(record["source"], record["result"]["status"])].append(record)
            else:
                # Always emit the entire Query, including when a step filter matches.
                print(json.dumps(record, ensure_ascii=False))
                exported += 1
                if exported >= args.limit:
                    break
    for (source, status), records in list(groups.items())[:args.limit]:
        summary = {"source": source, "status": status, "queries": len(records),
                   "model_calls": sum(r["tokens"]["model_calls"] for r in records),
                   "calls_with_usage": sum(r["tokens"]["calls_with_usage"] for r in records),
                   "usage_complete": all(r["tokens"]["complete"] for r in records)}
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            values = [r["tokens"][key] for r in records if r["tokens"][key] is not None]
            summary[key] = sum(values) if values else None
        durations = [r["time"]["duration_ms"] for r in records if r["time"]["duration_ms"] is not None]
        summary["avg_duration_ms"] = sum(durations) / len(durations) if durations else None
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
