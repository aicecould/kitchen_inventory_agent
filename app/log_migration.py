"""Back up and correct legacy model identities using preserved response evidence."""

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3

from app.llm_log import model_identity, response_model_identity


def migrate(path: Path) -> tuple[Path, int]:
    backup = path.with_name(f"{path.stem}.before-model-identity-{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}.db")
    count = 0
    with sqlite3.connect(path) as db:
        with sqlite3.connect(backup) as target:
            db.backup(target)
        db.execute("BEGIN IMMEDIATE")
        for query_id, raw in db.execute("SELECT id, record_json FROM query_logs").fetchall():
            record = json.loads(raw)
            if record.get("schema_version", 1) >= 2:
                continue
            config = record.get("runtime", {}).get("config", {})
            for step in record["steps"]:
                if step["kind"] != "llm":
                    continue
                params = step.get("parameters") or {}
                requested = params.get("model") or params.get("model_name")
                configured = config.get("llm_model")
                configured = configured or (config.get("deepseek_model") if config.get("llm_provider", "deepseek") == "deepseek" else None)
                content = step.get("content")
                identity = response_model_identity(configured, requested, content if isinstance(content, list) else [], {})
                if isinstance(content, dict) and isinstance(content.get("model"), str):
                    identity = model_identity(configured, requested, content["model"], "response_body")
                step["legacy_model"] = step.get("model")
                step["model_identity"] = identity
                step["model"] = identity["response_model"]
            record["models"] = sorted({step["model"] for step in record["steps"]
                                       if step["kind"] == "llm" and step.get("model")})
            record["schema_version"] = 2
            db.execute("UPDATE query_logs SET record_json=? WHERE id=?",
                       (json.dumps(record, ensure_ascii=False), query_id))
            count += 1
    return backup, count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    args = parser.parse_args()
    if not args.db.is_file():
        parser.error("Log database does not exist")
    backup, count = migrate(args.db)
    print(json.dumps({"backup": str(backup), "updated_queries": count}))
