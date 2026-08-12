"""Strict loading for fixed JSON benchmark records."""

from __future__ import annotations

import json
from pathlib import Path


def load_json_records(
    path: Path, *, required_keys: set[str]
) -> list[dict[str, object]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid benchmark JSON: {path.name}") from exc
    if not isinstance(payload, list):
        raise ValueError("benchmark must be a JSON array")
    records: list[dict[str, object]] = []
    identifiers: set[str] = set()
    for index, value in enumerate(payload):
        if not isinstance(value, dict):
            raise ValueError(f"benchmark row {index} must be an object")
        for key in sorted(required_keys):
            if key not in value:
                raise ValueError(f"benchmark row {index} missing required key: {key}")
        identifier = value.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"benchmark row {index} has invalid id")
        if identifier in identifiers:
            raise ValueError(f"duplicate benchmark id: {identifier}")
        identifiers.add(identifier)
        records.append(value)
    return records
