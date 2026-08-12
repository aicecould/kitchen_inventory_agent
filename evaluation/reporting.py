"""Write raw JSON and human-readable Markdown from the same result object."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluation.models import EvaluationRun


_FORBIDDEN = ("api_key", "secret", "token_value", "prompt", "raw_response")


def _validate_metadata(value: object, path: str = "metadata") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).casefold()
            if any(term in normalized for term in _FORBIDDEN):
                raise ValueError(f"forbidden metadata key: {path}.{key}")
            _validate_metadata(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_metadata(child, f"{path}[{index}]")


def _format(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list)):
        return f"`{json.dumps(value, ensure_ascii=False, sort_keys=True)}`"
    return str(value)


def render_markdown(run: EvaluationRun) -> str:
    _validate_metadata(run.metadata)
    lines = ["# Kitchen Inventory Agent 快速评测报告", "", "## 运行信息", ""]
    for key, value in sorted(run.metadata.items()):
        lines.append(f"- {key}: {_format(value)}")
    for suite in run.suites:
        lines.extend([
            "", f"## {suite.name}", "",
            f"证据类型：`{suite.evidence.value}`；样本数：{len(suite.samples)}。", "",
            "| 指标 | 数值 |", "| --- | ---: |",
        ])
        for key, value in sorted(suite.summary.items()):
            lines.append(f"| {key} | {_format(value)} |")
        if suite.error:
            lines.extend(["", f"套件错误：`{suite.error}`"])
    lines.extend(["", "## 局限", ""])
    lines.extend(f"- {value}" for value in run.limitations)
    return "\n".join(lines) + "\n"


def write_reports(
    run: EvaluationRun, output_dir: Path, run_id: str
) -> tuple[Path, Path]:
    _validate_metadata(run.metadata)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_id}.json"
    markdown_path = output_dir / f"{run_id}.md"
    json_path.write_text(
        json.dumps(run.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(run), encoding="utf-8")
    return json_path, markdown_path
