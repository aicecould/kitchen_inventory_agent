"""Unified command-line runner for the quick evaluation suite."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import subprocess
import tempfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from app.config import get_settings
from app.llm_log import logging_context
from evaluation.auditor import run_auditor_suite
from evaluation.classifier import run_classifier_suite
from evaluation.cooking_session import run_cooking_session_suite
from evaluation.end_to_end import run_end_to_end_suite
from evaluation.models import EvaluationRun, EvidenceKind, SuiteResult
from evaluation.online import run_gate_cost_probe
from evaluation.reporting import write_reports
from evaluation.retrieval import run_retrieval_suite
from evaluation.supervisor import run_supervisor_suite


BENCHMARKS = Path(__file__).resolve().parent / "benchmarks"
LIMITATIONS = [
    "快速 benchmark 样本量小，不代表统计显著性。",
    "检索目标由本地已收录菜谱反向构造，偏向已知语料。",
    "所选模型输出和推理延迟存在波动。",
    "故障注入覆盖已知风险，不代表生产错误的真实发生频率。",
    "分类器元数据声明测试集 245 条，当前可审查去重数据重放为 243 条。",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _repo_root() -> Path:
    configured = os.getenv("EVALUATION_PROJECT_ROOT")
    if configured:
        return Path(configured).resolve()
    checkout = Path(__file__).resolve().parents[1]
    common_git = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=checkout,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if common_git:
        common_root = Path(common_git).resolve().parent
        if (common_root / "recommender/artifacts/recipes.db").is_file():
            return common_root
    return checkout


def _git_metadata() -> tuple[str, bool]:
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
        check=False,
    ).stdout.strip() or "unavailable"
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True,
        check=False,
    ).stdout.strip())
    return commit, dirty


def run_quick(*, offline: bool, suite: str = "quick") -> EvaluationRun:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    with logging_context(source="evaluation", run_id=run_id):
        return _run_quick(offline=offline, run_id=run_id, suite=suite)


def _run_quick(*, offline: bool, run_id: str, suite: str = "quick") -> EvaluationRun:
    if suite not in {"quick", "llm"}:
        raise ValueError("suite must be quick or llm")
    source_root = _repo_root()
    settings = get_settings()
    settings = replace(
        settings,
        food_classifier_model_path=source_root / settings.food_classifier_model_path,
        food_classifier_metadata_path=source_root / settings.food_classifier_metadata_path,
        recommender_db_path=source_root / settings.recommender_db_path,
        recommender_faiss_path=source_root / settings.recommender_faiss_path,
        recommender_metadata_path=source_root / settings.recommender_metadata_path,
        recommender_model_path=source_root / settings.recommender_model_path,
    )
    commit, dirty = _git_metadata()
    suites: list[SuiteResult] = []
    with tempfile.TemporaryDirectory(prefix="kitchen-evaluation-") as temp_value:
        temp_dir = Path(temp_value)
        runners = [
            ("classifier_router", lambda: run_classifier_suite(
                settings, BENCHMARKS / "router_challenges.json")),
            ("local_retrieval", lambda: run_retrieval_suite(
                settings, BENCHMARKS / "retrieval_targets.json",
                BENCHMARKS / "inventory_scenarios.json")),
            ("deterministic_auditor", lambda: run_auditor_suite(
                BENCHMARKS / "audit_cases.json")),
            ("gate_cost_probe", lambda: run_gate_cost_probe(
                settings,
                ["今天有什么科技新闻？", "帮我写 Python 排序函数", "Will it rain tomorrow?"],
                offline=offline)),
            ("repair_subagent", lambda: run_supervisor_suite(
                settings, BENCHMARKS / "supervisor_cases.json", offline=offline)),
            ("cooking_session", lambda: run_cooking_session_suite(
                settings, BENCHMARKS / "cooking_sessions.json", offline=offline,
                temp_dir=temp_dir / "cooking")),
            ("end_to_end_paths", lambda: run_end_to_end_suite(
                settings, BENCHMARKS / "end_to_end_cases.json", offline=offline,
                temp_dir=temp_dir / "e2e")),
        ]
        for name, runner in runners:
            if suite == "llm" and name not in {
                "deterministic_auditor", "gate_cost_probe", "repair_subagent"
            }:
                continue
            try:
                (temp_dir / "cooking").mkdir(exist_ok=True)
                (temp_dir / "e2e").mkdir(exist_ok=True)
                suites.append(runner())
            except Exception as exc:
                suites.append(SuiteResult(
                    name=name, evidence=EvidenceKind.UNAVAILABLE,
                    error=type(exc).__name__, summary={"status": "failed"},
                ))
    metadata = {
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": commit,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "mode": "offline" if offline else "online",
        "suite": suite,
        "deepseek_model": settings.deepseek_model,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "llm_max_output_tokens": (
            settings.qwen_max_output_tokens if settings.llm_provider == "qwen"
            else settings.deepseek_max_output_tokens
        ),
        "llm_timeout_seconds": (
            settings.qwen_timeout_seconds if settings.llm_provider == "qwen"
            else settings.http_timeout_seconds
        ),
        "benchmark_hashes": {
            path.name: _sha256(path) for path in sorted(BENCHMARKS.glob("*.json"))
        },
    }
    limitations = list(LIMITATIONS)
    if suite == "llm":
        limitations = [
            "仅运行领域门禁成本探针、回答修复子 Agent 与确定性审核基线。",
            "门禁探针的 passed 表示请求成功，不等同于拒答内容正确。",
            "确定性审核为规则故障注入结果，不是模型生成正确率。",
            "不包含真实业务端到端、库存工具调用、菜谱检索或多轮烹饪生成评测。",
            "样本量仅 9 次模型调用，不能推导通用能力或统计显著性。",
            "记录单次实测结果，不对失败样本反复重跑取优。",
        ]
    return EvaluationRun(metadata=metadata, suites=suites, limitations=limitations)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quick system evaluation")
    parser.add_argument("--suite", choices=["quick", "llm"], default="quick")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(f"{get_settings().llm_provider} maximum calls: gate_cost_probe=3, repair_subagent=6")
        print(f"Spoonacular maximum calls: {0 if args.suite == 'llm' else 5}")
        return
    run = run_quick(offline=args.offline, suite=args.suite)
    run_id = str(run.metadata["run_id"])
    json_path, markdown_path = write_reports(run, args.output_dir, run_id)
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
