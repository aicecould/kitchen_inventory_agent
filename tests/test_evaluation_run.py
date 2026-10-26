import json
import os
import subprocess
import sys
from pathlib import Path

from evaluation.run import _repo_root

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)


def test_repo_root_finds_shared_checkout_artifacts(monkeypatch) -> None:
    monkeypatch.delenv("EVALUATION_PROJECT_ROOT", raising=False)
    assert (_repo_root() / "recommender/artifacts/recipes.db").is_file()


def test_offline_cli_writes_one_report_pair(tmp_path) -> None:
    environment = dict(os.environ)
    environment["EVALUATION_PROJECT_ROOT"] = str(ROOT)
    completed = subprocess.run(
        [str(PYTHON), "-m", "evaluation.run", "--suite", "quick", "--offline",
         "--output-dir", str(tmp_path)],
        cwd=ROOT, env=environment, text=True, capture_output=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    json_files = list(tmp_path.glob("*.json"))
    markdown_files = list(tmp_path.glob("*.md"))
    assert len(json_files) == len(markdown_files) == 1
    payload = json.loads(json_files[0].read_text(encoding="utf-8"))
    names = {suite["name"] for suite in payload["suites"]}
    assert {"classifier_router", "local_retrieval", "deterministic_auditor",
            "cooking_session", "end_to_end_paths"}.issubset(names)
