"""Download, verify GPU generation, then run the local inference service."""

import fcntl
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
RUN_DIR = PROJECT_ROOT / ".run" / "qwen3_1_7b"


def status(stage, **details):
    (RUN_DIR / "deployment-status.json").write_text(
        json.dumps({"stage": stage, **details}, ensure_ascii=False, indent=2) + "\n"
    )
    print(stage, flush=True)


if __name__ == "__main__":
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with (RUN_DIR / ".deploy.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for stage, script in (("downloading", "download.py"),
                                  ("verifying_gpu_inference", "smoke.py"),
                                  ("starting_service", "serve.py")):
                status(stage)
                subprocess.run(
                    [sys.executable, "-m", "llm.qwen3_1_7b." + Path(script).stem],
                    cwd=PROJECT_ROOT, check=True,
                )
            status("stopped")
        except Exception as exc:
            status("failed", error=str(exc))
            raise
