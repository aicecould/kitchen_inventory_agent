from pathlib import Path
import subprocess

import pytest


ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS = (
    ROOT_DIR / "scripts" / "start.sh",
    ROOT_DIR / "scripts" / "stop.sh",
    ROOT_DIR / "classifier" / "build_fasttext.sh",
)


@pytest.mark.parametrize(
    ("script", "expected"),
    [
        (SCRIPTS[0], "--no-reload"),
        (SCRIPTS[1], "--port"),
        (SCRIPTS[2], "FastText"),
    ],
)
def test_linux_script_help_works_outside_project(
    script: Path,
    expected: str,
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        ["bash", str(script), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert expected in result.stdout


@pytest.mark.parametrize("script", SCRIPTS)
def test_linux_script_has_valid_bash_syntax(script: Path) -> None:
    result = subprocess.run(
        ["bash", "-n", str(script)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("script", SCRIPTS[:2])
def test_service_script_rejects_invalid_port(script: Path) -> None:
    result = subprocess.run(
        ["bash", str(script), "--port", "70000"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "1 到 65535" in result.stderr
