"""Tests for scripts/migrate-frontmatter.py."""
from pathlib import Path
import subprocess
import shutil

REPO = Path(__file__).parent.parent
SCRIPT = REPO / "scripts" / "migrate-frontmatter.py"
FIX = REPO / "tests" / "fixtures" / "migrate"


def run(args, cwd):
    return subprocess.run(
        ["python3", str(SCRIPT), *args],
        cwd=cwd, capture_output=True, text=True, check=True,
    )


def test_adds_hardware_any_to_hypothesis(tmp_path):
    work = tmp_path / "wiki" / "hypotheses"
    work.mkdir(parents=True)
    target = work / "test.md"
    target.write_text((FIX / "before-hypothesis.md").read_text())

    run(["--root", str(tmp_path)], cwd=REPO)

    expected = (FIX / "after-hypothesis.md").read_text()
    assert target.read_text() == expected


def test_idempotent(tmp_path):
    work = tmp_path / "wiki" / "hypotheses"
    work.mkdir(parents=True)
    target = work / "test.md"
    target.write_text((FIX / "after-hypothesis.md").read_text())

    run(["--root", str(tmp_path)], cwd=REPO)

    # Already-migrated file is unchanged
    assert target.read_text() == (FIX / "after-hypothesis.md").read_text()
