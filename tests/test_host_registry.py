"""Tests for scripts/host_registry.py."""
from pathlib import Path
import shutil
import subprocess

REPO = Path(__file__).parent.parent
SCRIPT = REPO / "scripts" / "host_registry.py"
FIX = REPO / "tests" / "fixtures" / "registry"


def setup_workdir(tmp: Path) -> Path:
    shutil.copy(FIX / "hosts.toml", tmp / ".hosts.toml")
    shutil.copy(FIX / "host-state.toml", tmp / ".host-state.toml")
    return tmp


def run(*args, cwd: Path) -> str:
    res = subprocess.run(
        ["python3", str(SCRIPT), *args],
        cwd=cwd, capture_output=True, text=True, check=True,
    )
    return res.stdout


def test_list(tmp_path):
    setup_workdir(tmp_path)
    out = run("list", cwd=tmp_path).strip().splitlines()
    assert out == ["b200-1", "h100-1", "mi300x-1"]


def test_list_summary(tmp_path):
    setup_workdir(tmp_path)
    out = run("list", "--summary", cwd=tmp_path)
    assert "h100-1" in out and "nvidia" in out and "ready" in out


def test_get_field(tmp_path):
    setup_workdir(tmp_path)
    assert run("get", "h100-1", "ip", cwd=tmp_path).strip() == "203.0.113.42"
    assert run("get", "h100-1", "vendor", cwd=tmp_path).strip() == "nvidia"


def test_get_ssh_target_synthetic(tmp_path):
    setup_workdir(tmp_path)
    assert run("get", "h100-1", "ssh_target", cwd=tmp_path).strip() == "ubuntu@203.0.113.42"


def test_match_hardware(tmp_path):
    setup_workdir(tmp_path)
    assert run("match", "--hardware", "h100", cwd=tmp_path).strip() == "h100-1"
    assert run("match", "--hardware", "any", cwd=tmp_path).strip().splitlines() == [
        "b200-1", "h100-1", "mi300x-1",
    ]


def test_match_vendor(tmp_path):
    setup_workdir(tmp_path)
    assert sorted(run("match", "--vendor", "nvidia", cwd=tmp_path).strip().splitlines()) == [
        "b200-1", "h100-1",
    ]
    assert run("match", "--vendor", "amd", cwd=tmp_path).strip() == "mi300x-1"


def test_state_set(tmp_path):
    setup_workdir(tmp_path)
    run("state", "b200-1", "--set", "ready", cwd=tmp_path)
    text = (tmp_path / ".host-state.toml").read_text()
    assert 'setup_state = "ready"' in text and "b200-1" in text


def test_schedule_intersection(tmp_path):
    setup_workdir(tmp_path)
    out = run(
        "schedule",
        "--hypothesis-hardware", "any",
        "--engine-supported", "h100,b200,mi300x",
        cwd=tmp_path,
    ).strip()
    # b200-1 is pending so excluded; h100-1 ready, mi300x-1 ready → first wins
    assert out in {"h100-1", "mi300x-1"}


def test_schedule_intersection_amd_only(tmp_path):
    setup_workdir(tmp_path)
    out = run(
        "schedule",
        "--hypothesis-hardware", "amd",
        "--engine-supported", "h100,b200,mi300x",
        cwd=tmp_path,
    ).strip()
    assert out == "mi300x-1"


def test_schedule_no_match(tmp_path):
    setup_workdir(tmp_path)
    # TRT-LLM only supports nvidia; hyp says amd — empty
    out = run(
        "schedule",
        "--hypothesis-hardware", "amd",
        "--engine-supported", "h100,b200",
        cwd=tmp_path,
    ).strip()
    assert out == "none"


def test_schedule_excludes(tmp_path):
    setup_workdir(tmp_path)
    out = run(
        "schedule",
        "--hypothesis-hardware", "nvidia",
        "--engine-supported", "h100,b200,mi300x",
        "--exclude", "h100-1",
        cwd=tmp_path,
    ).strip()
    # Only nvidia hosts are h100-1 (ready, excluded) and b200-1 (pending). Empty.
    assert out == "none"
