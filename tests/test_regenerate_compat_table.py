"""Tests for scripts/regenerate-compat-table.py."""
from pathlib import Path
import subprocess

REPO = Path(__file__).parent.parent
SCRIPT = REPO / "scripts" / "regenerate-compat-table.py"


def write(path: Path, body: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def test_table_reflects_engine_supported_hardware(tmp_path):
    write(tmp_path / "wiki" / "engines" / "vllm.md", """---
title: "vLLM"
type: engine
supported_hardware: [h100, b200, mi300x]
---
body
""")
    write(tmp_path / "wiki" / "engines" / "trt.md", """---
title: "TensorRT-LLM"
type: engine
supported_hardware: [h100, b200]
---
body
""")
    write(tmp_path / "wiki" / "hardware" / "h100.md", """---
title: "NVIDIA H100"
type: hardware
display_order: 1
---
body
""")
    write(tmp_path / "wiki" / "hardware" / "b200.md", """---
title: "NVIDIA B200"
type: hardware
display_order: 2
---
body
""")
    write(tmp_path / "wiki" / "hardware" / "mi300x.md", """---
title: "AMD MI300X"
type: hardware
display_order: 3
---
body
""")

    subprocess.run(
        ["python3", str(SCRIPT), "--root", str(tmp_path)],
        check=True, cwd=REPO,
    )

    out = (tmp_path / "wiki" / "concepts" / "engine-hardware-compatibility.md").read_text()
    assert "| vLLM | ✓ | ✓ | ✓ |" in out
    assert "| TensorRT-LLM | ✓ | ✓ | ✗ |" in out
    assert "| h100 | b200 | mi300x |" in out
