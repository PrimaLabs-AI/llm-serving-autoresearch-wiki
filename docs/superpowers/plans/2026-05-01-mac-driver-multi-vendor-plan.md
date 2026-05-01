# Mac-Driver Multi-Vendor Autoresearch Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Claude Code on the user's Mac drive autonomous LLM-serving optimization experiments across long-lived NVIDIA (H100/B200) and AMD (MI300X) GPU boxes via SSH, with the wiki on the Mac as the only authoritative state.

**Architecture:** Bash conductor on Mac runs `run_loop.sh`. Per round, it (1) ensures host setup via SSH bootstrap, (2) calls `claude --print` with `prompts/pick.md` to choose the next hypothesis, (3) reads `hypothesis.hardware × engine.supported_hardware × .hosts.toml` to pick a compatible host, (4) calls `claude --print` with `prompts/run.md` so Claude SSHes in, runs `benchmark_harness.py`, rsyncs results back, and writes the experiment page. Hardware compatibility lives in frontmatter and host-registry tags — bash enforces it deterministically. Each round is a fresh Claude session; cross-round memory is the wiki.

**Tech Stack:** Bash 5+, Python 3.11+ (stdlib `tomllib` only, no external Python deps), pytest (new), SSH with ControlMaster pooling, rsync, git, Claude Code (`claude --print` headless mode).

**Spec:** [`docs/superpowers/specs/2026-05-01-mac-driver-multi-vendor-design.md`](../specs/2026-05-01-mac-driver-multi-vendor-design.md)

---

## Slice 1: Engine ingestion (no GPU)

Populate `wiki/engines/{vllm,sglang,tensorrt-llm}.md` from upstream sources so Claude has real priors before the loop runs. Adds the `supported_hardware` frontmatter field that the scheduler depends on. Uses the existing `INGEST-CODEBASE` operation defined in `SCHEMA.md`.

### Task 1.1: Ingest vLLM

**Files:**
- Modify: `wiki/engines/vllm.md` (replace stub)
- Modify: `wiki/index.md` (update one-line status)
- Modify: `wiki/log.md` (append-only entry, newest on top)

- [ ] **Step 1: Read the current vLLM stub and the SCHEMA engine template**

```bash
cat wiki/engines/vllm.md
grep -A 12 "^### engine" SCHEMA.md
```

- [ ] **Step 2: Verify vLLM source is available locally**

```bash
ls raw/code/vllm/README.md raw/code/vllm/.git/HEAD 2>/dev/null && \
  git -C raw/code/vllm rev-parse --short HEAD
```

If absent, the user will need to add `vllm` as a submodule under `raw/code/vllm`. Stop and request that before continuing.

- [ ] **Step 3: Populate the page per the SCHEMA `engine` template**

Required frontmatter (extends current minimal stub):

```yaml
---
title: "vLLM"
type: engine
tags: [engine, vllm, paged-attention]
commit: <short-sha-from-step-2>
supported_hardware: [h100, h200, b200, mi300x]
created: <existing-or-today>
updated: <today>
---
```

Required H2 sections (`SCHEMA.md` lines 132-146):
- Overview
- Architecture (scheduler, KV cache manager, batching strategy)
- Key abstractions
- Entry points
- Serving-relevant surfaces — **mandatory**, with file:line refs from `raw/code/vllm/`
- Tunable knobs (scheduler / KV cache / batching / parallelism / quantization / CUDA graphs)
- Supported models
- Known strengths/weaknesses
- Connections (links to `wiki/workloads/`, `wiki/hypotheses/`)
- Sources (raw paths)

The Tunable knobs section must list the flags `benchmark_harness.py` already maps (see `benchmark_harness.py:36-50`): `max_num_seqs`, `max_num_batched_tokens`, `gpu_memory_utilization`, `enable_prefix_caching`, `enable_chunked_prefill`, `block_size`, `tensor_parallel_size`, `pipeline_parallel_size`, `data_parallel_size`, `quantization`, `dtype`, `enforce_eager`, `swap_space`. Each with file:line ref into `raw/code/vllm/`.

- [ ] **Step 4: Update `wiki/index.md` engine section**

Replace the stub line:

```markdown
- [vLLM](engines/vllm.md) — PagedAttention serving; commit `<sha>` — supported on H100/H200/B200/MI300X
```

- [ ] **Step 5: Append to `wiki/log.md` (newest on top)**

```markdown
## [2026-05-01] ingest-codebase | vllm

**Op**: ingest-codebase
**Pages created**: —
**Pages updated**: wiki/engines/vllm.md, wiki/index.md
**Key result**: vLLM ingested at commit <sha>; supported_hardware = [h100, h200, b200, mi300x]
**Notes**: First non-stub engine page. Frontmatter `supported_hardware` field unblocks slice 6 scheduler.
```

- [ ] **Step 6: Commit**

```bash
git add wiki/engines/vllm.md wiki/index.md wiki/log.md
git commit -m "Ingest vLLM into wiki/engines/vllm.md"
```

### Task 1.2: Ingest SGLang

**Files:**
- Modify: `wiki/engines/sglang.md`
- Modify: `wiki/index.md`
- Modify: `wiki/log.md`

- [ ] **Step 1: Verify SGLang source is available**

```bash
ls raw/code/sglang/README.md 2>/dev/null && git -C raw/code/sglang rev-parse --short HEAD
```

If absent, request the user add it as a submodule.

- [ ] **Step 2: Populate the page per the engine template**

Same shape as task 1.1 step 3. Frontmatter:

```yaml
---
title: "SGLang"
type: engine
tags: [engine, sglang, radix-attention]
commit: <short-sha>
supported_hardware: [h100, h200, b200, mi300x]
created: <existing-or-today>
updated: <today>
---
```

Tunable knobs section must list the flags `benchmark_harness.py:57-68` maps: `tp`, `dp`, `pp`, `mem_fraction_static`, `chunk_prefill_size`, `enable_overlap_schedule`, `enable_dp_attention`, `disable_cuda_graph`, `quantization`, `dtype`, with file:line refs into `raw/code/sglang/`.

- [ ] **Step 3: Update `wiki/index.md`**

```markdown
- [SGLang](engines/sglang.md) — RadixAttention serving; commit `<sha>` — supported on H100/H200/B200/MI300X
```

- [ ] **Step 4: Append to `wiki/log.md`**

Format identical to task 1.1 step 5, with `sglang` substitutions.

- [ ] **Step 5: Commit**

```bash
git add wiki/engines/sglang.md wiki/index.md wiki/log.md
git commit -m "Ingest SGLang into wiki/engines/sglang.md"
```

### Task 1.3: Ingest TensorRT-LLM

**Files:**
- Modify: `wiki/engines/tensorrt-llm.md`
- Modify: `wiki/index.md`
- Modify: `wiki/log.md`

- [ ] **Step 1: Verify TensorRT-LLM source is available**

```bash
ls raw/code/TensorRT-LLM/README.md 2>/dev/null && git -C raw/code/TensorRT-LLM rev-parse --short HEAD
```

If absent, request the user add it as a submodule.

- [ ] **Step 2: Populate per the engine template**

Frontmatter — note `supported_hardware` excludes MI300X (NVIDIA-only):

```yaml
---
title: "TensorRT-LLM"
type: engine
tags: [engine, tensorrt-llm, nvidia]
commit: <short-sha>
supported_hardware: [h100, h200, b200]
created: <existing-or-today>
updated: <today>
---
```

Tunable knobs from `benchmark_harness.py:74-84`: `tensor_parallel`, `pipeline_parallel`, `dtype`, `max_batch_size`, `max_seq_len`, plus engine-specific: `--use_paged_kv_cache`, `--use_inflight_batching`, `--gemm_plugin`, `--gpt_attention_plugin`. File:line refs into `raw/code/TensorRT-LLM/`.

- [ ] **Step 3: Update `wiki/index.md` and `wiki/log.md`**

Same shape as previous tasks. Note "**NVIDIA only**" in the index line.

- [ ] **Step 4: Commit**

```bash
git add wiki/engines/tensorrt-llm.md wiki/index.md wiki/log.md
git commit -m "Ingest TensorRT-LLM into wiki/engines/tensorrt-llm.md"
```

---

## Slice 2: Hardware pages (no GPU)

Seed `wiki/hardware/` with the three target tiers. Mirrors the shape of `wiki/engines/` and `wiki/workloads/`.

### Task 2.1: Create hardware page template directory and H100 page

**Files:**
- Create: `wiki/hardware/h100.md`

- [ ] **Step 1: Create the directory and the H100 page**

```bash
mkdir -p wiki/hardware
```

Page contents:

```markdown
---
title: "NVIDIA H100"
type: hardware
tags: [hardware, nvidia, hopper]
vendor: nvidia
arch: hopper
created: 2026-05-01
updated: 2026-05-01
---

NVIDIA H100 (Hopper) datacenter GPU. Tensor Core gen 4 with FP8 support, transformer engine, TMA, async copies. Primary target for vLLM/SGLang/TensorRT-LLM serving.

## Specs

| Metric | SXM5 | PCIe |
|---|---|---|
| Peak BF16/FP16 | 989 TFLOPs | 756 TFLOPs |
| Peak FP8 | 1979 TFLOPs | 1513 TFLOPs |
| HBM | 80 GB HBM3 | 80 GB HBM3 |
| HBM bandwidth | 3.35 TB/s | 2.0 TB/s |
| NVLink | 900 GB/s (4th gen, 18 links) | 600 GB/s (PCIe Gen 5) |
| SMs | 132 | 114 |
| TDP | 700 W | 350 W |

## Memory hierarchy

HBM3 (80 GB, 3.35 TB/s) → L2 (50 MB shared) → register file (256 KB/SM) → shared mem/L1 (228 KB/SM split). Distributed shared memory across SM clusters via TMA.

## Compute features

- 4th-gen Tensor Cores (FP8, FP16, BF16, TF32, INT8)
- Transformer Engine: dynamic FP8 scaling for attention/FFN
- TMA (Tensor Memory Accelerator) for async tile loads
- Thread Block Cluster + Distributed Shared Memory
- Asynchronous copies (cp.async)

## Engine support

| Engine | Supported | Notes |
|---|---|---|
| vLLM | ✓ | First-class |
| SGLang | ✓ | First-class |
| TensorRT-LLM | ✓ | Best single-GPU peak; longer compile time |

## Known performance ceilings

To be filled from experiments. Reference: scaling-book ch7 inference rooflines, vLLM published benchmarks.

## Optimization gotchas

- Requires CUDA 12.0+
- FP8 transformer engine needs PyTorch 2.1+ and Hopper-aware kernels
- NVLink topology matters for tensor parallelism — verify `nvidia-smi topo -m`

## Connections

- Engines: [vLLM](../engines/vllm.md), [SGLang](../engines/sglang.md), [TensorRT-LLM](../engines/tensorrt-llm.md)

## Sources

- NVIDIA H100 datasheet
- Hopper architecture whitepaper
```

- [ ] **Step 2: Commit**

```bash
git add wiki/hardware/h100.md
git commit -m "Add wiki/hardware/h100.md"
```

### Task 2.2: Create B200 page

**Files:**
- Create: `wiki/hardware/b200.md`

- [ ] **Step 1: Create the page**

Same template as H100, with Blackwell-specific facts:

```markdown
---
title: "NVIDIA B200"
type: hardware
tags: [hardware, nvidia, blackwell]
vendor: nvidia
arch: blackwell
created: 2026-05-01
updated: 2026-05-01
---

NVIDIA B200 (Blackwell) datacenter GPU. 5th-gen Tensor Cores with FP4 support, 2nd-gen Transformer Engine, NVLink-5. Successor to H100 with ~2.5× FP8 throughput per chip.

## Specs

| Metric | SXM |
|---|---|
| Peak BF16/FP16 | 2.25 PFLOPs |
| Peak FP8 | 4.5 PFLOPs |
| Peak FP4 | 9 PFLOPs |
| HBM | 192 GB HBM3e |
| HBM bandwidth | 8.0 TB/s |
| NVLink | 1.8 TB/s (5th gen) |
| TDP | 1000 W |

## Memory hierarchy

HBM3e (192 GB, 8 TB/s) → L2 → SM-local. Two reticle-sized dies bridged with high-bandwidth NV-HBI interconnect; appears as single GPU to software.

## Compute features

- 5th-gen Tensor Cores (FP4, FP6, FP8, FP16, BF16, TF32, INT8)
- 2nd-gen Transformer Engine — per-tensor FP8/FP4 scaling
- NVLink-5 (1.8 TB/s) for 72-GPU NVLink domains (NVL72)
- Dual-die design transparent to software

## Engine support

| Engine | Supported | Notes |
|---|---|---|
| vLLM | ✓ | Requires vLLM ≥ 0.6 + CUDA 12.8 |
| SGLang | ✓ | Same caveats |
| TensorRT-LLM | ✓ | FP4 inference path requires nightly builds early 2026 |

## Known performance ceilings

To be filled from experiments. ~2.5× FP8 peak vs H100 per-chip headline.

## Optimization gotchas

- CUDA 12.8+ required for full Blackwell feature set
- FP4 path needs explicit kernel selection — not enabled by default in vLLM as of early 2026
- NVLink-5 vs PCIe Gen 5 bandwidth gap is wider than on Hopper — pay attention to TP placement

## Connections

- Engines: [vLLM](../engines/vllm.md), [SGLang](../engines/sglang.md), [TensorRT-LLM](../engines/tensorrt-llm.md)

## Sources

- NVIDIA Blackwell architecture whitepaper
- NVIDIA B200 datasheet
```

- [ ] **Step 2: Commit**

```bash
git add wiki/hardware/b200.md
git commit -m "Add wiki/hardware/b200.md"
```

### Task 2.3: Create MI300X page

**Files:**
- Create: `wiki/hardware/mi300x.md`

- [ ] **Step 1: Create the page**

```markdown
---
title: "AMD Instinct MI300X"
type: hardware
tags: [hardware, amd, cdna3, rocm]
vendor: amd
arch: cdna3
created: 2026-05-01
updated: 2026-05-01
---

AMD Instinct MI300X (CDNA3) datacenter GPU. Chiplet-based design with 8 XCDs (compute dies) and 4 IODs, 192 GB HBM3, 5.3 TB/s bandwidth. Primary AMD target for LLM serving via ROCm.

## Specs

| Metric | OAM |
|---|---|
| Peak BF16/FP16 | 1.31 PFLOPs |
| Peak FP8 | 2.61 PFLOPs |
| HBM | 192 GB HBM3 |
| HBM bandwidth | 5.3 TB/s |
| Infinity Fabric | 896 GB/s (peer-to-peer) |
| Compute units | 304 |
| TDP | 750 W |

## Memory hierarchy

HBM3 (192 GB, 5.3 TB/s) → Infinity Cache (256 MB shared across IODs) → L2 (XCD-local) → LDS (64 KB/CU) → register file. Chiplet topology means latency varies depending on data origin.

## Compute features

- Matrix cores: BF16/FP16 at 1.31 PFLOPs, FP8 at 2.61 PFLOPs
- No native FP4 (as of CDNA3)
- Sparsity: 2:4 structured sparsity for matrix cores
- Unified memory between CPU+GPU on MI300A (APU variant); MI300X is GPU-only

## Engine support

| Engine | Supported | Notes |
|---|---|---|
| vLLM | ✓ | ROCm build via `VLLM_TARGET_DEVICE=rocm`; flash-attn via Triton-on-ROCm |
| SGLang | ✓ | ROCm build via `--device rocm` |
| TensorRT-LLM | ✗ | NVIDIA-only — not portable |

## Known performance ceilings

To be filled from experiments. Tighter HBM/FLOPs ratio than H100 (more bandwidth per FLOP) — some bandwidth-bound serving workloads may favor MI300X.

## Optimization gotchas

- ROCm 6.2+ required for recent vLLM/SGLang ROCm builds
- Flash Attention path uses Triton-on-ROCm or AMD composable-kernels (CK); not the same kernels as NVIDIA flash-attn
- HIP graphs (CUDA-graph equivalent) are still maturing; use `--enforce-eager` if you hit instability
- FP8 supported but ecosystem is younger than NVIDIA's transformer engine
- TensorRT-LLM hypotheses are blocked by `engine.supported_hardware` — scheduler will skip them
- Chiplet topology: peer-to-peer bandwidth across XCDs/IODs is uneven; tensor parallelism layout matters

## Connections

- Engines: [vLLM](../engines/vllm.md), [SGLang](../engines/sglang.md)

## Sources

- AMD CDNA3 architecture whitepaper
- AMD Instinct MI300X datasheet
- ROCm 6.x release notes
```

- [ ] **Step 2: Commit**

```bash
git add wiki/hardware/mi300x.md
git commit -m "Add wiki/hardware/mi300x.md"
```

### Task 2.4: Update wiki index with hardware section

**Files:**
- Modify: `wiki/index.md`
- Modify: `wiki/log.md`

- [ ] **Step 1: Insert hardware section after Engines section in `wiki/index.md`**

Add a new H2 between Engines (3) and Workloads:

```markdown
## Hardware (3)
- [NVIDIA H100](hardware/h100.md) — Hopper, 80 GB HBM3, 989 BF16 TFLOPs / 1979 FP8 TFLOPs (SXM5)
- [NVIDIA B200](hardware/b200.md) — Blackwell, 192 GB HBM3e, 2.25 BF16 PFLOPs / 4.5 FP8 PFLOPs / 9 FP4 PFLOPs
- [AMD Instinct MI300X](hardware/mi300x.md) — CDNA3, 192 GB HBM3, 1.31 BF16 PFLOPs / 2.61 FP8 PFLOPs
```

Update the page-count in line 2 (`Last updated: ... — N pages`) by +3.

- [ ] **Step 2: Append a single combined log entry**

```markdown
## [2026-05-01] manual | seed wiki/hardware/

**Op**: manual
**Pages created**: wiki/hardware/h100.md, wiki/hardware/b200.md, wiki/hardware/mi300x.md
**Pages updated**: wiki/index.md
**Key result**: New `hardware` page type seeded; supports phase-3 hardware-as-dimension schema work.
**Notes**: H200 and MI325X stubs deferred until first experiment touches them.
```

- [ ] **Step 3: Commit**

```bash
git add wiki/index.md wiki/log.md
git commit -m "Add hardware section to wiki index"
```

---

## Slice 3: Schema additions and frontmatter migration (no GPU)

Lock in the new `hardware` page type, the new frontmatter fields, and migrate existing pages. This is the dependency every other slice's scheduling logic needs.

### Task 3.1: Update SCHEMA.md — add hardware page type

**Files:**
- Modify: `SCHEMA.md`

- [ ] **Step 1: Add `hardware` to the page-type list at line 79**

Find the line in SCHEMA.md describing page types in the directory layout description (`wiki/concepts/`, `wiki/models/`, etc.) and add a `wiki/hardware/` entry.

- [ ] **Step 2: Add `hardware` to the frontmatter `type:` enum**

In SCHEMA.md's "Page format" section, find the `type:` field documentation and add `hardware` to the allowed values.

- [ ] **Step 3: Add the `hardware` page-type template section**

Insert a new section between `engine` and `workload` page templates:

```markdown
### hardware  (`wiki/hardware/<slug>.md`)
A GPU/accelerator tier under optimization (e.g., h100, b200, mi300x).
- H2: Specs (table), Memory hierarchy, Compute features, Engine support (table), Known performance ceilings, Optimization gotchas, Connections, Sources.
- Frontmatter must carry `vendor:` (`nvidia`/`amd`/`tpu`), `arch:` (`hopper`/`blackwell`/`cdna3`/etc.).
- Slugs (`h100`, `b200`, `mi300x`) are referenced by `hosts.toml`, by `hypothesis.hardware`, by `experiment.hardware`/`host`, by `model.target_hardware`, and by `engine.supported_hardware`.
```

- [ ] **Step 4: Add `hardware` and `host` frontmatter fields to existing page-type sections**

For each page type, add the required fields to its `Frontmatter adds:` block:

- `hypothesis`: `hardware: any | nvidia | amd | <slug>`
- `experiment`: `hardware: <slug>` and `host: <host-name-from-hosts-toml>`
- `model`: `target_hardware: [<slug>, ...]`
- `engine`: `supported_hardware: [<slug>, ...]`

- [ ] **Step 5: Add behavioral rule #15 enforcing the new fields**

After existing rule #14 in the "Behavioral rules" section:

```markdown
15. **Hardware compatibility is data, not prose.** Every hypothesis carries `hardware:`; every experiment carries `hardware:` and `host:`; every engine carries `supported_hardware:`. The orchestration loop's scheduler intersects these to dispatch a round to a compatible host. Pages missing these fields will fail `LINT`.
```

- [ ] **Step 6: Update `LINT` operation to check the new fields**

In the `LINT` operation section, add bullet:

```markdown
- Hypotheses missing `hardware:` frontmatter; experiments missing `hardware:` or `host:`; engines missing `supported_hardware:`. (Auto-fix not safe — flag for human.)
```

- [ ] **Step 7: Commit**

```bash
git add SCHEMA.md
git commit -m "SCHEMA: add hardware page type and compatibility frontmatter"
```

### Task 3.2: Write frontmatter migration script

**Files:**
- Create: `scripts/migrate-frontmatter.py`
- Create: `tests/test_migrate_frontmatter.py`

- [ ] **Step 1: Create test fixtures and write the failing test**

```bash
mkdir -p tests/fixtures/migrate
```

`tests/fixtures/migrate/before-hypothesis.md`:
```markdown
---
title: "Test hypothesis"
type: hypothesis
tags: [stub]
status: open
expected_gain: "10%"
confidence: medium
effort: S
origin: human
---
Body.
```

`tests/fixtures/migrate/after-hypothesis.md`:
```markdown
---
title: "Test hypothesis"
type: hypothesis
tags: [stub]
status: open
expected_gain: "10%"
confidence: medium
effort: S
origin: human
hardware: any
---
Body.
```

`tests/test_migrate_frontmatter.py`:
```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /Users/adarshverma/Developer/Primalabs/llm-serving-autoresearch-wiki
python3 -m pytest tests/test_migrate_frontmatter.py -v
```

Expected: FAIL with "No such file or directory" (script doesn't exist yet).

- [ ] **Step 3: Implement the migration script**

`scripts/migrate-frontmatter.py`:
```python
#!/usr/bin/env python3
"""Add new required frontmatter fields to existing wiki pages.

Idempotent: pages that already carry the new fields are left unchanged.

Usage:
  python3 scripts/migrate-frontmatter.py [--root <repo-root>] [--dry-run]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

FRONTMATTER_DELIM = "---"


def split_frontmatter(text: str) -> tuple[list[str] | None, str]:
    """Return (frontmatter-lines, body). frontmatter-lines is None if absent."""
    if not text.startswith(FRONTMATTER_DELIM + "\n"):
        return None, text
    rest = text[len(FRONTMATTER_DELIM) + 1:]
    end = rest.find("\n" + FRONTMATTER_DELIM + "\n")
    if end == -1:
        return None, text
    fm = rest[:end].splitlines()
    body = rest[end + len(FRONTMATTER_DELIM) + 2:]
    return fm, body


def serialize(fm: list[str], body: str) -> str:
    return FRONTMATTER_DELIM + "\n" + "\n".join(fm) + "\n" + FRONTMATTER_DELIM + "\n" + body


def has_field(fm: list[str], key: str) -> bool:
    prefix = f"{key}:"
    return any(line.strip().startswith(prefix) for line in fm)


def add_field(fm: list[str], key: str, value: str) -> list[str]:
    if has_field(fm, key):
        return fm
    return fm + [f"{key}: {value}"]


def migrate_hypothesis(fm: list[str]) -> list[str]:
    return add_field(fm, "hardware", "any")


def migrate_experiment(fm: list[str]) -> list[str]:
    fm = add_field(fm, "hardware", "tpu-v6e")
    fm = add_field(fm, "host", "legacy-tpu")
    return fm


def migrate_model(fm: list[str]) -> list[str]:
    return add_field(fm, "target_hardware", "[tpu-v6e]")


MIGRATIONS = {
    "hypotheses": migrate_hypothesis,
    "experiments": migrate_experiment,
    "models": migrate_model,
}


def migrate_file(path: Path, migrate, dry_run: bool) -> bool:
    text = path.read_text()
    fm, body = split_frontmatter(text)
    if fm is None:
        return False
    new_fm = migrate(fm)
    if new_fm == fm:
        return False
    if not dry_run:
        path.write_text(serialize(new_fm, body))
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    root = Path(args.root) / "wiki"
    if not root.exists():
        print(f"no wiki dir at {root}", file=sys.stderr)
        sys.exit(2)

    changed = 0
    for subdir, fn in MIGRATIONS.items():
        target = root / subdir
        if not target.exists():
            continue
        for md in sorted(target.rglob("*.md")):
            if migrate_file(md, fn, args.dry_run):
                changed += 1
                print(f"migrated: {md.relative_to(args.root)}")

    print(f"{changed} files {'would be ' if args.dry_run else ''}migrated")


if __name__ == "__main__":
    main()
```

```bash
chmod +x scripts/migrate-frontmatter.py
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
python3 -m pytest tests/test_migrate_frontmatter.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/migrate-frontmatter.py tests/test_migrate_frontmatter.py tests/fixtures/migrate/
git commit -m "Add scripts/migrate-frontmatter.py with tests"
```

### Task 3.3: Run migration on existing hypotheses, experiments, models

**Files:**
- Modify: every `wiki/hypotheses/*.md` (8 files)
- Modify: every `wiki/experiments/*.md` (~19 files, recursive)
- Modify: every `wiki/models/*.md` (whatever exists; may be 0)

- [ ] **Step 1: Dry-run to see what will change**

```bash
python3 scripts/migrate-frontmatter.py --dry-run
```

Expected output: a list of files that will be migrated, ending with a count line.

- [ ] **Step 2: Run for real**

```bash
python3 scripts/migrate-frontmatter.py
```

- [ ] **Step 3: Spot-check the changes**

```bash
git diff --stat wiki/
head -20 wiki/hypotheses/prefix-caching-multi-turn-agentic.md
```

Confirm `hardware: any` was added; nothing else changed.

- [ ] **Step 4: Engine `supported_hardware` was already added in slice 1**

No further migration needed for engines. Verify:

```bash
grep -l "^supported_hardware:" wiki/engines/*.md | wc -l
```

Expected: `3`.

- [ ] **Step 5: Commit**

```bash
git add wiki/
git commit -m "Migrate existing wiki pages to new hardware-aware frontmatter"
```

### Task 3.4: Generated engine-hardware compatibility table

**Files:**
- Create: `scripts/regenerate-compat-table.py`
- Create: `wiki/concepts/engine-hardware-compatibility.md`
- Create: `tests/test_regenerate_compat_table.py`

- [ ] **Step 1: Write the failing test**

`tests/test_regenerate_compat_table.py`:
```python
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
---
body
""")
    write(tmp_path / "wiki" / "hardware" / "b200.md", """---
title: "NVIDIA B200"
type: hardware
---
body
""")
    write(tmp_path / "wiki" / "hardware" / "mi300x.md", """---
title: "AMD MI300X"
type: hardware
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
```

- [ ] **Step 2: Run the test, expect failure**

```bash
python3 -m pytest tests/test_regenerate_compat_table.py -v
```

Expected: FAIL — script doesn't exist.

- [ ] **Step 3: Implement the script**

`scripts/regenerate-compat-table.py`:
```python
#!/usr/bin/env python3
"""Regenerate wiki/concepts/engine-hardware-compatibility.md from
engine pages' `supported_hardware:` frontmatter and the set of hardware
pages under wiki/hardware/.

Output is committed; the LINT operation calls this script and refuses
to commit if the file would change.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

FRONTMATTER_DELIM = "---"


def parse_frontmatter(text: str) -> dict | None:
    if not text.startswith(FRONTMATTER_DELIM + "\n"):
        return None
    rest = text[len(FRONTMATTER_DELIM) + 1:]
    end = rest.find("\n" + FRONTMATTER_DELIM + "\n")
    if end == -1:
        return None
    out = {}
    for line in rest[:end].splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        out[k.strip()] = v.strip()
    return out


def parse_list(s: str) -> list[str]:
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        return [x.strip() for x in s[1:-1].split(",") if x.strip()]
    return [x.strip() for x in s.split(",") if x.strip()]


def collect_engines(root: Path) -> dict[str, list[str]]:
    out = {}
    for md in sorted((root / "wiki" / "engines").glob("*.md")):
        fm = parse_frontmatter(md.read_text())
        if not fm or fm.get("type") != "engine":
            continue
        title = fm.get("title", md.stem).strip().strip('"')
        supported = parse_list(fm.get("supported_hardware", "[]"))
        out[title] = supported
    return out


def collect_hardware(root: Path) -> list[str]:
    out = []
    for md in sorted((root / "wiki" / "hardware").glob("*.md")):
        fm = parse_frontmatter(md.read_text())
        if not fm or fm.get("type") != "hardware":
            continue
        out.append(md.stem)
    return out


def render_table(engines: dict[str, list[str]], hardware: list[str]) -> str:
    head = "| Engine | " + " | ".join(hardware) + " |"
    sep = "|---|" + "|".join("---" for _ in hardware) + "|"
    rows = [head, sep]
    for engine in sorted(engines):
        cells = []
        for hw in hardware:
            cells.append("✓" if hw in engines[engine] else "✗")
        rows.append(f"| {engine} | " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n"


HEADER = """---
title: "Engine × Hardware Compatibility"
type: concept
tags: [generated, compatibility]
created: 2026-05-01
updated: 2026-05-01
---

> **Generated** by `scripts/regenerate-compat-table.py` from `wiki/engines/*.md`
> `supported_hardware:` frontmatter. **Do not edit by hand** — edits will be
> overwritten on the next `LINT`.

The orchestration loop's scheduler reads each engine's `supported_hardware`
field directly. This page is a human-readable readout of that data.

"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    args = p.parse_args()
    root = Path(args.root)

    engines = collect_engines(root)
    hardware = collect_hardware(root)
    if not engines or not hardware:
        print("no engines or hardware pages found", file=sys.stderr)
        sys.exit(2)

    out_dir = root / "wiki" / "concepts"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "engine-hardware-compatibility.md").write_text(HEADER + render_table(engines, hardware))

    print(f"wrote {out_dir / 'engine-hardware-compatibility.md'}")


if __name__ == "__main__":
    main()
```

```bash
chmod +x scripts/regenerate-compat-table.py
```

- [ ] **Step 4: Run the test, expect pass**

```bash
python3 -m pytest tests/test_regenerate_compat_table.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Generate the real table from the wiki**

```bash
python3 scripts/regenerate-compat-table.py
cat wiki/concepts/engine-hardware-compatibility.md
```

Expected: a 3-row table (vLLM, SGLang, TensorRT-LLM) × 3 columns (h100, b200, mi300x), with `✗` only in TensorRT-LLM × mi300x.

- [ ] **Step 6: Commit**

```bash
git add scripts/regenerate-compat-table.py wiki/concepts/engine-hardware-compatibility.md tests/test_regenerate_compat_table.py
git commit -m "Generated engine-hardware compatibility table"
```

---

## Slice 4: Host registry (no GPU)

The data plumbing the orchestration loop sits on. Pure Python, no SSH yet — `reachable` and `schedule` subcommands stub their network behavior.

### Task 4.1: Write tests for host_registry.py

**Files:**
- Create: `tests/test_host_registry.py`
- Create: `tests/fixtures/registry/hosts.toml`
- Create: `tests/fixtures/registry/host-state.toml`

- [ ] **Step 1: Create test fixtures**

`tests/fixtures/registry/hosts.toml`:
```toml
[hosts.h100-1]
ip        = "203.0.113.42"
user      = "ubuntu"
ssh_key   = "~/.ssh/lambda_key"
vendor    = "nvidia"
hardware  = "h100"
gpu_count = 8

[hosts.b200-1]
ip        = "203.0.113.99"
user      = "ubuntu"
ssh_key   = "~/.ssh/runpod_key"
vendor    = "nvidia"
hardware  = "b200"
gpu_count = 8

[hosts.mi300x-1]
ip        = "203.0.113.11"
user      = "ubuntu"
ssh_key   = "~/.ssh/tw_key"
vendor    = "amd"
hardware  = "mi300x"
gpu_count = 8
```

`tests/fixtures/registry/host-state.toml`:
```toml
[hosts.h100-1]
setup_state = "ready"

[hosts.b200-1]
setup_state = "pending"

[hosts.mi300x-1]
setup_state = "ready"
```

- [ ] **Step 2: Write the failing test file**

`tests/test_host_registry.py`:
```python
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
    # hyp.hardware=any, engine supports h100+b200+mi300x — pick first ready
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
    # hyp.hardware=amd, vLLM supports it — only mi300x-1 fits
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
```

- [ ] **Step 3: Run the tests, expect all to fail**

```bash
python3 -m pytest tests/test_host_registry.py -v
```

Expected: 10 failed (script does not exist).

### Task 4.2: Implement host_registry.py

**Files:**
- Create: `scripts/host_registry.py`

- [ ] **Step 1: Write the implementation**

`scripts/host_registry.py`:
```python
#!/usr/bin/env python3
"""Host registry helper for the autoresearch loop.

Reads:
  - .hosts.toml         user-edited list of provisioned hosts
  - .host-state.toml    driver-written setup/dispatch state

Writes:
  - .host-state.toml    via `state` and `state-error` subcommands

All TOML I/O is centralized here. Bash callers only see plain text
(one host name per line, or "none", or a single value).
"""
from __future__ import annotations
import argparse
import datetime as dt
import sys
import tomllib
from pathlib import Path

HOSTS_FILE = ".hosts.toml"
STATE_FILE = ".host-state.toml"


def _read(path: Path) -> dict:
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text())


def load_hosts(root: Path) -> dict:
    return _read(root / HOSTS_FILE).get("hosts", {})


def load_state(root: Path) -> dict:
    return _read(root / STATE_FILE).get("hosts", {})


def write_state(root: Path, state: dict) -> None:
    """TOML write — minimal, no external dep."""
    lines = []
    for name in sorted(state):
        lines.append(f"[hosts.{name}]")
        for k, v in state[name].items():
            if isinstance(v, str):
                # Escape backslashes and double quotes
                escaped = v.replace("\\", "\\\\").replace('"', '\\"')
                lines.append(f'{k} = "{escaped}"')
            else:
                lines.append(f"{k} = {v}")
        lines.append("")
    (root / STATE_FILE).write_text("\n".join(lines))


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def cmd_list(args, root: Path) -> int:
    hosts = load_hosts(root)
    state = load_state(root)
    if args.summary:
        for name in sorted(hosts):
            h = hosts[name]
            s = state.get(name, {}).get("setup_state", "pending")
            print(f"{name}\t{h.get('vendor', '?')}\t{h.get('hardware', '?')}\t{s}")
    else:
        for name in sorted(hosts):
            print(name)
    return 0


def synthetic(host: dict, key: str) -> str | None:
    if key == "ssh_target":
        return f"{host['user']}@{host['ip']}"
    return None


def cmd_get(args, root: Path) -> int:
    hosts = load_hosts(root)
    if args.host not in hosts:
        print(f"unknown host: {args.host}", file=sys.stderr)
        return 2
    h = hosts[args.host]
    syn = synthetic(h, args.field)
    if syn is not None:
        print(syn)
        return 0
    if args.field not in h:
        print(f"unknown field: {args.field}", file=sys.stderr)
        return 2
    print(h[args.field])
    return 0


def matches_hypothesis_hardware(host: dict, hyp_hw: str) -> bool:
    if hyp_hw == "any":
        return True
    if hyp_hw in {"nvidia", "amd"}:
        return host.get("vendor") == hyp_hw
    return host.get("hardware") == hyp_hw


def cmd_match(args, root: Path) -> int:
    hosts = load_hosts(root)
    matched = []
    for name in sorted(hosts):
        h = hosts[name]
        if args.hardware is not None:
            if matches_hypothesis_hardware(h, args.hardware):
                matched.append(name)
        elif args.vendor is not None:
            if h.get("vendor") == args.vendor:
                matched.append(name)
    for n in matched:
        print(n)
    return 0


def cmd_state(args, root: Path) -> int:
    state = load_state(root)
    if args.host not in state:
        state[args.host] = {}
    if args.set is not None:
        state[args.host]["setup_state"] = args.set
        state[args.host][f"setup_{'finished' if args.set == 'ready' else 'started'}"] = now_iso()
    if args.set_error is not None:
        state[args.host]["last_error"] = args.set_error
    write_state(root, state)
    return 0


def cmd_schedule(args, root: Path) -> int:
    hosts = load_hosts(root)
    state = load_state(root)
    engine_supported = set(args.engine_supported.split(","))
    excluded = set(args.exclude or [])
    hyp_hw = args.hypothesis_hardware

    for name in sorted(hosts):
        if name in excluded:
            continue
        if state.get(name, {}).get("setup_state") != "ready":
            continue
        h = hosts[name]
        if h.get("hardware") not in engine_supported:
            continue
        if not matches_hypothesis_hardware(h, hyp_hw):
            continue
        print(name)
        return 0

    print("none")
    return 0


def cmd_reachable(args, root: Path) -> int:
    """Stub for slice 4. Real ssh ping wired in slice 5."""
    import subprocess
    hosts = load_hosts(root)
    if args.host not in hosts:
        return 2
    h = hosts[args.host]
    target = f"{h['user']}@{h['ip']}"
    key = Path(h["ssh_key"]).expanduser()
    res = subprocess.run(
        [
            "ssh", "-i", str(key),
            "-o", "ConnectTimeout=10",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            target, "echo ok",
        ],
        capture_output=True, text=True, timeout=15,
    )
    return 0 if res.returncode == 0 and "ok" in res.stdout else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list")
    sp.add_argument("--summary", action="store_true")
    sp.set_defaults(fn=cmd_list)

    sp = sub.add_parser("get")
    sp.add_argument("host")
    sp.add_argument("field")
    sp.set_defaults(fn=cmd_get)

    sp = sub.add_parser("match")
    grp = sp.add_mutually_exclusive_group(required=True)
    grp.add_argument("--hardware")
    grp.add_argument("--vendor")
    sp.set_defaults(fn=cmd_match)

    sp = sub.add_parser("state")
    sp.add_argument("host")
    sp.add_argument("--set", choices=["pending", "running", "ready", "failed", "unreachable"])
    sp.add_argument("--set-error")
    sp.set_defaults(fn=cmd_state)

    sp = sub.add_parser("schedule")
    sp.add_argument("--hypothesis-hardware", required=True)
    sp.add_argument("--engine-supported", required=True)
    sp.add_argument("--exclude", action="append")
    sp.set_defaults(fn=cmd_schedule)

    sp = sub.add_parser("reachable")
    sp.add_argument("host")
    sp.set_defaults(fn=cmd_reachable)

    return p


def main():
    args = build_parser().parse_args()
    root = Path(args.root)
    sys.exit(args.fn(args, root) or 0)


if __name__ == "__main__":
    main()
```

```bash
chmod +x scripts/host_registry.py
```

- [ ] **Step 2: Run the tests, expect all to pass**

```bash
python3 -m pytest tests/test_host_registry.py -v
```

Expected: 10 passed.

- [ ] **Step 3: Commit**

```bash
git add scripts/host_registry.py tests/test_host_registry.py tests/fixtures/registry/
git commit -m "Add scripts/host_registry.py with TOML parsing and scheduling"
```

### Task 4.3: Add .hosts.example.toml and .gitignore entries

**Files:**
- Create: `.hosts.example.toml`
- Modify: `.gitignore`

- [ ] **Step 1: Create the example file**

`.hosts.example.toml`:
```toml
# Copy to .hosts.toml and add one [hosts.<name>] block per provisioned box.
# .hosts.toml is gitignored — your IPs and key paths stay on your Mac.
#
# Required fields per host:
#   ip         — the public IP or DNS name
#   user       — the SSH user (usually `ubuntu` or `root`)
#   ssh_key    — path to the private key on your Mac (~ is expanded)
#   vendor     — "nvidia" or "amd"
#   hardware   — slug matching wiki/hardware/<slug>.md
#                  (e.g. "h100", "h200", "b200", "mi300x", "mi325x")
#   gpu_count  — number of GPUs in the box
# Optional:
#   notes      — free-text reminder for you (cloud, hourly cost, etc.)

[hosts.h100-1]
ip        = "203.0.113.42"
user      = "ubuntu"
ssh_key   = "~/.ssh/lambda_key"
vendor    = "nvidia"
hardware  = "h100"
gpu_count = 8
notes     = "Lambda Cloud, 8xH100 SXM5"

# [hosts.b200-1]
# ip        = "203.0.113.99"
# user      = "ubuntu"
# ssh_key   = "~/.ssh/runpod_key"
# vendor    = "nvidia"
# hardware  = "b200"
# gpu_count = 8

# [hosts.mi300x-1]
# ip        = "203.0.113.11"
# user      = "ubuntu"
# ssh_key   = "~/.ssh/tensorwave_key"
# vendor    = "amd"
# hardware  = "mi300x"
# gpu_count = 8
```

- [ ] **Step 2: Update `.gitignore`**

Append:

```
# Mac-driver host registry — local config, never commit
.hosts.toml
.host-state.toml
```

- [ ] **Step 3: Commit**

```bash
git add .hosts.example.toml .gitignore
git commit -m "Add .hosts.example.toml and gitignore registry files"
```

### Task 4.4: Write docs/host-onboarding.md

**Files:**
- Create: `docs/host-onboarding.md`

- [ ] **Step 1: Write the doc**

`docs/host-onboarding.md`:
```markdown
# Host Onboarding

How to add a fresh GPU box to the autoresearch loop.

## Prerequisites

You've rented a GPU instance from a cloud provider (Lambda Cloud, RunPod, Crusoe, TensorWave, etc.) and have:
- A public IP or DNS name
- An SSH user (usually `ubuntu` or `root`)
- A private key file on your Mac
- The image is Ubuntu 22.04+ with NVIDIA drivers (or ROCm for AMD)

## Add the box to your registry

If this is your first box:

```bash
cp .hosts.example.toml .hosts.toml
```

Edit `.hosts.toml` and add a block per host:

```toml
[hosts.h100-1]
ip        = "203.0.113.42"
user      = "ubuntu"
ssh_key   = "~/.ssh/lambda_key"
vendor    = "nvidia"             # or "amd"
hardware  = "h100"               # slug from wiki/hardware/
gpu_count = 8
```

The `hardware` slug **must** match a page under `wiki/hardware/<slug>.md`. The currently supported slugs are `h100`, `b200`, `mi300x`. To add a new one, also create a wiki page for it.

## Verify the box is reachable

```bash
python3 scripts/host_registry.py reachable h100-1
echo $?    # 0 = reachable, 1 = unreachable
```

## Run setup remotely

The first time the loop dispatches a round to a host, it will SSH in and run setup automatically. To trigger setup eagerly without running a round:

```bash
./scripts/remote-setup.sh h100-1
```

This SSH bootstrap clones the repo, detects the vendor (`nvidia-smi` vs `rocm-smi`), and runs the appropriate setup script (`setup-cuda.sh` or `setup-rocm.sh`). The bootstrap is mechanical — no Claude session involved.

Setup state is tracked in `.host-state.toml` (gitignored). Possible states:
- `pending` — bootstrap not yet run
- `running` — bootstrap in progress
- `ready` — bootstrap done; host can take rounds
- `failed` — bootstrap errored; check `last_error` and rerun
- `unreachable` — SSH failed reachability check; check IP / key / firewall

## What the box ends up with

After successful setup:
- Repo cloned at `~/llm-serving-autoresearch-wiki` (the bootstrap uses `git clone`)
- Engines installed: `vllm`, `sglang`, plus `tensorrt-llm` on NVIDIA only
- HF cache warmed for the model in `.env` (if `HF_TOKEN` is set)
- The box never reads or writes the wiki — it only emits `raw/benchmarks/<run>/` artifacts which the Mac rsync's back

## Replacing or destroying a box

When you tear down a box, just remove its block from `.hosts.toml` (and optionally `rm` the corresponding entry from `.host-state.toml`). The next loop run will simply not dispatch to it.

If a box's IP changes (cloud reboot), update `.hosts.toml`. The state stays valid — only the routing changed.

## Wiki write conflicts (FYI)

If you edit `wiki/index.md` (or any other wiki file) while a loop is running, the round's `git commit` step will fail and the loop will halt cleanly with a conflict message. Resolve the conflict, then run `./run_loop.sh --resume` to continue.
```

- [ ] **Step 2: Commit**

```bash
git add docs/host-onboarding.md
git commit -m "Add docs/host-onboarding.md"
```

---

## Slice 5: Setup dispatcher and remote bootstrap (no GPU)

Refactor `setup.sh` into a vendor dispatcher; add the bash bootstrap that runs on the remote box.

### Task 5.1: Move setup.sh body to scripts/setup-cuda.sh

**Files:**
- Create: `scripts/setup-cuda.sh` (verbatim copy of current `setup.sh`)
- Modify: `setup.sh` (rewritten as dispatcher)

- [ ] **Step 1: Move the existing setup.sh to scripts/setup-cuda.sh**

```bash
mkdir -p scripts
cp setup.sh scripts/setup-cuda.sh
chmod +x scripts/setup-cuda.sh
```

- [ ] **Step 2: Rewrite setup.sh as a dispatcher**

Replace `setup.sh` contents:

```bash
#!/usr/bin/env bash
# Top-level setup dispatcher. Detects vendor and delegates.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

detect_vendor() {
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q GPU; then
        echo nvidia; return
    fi
    if command -v rocm-smi >/dev/null 2>&1 && rocm-smi --showid 2>/dev/null | grep -q GPU; then
        echo amd; return
    fi
    echo "ERROR: no NVIDIA or AMD GPU detected (no nvidia-smi or rocm-smi)" >&2
    exit 1
}

vendor="$(detect_vendor)"
echo "Detected vendor: $vendor"

case "$vendor" in
    nvidia) exec bash "$SCRIPT_DIR/scripts/setup-cuda.sh" "$@" ;;
    amd)    exec bash "$SCRIPT_DIR/scripts/setup-rocm.sh" "$@" ;;
    *)      echo "ERROR: unknown vendor: $vendor" >&2; exit 1 ;;
esac
```

```bash
chmod +x setup.sh
```

- [ ] **Step 3: Verify dispatcher syntax**

```bash
bash -n setup.sh
bash -n scripts/setup-cuda.sh
```

Expected: no output (syntax OK).

- [ ] **Step 4: Smoke-test on Mac (will fail vendor-detection — expected)**

```bash
./setup.sh 2>&1 | head -3
echo "exit: $?"
```

Expected: prints "ERROR: no NVIDIA or AMD GPU detected" and exits 1.

- [ ] **Step 5: Commit**

```bash
git add setup.sh scripts/setup-cuda.sh
git commit -m "Refactor setup.sh into vendor-detecting dispatcher"
```

### Task 5.2: Create scripts/setup-rocm.sh stub

**Files:**
- Create: `scripts/setup-rocm.sh`

- [ ] **Step 1: Create the stub**

`scripts/setup-rocm.sh`:
```bash
#!/usr/bin/env bash
# AMD/ROCm setup — stub; full implementation in slice 8.
set -euo pipefail

echo "FAIL=rocm_not_implemented" >&2
echo "ROCm setup is not implemented in this slice. See slice 8 of" >&2
echo "docs/superpowers/plans/2026-05-01-mac-driver-multi-vendor-plan.md" >&2
exit 64
```

```bash
chmod +x scripts/setup-rocm.sh
```

- [ ] **Step 2: Verify**

```bash
bash -n scripts/setup-rocm.sh
./scripts/setup-rocm.sh; echo "exit: $?"
```

Expected: exit 64 with the FAIL message on stderr.

- [ ] **Step 3: Commit**

```bash
git add scripts/setup-rocm.sh
git commit -m "Add scripts/setup-rocm.sh stub (slice 8 fills in)"
```

### Task 5.3: Create scripts/remote-bootstrap.sh

This is the body that runs **on the remote box**, fed in via `ssh ... 'bash -s' < remote-bootstrap.sh`. It is meant to be self-contained and idempotent.

**Files:**
- Create: `scripts/remote-bootstrap.sh`

- [ ] **Step 1: Write the bootstrap script**

`scripts/remote-bootstrap.sh`:
```bash
#!/usr/bin/env bash
# Runs on the remote GPU box, executed via:
#   ssh "$user@$ip" 'bash -s' < scripts/remote-bootstrap.sh
#
# Output convention: prints DONE on success, FAIL=<reason> on failure
# (caller greps for these on stdout).
#
# Idempotent: rerunnable. Picks up where it left off.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/PrimaLabs-AI/llm-serving-autoresearch-wiki}"
REPO_DIR="${REPO_DIR:-$HOME/llm-serving-autoresearch-wiki}"
BRANCH="${BRANCH:-mac-driver-multi-vendor}"

step() { echo ">> $*"; }
fail() { echo "FAIL=$1" >&2; exit 1; }

step "ensure git is installed"
if ! command -v git >/dev/null 2>&1; then
    sudo apt-get update -y >/dev/null 2>&1 || true
    sudo apt-get install -y git || fail "git_install"
fi

step "ensure repo is checked out at $REPO_DIR"
if [ ! -d "$REPO_DIR/.git" ]; then
    git clone "$REPO_URL" "$REPO_DIR" || fail "git_clone"
fi
cd "$REPO_DIR"
git fetch --all --prune || fail "git_fetch"
git checkout "$BRANCH" || fail "git_checkout"
git pull --ff-only origin "$BRANCH" || fail "git_pull"

step "verify .env exists"
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "  (copied .env.example to .env — you may need to add HF_TOKEN)"
    else
        fail "no_env_file"
    fi
fi
# Source .env so later steps see HF_TOKEN, MODEL
set -a; source .env; set +a

step "run setup.sh"
./setup.sh || fail "setup"

step "warm HF cache for ${MODEL:-<unset>}"
if [ -n "${MODEL:-}" ]; then
    if [ -n "${HF_TOKEN:-}" ]; then
        export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    fi
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')" || fail "hf_warm_${MODEL//\//_}"
fi

echo "DONE"
```

```bash
chmod +x scripts/remote-bootstrap.sh
```

- [ ] **Step 2: Verify syntax**

```bash
bash -n scripts/remote-bootstrap.sh
```

Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add scripts/remote-bootstrap.sh
git commit -m "Add scripts/remote-bootstrap.sh remote install runner"
```

### Task 5.4: Create scripts/remote-setup.sh wrapper

**Files:**
- Create: `scripts/remote-setup.sh`

- [ ] **Step 1: Write the wrapper**

`scripts/remote-setup.sh`:
```bash
#!/usr/bin/env bash
# Mac-side wrapper that ships scripts/remote-bootstrap.sh to a host
# and runs it. Updates .host-state.toml around the dispatch.
#
# Usage: ./scripts/remote-setup.sh <host-name>

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <host-name>" >&2
    exit 2
fi

HOST="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Resolve host details
ssh_target="$(python3 "$SCRIPT_DIR/host_registry.py" --root "$REPO_DIR" get "$HOST" ssh_target)"
ssh_key="$(python3 "$SCRIPT_DIR/host_registry.py" --root "$REPO_DIR" get "$HOST" ssh_key)"
ssh_key_expanded="${ssh_key/#\~/$HOME}"

echo "[remote-setup] $HOST → $ssh_target"
python3 "$SCRIPT_DIR/host_registry.py" --root "$REPO_DIR" state "$HOST" --set running

# Stream the bootstrap to the box; capture stdout and stderr separately
out_file="$(mktemp)"
err_file="$(mktemp)"
trap 'rm -f "$out_file" "$err_file"' EXIT

set +e
ssh -i "$ssh_key_expanded" \
    -o StrictHostKeyChecking=accept-new \
    -o ControlMaster=auto \
    -o ControlPath="$HOME/.ssh/cm-%r@%h:%p" \
    -o ControlPersist=10m \
    "$ssh_target" 'bash -s' < "$SCRIPT_DIR/remote-bootstrap.sh" \
    > "$out_file" 2> "$err_file"
rc=$?
set -e

cat "$out_file"
cat "$err_file" >&2

if [ $rc -eq 0 ] && grep -q "^DONE$" "$out_file"; then
    python3 "$SCRIPT_DIR/host_registry.py" --root "$REPO_DIR" state "$HOST" --set ready
    echo "[remote-setup] $HOST ready"
    exit 0
fi

reason="$(grep -m1 '^FAIL=' "$err_file" "$out_file" 2>/dev/null | sed 's/.*FAIL=//' | head -1)"
python3 "$SCRIPT_DIR/host_registry.py" --root "$REPO_DIR" state "$HOST" --set failed
python3 "$SCRIPT_DIR/host_registry.py" --root "$REPO_DIR" state "$HOST" --set-error "${reason:-unknown}"
echo "[remote-setup] $HOST failed: ${reason:-unknown}" >&2
exit 1
```

```bash
chmod +x scripts/remote-setup.sh
```

- [ ] **Step 2: Verify syntax**

```bash
bash -n scripts/remote-setup.sh
```

- [ ] **Step 3: Commit**

```bash
git add scripts/remote-setup.sh
git commit -m "Add scripts/remote-setup.sh Mac-side bootstrap wrapper"
```

---

## Slice 6: Mac-driver run_loop.sh + prompts (GPU required to validate end-to-end)

The conductor. Coordinates reachability, setup, PICK, RUN, lint, commit. The two prompt files are the only natural-language piece of the design.

### Task 6.1: Write prompts/pick.md

**Files:**
- Create: `prompts/pick.md`

- [ ] **Step 1: Create the prompt**

`prompts/pick.md`:
```markdown
You are the picker step of the autoresearch loop.

Your task: read `wiki/index.md`, examine the open ranked hypotheses, and pick exactly one that can be scheduled given the available hosts.

# Inputs you receive

The user message contains:
- `round=<N>` — current round number (informational)
- `model=<HF-id>` — the model under optimization
- `registry_summary=…` — output of `scripts/host_registry.py list --summary`, one host per line: `name<TAB>vendor<TAB>hardware<TAB>setup_state`
- `excluded=<slug>,<slug>,…` — optional; hypotheses already shown to be unschedulable in this round

# What to do

1. Read `wiki/index.md` to find the ranked open hypotheses table.
2. For each hypothesis (top-down), read its page to learn its `hardware:` and `engine:` frontmatter values, then read the engine page to learn `supported_hardware:`.
3. Pick the highest-ranked hypothesis where:
   - status is `open`
   - hypothesis slug is **not** in `excluded`
   - the intersection (`hypothesis.hardware` ∩ `engine.supported_hardware` ∩ host registry's reachable+ready hosts) is non-empty
4. Print exactly one line on stdout:
   ```
   HYPOTHESIS=<slug>
   ```
   Use the slug from the file name (without `.md`).
5. If no hypothesis is schedulable, print:
   ```
   HYPOTHESIS=none
   ```
   On stderr, briefly explain why (e.g., "all open hypotheses target h100; only b200-1 ready").

# Constraints

- Only `HYPOTHESIS=<slug>` or `HYPOTHESIS=none` may go to stdout. Bash greps for it.
- Reasoning, exploration, tool output — all to stderr.
- Do not write to the wiki.
- Do not start a benchmark. The RUN step does that.

# Tools you have

- `Read`, `Bash` (with `grep`, `cat`, `python3 scripts/host_registry.py …`)

The system prompt for the wiki schema (`SCHEMA.md`) is already loaded — you know what hypothesis pages look like.
```

- [ ] **Step 2: Commit**

```bash
mkdir -p prompts
git add prompts/pick.md
git commit -m "Add prompts/pick.md for PICK turn of the loop"
```

### Task 6.2: Write prompts/run.md

**Files:**
- Create: `prompts/run.md`

- [ ] **Step 1: Create the prompt**

`prompts/run.md`:
```markdown
You are the runner step of the autoresearch loop.

Your task: SSH to a remote GPU host, run a benchmark, pull the artifacts back, and write a structured experiment page to `wiki/experiments/<run_slug>.md`.

# Inputs you receive

The user message contains:
- `hypothesis=<slug>` — the hypothesis you are testing
- `host=<host-name>` — the registered host to use (from .hosts.toml)
- `run_slug=<YYYY-MM-DD>-<short-name>` — used for output path naming
- `model=<HF-id>` — the model to serve

# What to do

1. **Read** the hypothesis page, the engine page (named in hypothesis frontmatter), the workload page (also named there), the host page (`wiki/hardware/<hardware-slug>.md`), and the model page if one exists.

2. **Compose the engine config diff** — what changes from baseline. Keep this as a JSON object that `benchmark_harness.py --config` accepts.

3. **Resolve host details:**
   ```bash
   ssh_target=$(python3 scripts/host_registry.py get $host ssh_target)
   ssh_key=$(python3 scripts/host_registry.py get $host ssh_key)
   ```

4. **Sync the branch on the remote box:**
   ```bash
   ssh -i "$ssh_key" "$ssh_target" \
       "cd ~/llm-serving-autoresearch-wiki && git fetch && git checkout $(git rev-parse --abbrev-ref HEAD) && git pull --ff-only"
   ```

5. **Run the benchmark on the box:**
   ```bash
   ssh -i "$ssh_key" "$ssh_target" \
       "cd ~/llm-serving-autoresearch-wiki && \
        python benchmark_harness.py \
          --engine <engine> --model <model> \
          --workload <workload> \
          --config '<json>' \
          --output-dir raw/benchmarks/<run_slug> \
          --launch-server"
   ```

6. **Pull the artifacts back to the Mac:**
   ```bash
   rsync -avz -e "ssh -i $ssh_key" \
       "${ssh_target}:~/llm-serving-autoresearch-wiki/raw/benchmarks/<run_slug>/" \
       "raw/benchmarks/<run_slug>/"
   ```

7. **Write the experiment page** at `wiki/experiments/<run_slug>.md` per the SCHEMA `experiment` template. Required frontmatter:
   - `hypothesis: <slug>`
   - `model: <model-slug>`
   - `engine: <engine-slug>`
   - `workload: <workload-slug>`
   - `hardware: <slug>` (must match the host's `hardware:` field)
   - `host: <host-name>`
   - `commit: <model-repo-sha-or-engine-sha>`
   - `verdict: supported | refuted | inconclusive | invalid`

   Required H2 sections per SCHEMA: Hypothesis under test, Setup (full command + diff from baseline), Baseline comparison, Results (metrics table), Profile / Benchmark, Observations, Verdict + reasoning, Next hypotheses.

   The Profile / Benchmark section MUST cite `raw/benchmarks/<run_slug>/` as a relative markdown link, and again under `## Sources`.

8. **Update the hypothesis page**: change `status:` to match the verdict; link the experiment page.

9. **Update `wiki/index.md`** (move hypothesis from open to refuted/supported; add experiment to the experiments list) and `wiki/log.md` (newest entry on top).

10. **Print the result on stdout, exactly two lines:**
    ```
    EXPERIMENT=wiki/experiments/<run_slug>.md
    VERDICT=<verdict>
    ```

# If the benchmark crashes

- Capture stderr to the experiment page's Profile/Benchmark section.
- Set `verdict: invalid`.
- Still print `EXPERIMENT=…` and `VERDICT=invalid` on stdout. Do not exit non-zero — bash will see VERDICT=invalid and treat it as a clean round.

# Constraints

- Only `EXPERIMENT=<path>` and `VERDICT=<verdict>` may go to stdout.
- All other output (commands, ssh transcript, reasoning) → stderr.
- Do not commit. Bash will commit after this step.
```

- [ ] **Step 2: Commit**

```bash
git add prompts/run.md
git commit -m "Add prompts/run.md for RUN turn of the loop"
```

### Task 6.3: Write scripts/lint-experiment-page.sh

**Files:**
- Create: `scripts/lint-experiment-page.sh`

- [ ] **Step 1: Write the linter**

`scripts/lint-experiment-page.sh`:
```bash
#!/usr/bin/env bash
# Lints a single experiment page for required frontmatter and structure.
# Exit 0 if clean, 1 if any issue is found (with errors on stderr).
#
# Usage: ./scripts/lint-experiment-page.sh wiki/experiments/<run_slug>.md

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <experiment-page>" >&2
    exit 2
fi

PAGE="$1"

if [ ! -f "$PAGE" ]; then
    echo "lint: file not found: $PAGE" >&2
    exit 1
fi

errors=0
err() { echo "lint: $PAGE: $*" >&2; errors=$((errors + 1)); }

# Required frontmatter fields
for key in title type tags hypothesis model engine workload hardware host verdict; do
    if ! grep -qE "^${key}:" "$PAGE"; then
        err "missing frontmatter: $key"
    fi
done

# Frontmatter `type:` must be `experiment`
if ! grep -qE '^type:\s*experiment' "$PAGE"; then
    err "type must be 'experiment'"
fi

# Verdict must be one of the allowed values
verdict_line="$(grep -E '^verdict:' "$PAGE" || true)"
case "$verdict_line" in
    *supported*|*refuted*|*inconclusive*|*invalid*) : ;;
    *) err "verdict must be one of: supported|refuted|inconclusive|invalid (got: ${verdict_line:-<missing>})" ;;
esac

# Required H2 sections
for h2 in "Hypothesis under test" "Setup" "Results" "Verdict" "Sources"; do
    if ! grep -qE "^## ${h2}" "$PAGE"; then
        err "missing H2 section: $h2"
    fi
done

# If verdict != invalid, the Profile / Benchmark section must exist
if ! grep -qE '^verdict:\s*invalid' "$PAGE"; then
    if ! grep -qE '^## Profile' "$PAGE"; then
        err "non-invalid verdict requires '## Profile / Benchmark' section"
    fi
fi

# Profile path must reference raw/benchmarks/ or raw/profiles/
if grep -qE '^## Profile' "$PAGE"; then
    if ! grep -qE 'raw/(benchmarks|profiles)/' "$PAGE"; then
        err "Profile section must cite a path under raw/benchmarks/ or raw/profiles/"
    fi
fi

if [ "$errors" -gt 0 ]; then
    echo "lint: $errors error(s)" >&2
    exit 1
fi
exit 0
```

```bash
chmod +x scripts/lint-experiment-page.sh
```

- [ ] **Step 2: Test the linter against an existing experiment**

```bash
./scripts/lint-experiment-page.sh wiki/experiments/llama3_8B_autoresearch_optimization/torchax/experiments/2026-04-25-baseline.md
```

This will likely report missing fields (`hardware:`, `host:`) until task 3.3 added them. If migration ran, it should pass cleanly. Investigate any reports — they may indicate real lint errors in legacy pages, in which case we annotate `# nolint` or fix.

- [ ] **Step 3: Commit**

```bash
git add scripts/lint-experiment-page.sh
git commit -m "Add scripts/lint-experiment-page.sh"
```

### Task 6.4: Rewrite run_loop.sh

**Files:**
- Modify: `run_loop.sh` (full rewrite)

- [ ] **Step 1: Replace run_loop.sh with the Mac-driver version**

```bash
#!/usr/bin/env bash
# Mac-side autoresearch loop driver.
#
# Per round:
#   1. reachability sweep (ssh ping all in-scope hosts)
#   2. setup pass (run remote-bootstrap.sh on any pending/failed hosts)
#   3. PICK turn (claude --print prints HYPOTHESIS=<slug>)
#   4. schedule (host_registry.py schedule …)
#   5. RUN turn (claude --print does ssh+benchmark+rsync+writeup)
#   6. lint experiment page
#   7. git commit wiki + raw/benchmarks/<run>
#
# Usage:
#   ./run_loop.sh --rounds 5 [--hosts h100-1,b200-1] [--model <id>] [--tag <name>]
#   ./run_loop.sh --resume         # continue after a halt (experimental)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATE="$(date +%Y-%m-%d)"

# Defaults
ROUNDS=5
HOSTS_FILTER=""
MODEL=""
TAG="loop"
RESUME=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --rounds) shift; ROUNDS="$1" ;;
        --hosts)  shift; HOSTS_FILTER="$1" ;;
        --model)  shift; MODEL="$1" ;;
        --tag)    shift; TAG="$1" ;;
        --resume) RESUME=true ;;
        *)        echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

LOG_DIR="$SCRIPT_DIR/raw/loops"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${DATE}-${TAG}.log"

# Convenience: log to file and stderr
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG_FILE" >&2; }

# Resolve in-scope hosts
all_hosts=()
mapfile -t all_hosts < <(python3 "$SCRIPT_DIR/scripts/host_registry.py" list)
if [ -z "$HOSTS_FILTER" ]; then
    in_scope=("${all_hosts[@]}")
else
    IFS=',' read -ra in_scope <<< "$HOSTS_FILTER"
fi

if [ ${#in_scope[@]} -eq 0 ]; then
    log "ERROR: no hosts in scope. Edit .hosts.toml (see docs/host-onboarding.md)."
    exit 2
fi

# Source MODEL from .env if not set on command line
if [ -z "$MODEL" ] && [ -f "$SCRIPT_DIR/.env" ]; then
    set -a; source "$SCRIPT_DIR/.env"; set +a
fi
if [ -z "${MODEL:-}" ]; then
    log "ERROR: --model is required (or set MODEL in .env)"
    exit 2
fi

reachability_sweep() {
    log "reachability sweep: ${in_scope[*]}"
    for h in "${in_scope[@]}"; do
        if python3 "$SCRIPT_DIR/scripts/host_registry.py" reachable "$h" >/dev/null 2>&1; then
            log "  $h: reachable"
        else
            log "  $h: UNREACHABLE"
            python3 "$SCRIPT_DIR/scripts/host_registry.py" state "$h" --set unreachable
        fi
    done
}

setup_pass() {
    log "setup pass"
    for h in "${in_scope[@]}"; do
        local state
        state="$(python3 "$SCRIPT_DIR/scripts/host_registry.py" list --summary | awk -v n="$h" '$1==n {print $4}')"
        case "$state" in
            ready) log "  $h: ready" ;;
            pending|failed)
                log "  $h: $state → bootstrapping"
                if "$SCRIPT_DIR/scripts/remote-setup.sh" "$h" >>"$LOG_FILE" 2>&1; then
                    log "  $h: bootstrap OK"
                else
                    log "  $h: bootstrap FAILED (see log)"
                fi
                ;;
            running) log "  $h: $state (concurrent run? skipping)" ;;
            unreachable) log "  $h: unreachable; skipping setup" ;;
        esac
    done
}

# Returns the hypothesis slug picked, or "none"
pick_turn() {
    local round="$1"
    local excluded="$2"
    local registry_summary
    registry_summary="$(python3 "$SCRIPT_DIR/scripts/host_registry.py" list --summary)"
    local user_msg
    user_msg=$(cat <<EOF
round=$round
model=$MODEL
excluded=${excluded:-}
registry_summary:
$registry_summary
EOF
)
    local out
    out="$(claude --print \
        --append-system-prompt "$(cat "$SCRIPT_DIR/prompts/pick.md")" \
        "$user_msg" 2>>"$LOG_FILE")"
    echo "$out" | grep -E '^HYPOTHESIS=' | head -1 | sed 's/^HYPOTHESIS=//'
}

# Returns "EXPERIMENT=<path>|VERDICT=<v>" on success, empty on failure
run_turn() {
    local hyp="$1"
    local host="$2"
    local run_slug="${DATE}-${hyp}"
    local user_msg
    user_msg=$(cat <<EOF
hypothesis=$hyp
host=$host
run_slug=$run_slug
model=$MODEL
EOF
)
    local out
    out="$(claude --print \
        --append-system-prompt "$(cat "$SCRIPT_DIR/prompts/run.md")" \
        "$user_msg" 2>>"$LOG_FILE")"
    local exp_line verdict_line
    exp_line="$(echo "$out" | grep -E '^EXPERIMENT=' | head -1)"
    verdict_line="$(echo "$out" | grep -E '^VERDICT=' | head -1)"
    if [ -n "$exp_line" ] && [ -n "$verdict_line" ]; then
        echo "${exp_line}|${verdict_line}"
    fi
}

run_round() {
    local round="$1"
    log "════ round $round / $ROUNDS ════"

    reachability_sweep
    setup_pass

    local excluded=""
    local hyp host scheduled
    local pick_attempts=0
    while [ "$pick_attempts" -lt 3 ]; do
        pick_attempts=$((pick_attempts + 1))
        hyp="$(pick_turn "$round" "$excluded")"
        if [ -z "$hyp" ] || [ "$hyp" = "none" ]; then
            log "  PICK: none (attempt $pick_attempts)"
            return 1
        fi

        # Read engine + hardware from hypothesis frontmatter
        local hyp_path="wiki/hypotheses/$hyp.md"
        if [ ! -f "$hyp_path" ]; then
            log "  PICK returned non-existent slug: $hyp"
            excluded="${excluded}${excluded:+,}$hyp"
            continue
        fi
        local hyp_hardware engine engine_supported
        hyp_hardware="$(awk '/^---$/{c++} c==1 && /^hardware:/{print $2; exit}' "$hyp_path")"
        engine="$(awk '/^---$/{c++} c==1 && /^engine:/{print $2; exit}' "$hyp_path")"
        if [ -z "$hyp_hardware" ] || [ -z "$engine" ]; then
            log "  hypothesis missing hardware: or engine: frontmatter"
            excluded="${excluded}${excluded:+,}$hyp"
            continue
        fi
        engine_supported="$(awk '/^---$/{c++} c==1 && /^supported_hardware:/{
            sub(/^supported_hardware:[[:space:]]*\[/, ""); sub(/\][[:space:]]*$/, ""); gsub(/[[:space:]]/, ""); print; exit
        }' "wiki/engines/$engine.md")"

        scheduled="$(python3 "$SCRIPT_DIR/scripts/host_registry.py" schedule \
            --hypothesis-hardware "$hyp_hardware" \
            --engine-supported "$engine_supported")"
        if [ "$scheduled" = "none" ]; then
            log "  schedule: $hyp unschedulable on current registry; excluding & retrying"
            excluded="${excluded}${excluded:+,}$hyp"
            continue
        fi

        host="$scheduled"
        log "  PICK: $hyp → host $host (attempt $pick_attempts)"
        break
    done

    if [ -z "${host:-}" ]; then
        log "  no schedulable hypothesis after 3 attempts; ending round"
        return 1
    fi

    log "  RUN: $hyp on $host"
    local result
    result="$(run_turn "$hyp" "$host")"
    if [ -z "$result" ]; then
        log "  RUN: malformed output; retrying once with stricter prompt"
        result="$(run_turn "$hyp" "$host")"
    fi
    if [ -z "$result" ]; then
        log "  RUN: failed twice; aborting round"
        return 1
    fi

    local exp_path verdict
    exp_path="$(echo "$result" | sed 's/|.*//; s/^EXPERIMENT=//')"
    verdict="$(echo "$result"  | sed 's/.*|//;  s/^VERDICT=//')"
    log "  result: $exp_path verdict=$verdict"

    if ! "$SCRIPT_DIR/scripts/lint-experiment-page.sh" "$exp_path" >>"$LOG_FILE" 2>&1; then
        log "  lint FAILED for $exp_path; aborting round (page left in place)"
        return 1
    fi

    git add wiki/ raw/benchmarks/ >>"$LOG_FILE" 2>&1
    if ! git commit -m "round $round: $hyp on $host ($verdict)" >>"$LOG_FILE" 2>&1; then
        log "  git commit FAILED (probable wiki edit conflict); halting"
        return 2
    fi
    log "  committed"
    return 0
}

log "════════════════ loop start ════════════════"
log "rounds=$ROUNDS hosts=${in_scope[*]} model=$MODEL tag=$TAG"

for r in $(seq 1 "$ROUNDS"); do
    set +e
    run_round "$r"
    rc=$?
    set -e
    case $rc in
        0) ;;  # ok
        1) log "round $r ended without commit; continuing"; ;;
        2) log "halting loop due to wiki conflict; resolve and rerun with --resume"; exit 2; ;;
        *) log "round $r exited with $rc; continuing"; ;;
    esac
done

log "════════════════ loop end ════════════════"
```

```bash
chmod +x run_loop.sh
```

- [ ] **Step 2: Verify syntax**

```bash
bash -n run_loop.sh
```

- [ ] **Step 3: Smoke test the help / no-host branch**

```bash
./run_loop.sh --rounds 1 --tag smoke 2>&1 | tail -5
```

Expected: errors out cleanly with "no hosts in scope" or "MODEL is required" — exercises the validation paths without touching SSH.

- [ ] **Step 4: Commit**

```bash
git add run_loop.sh
git commit -m "Rewrite run_loop.sh as Mac-driver conductor"
```

### Task 6.5: First end-to-end smoke test (requires GPU box)

**Files:**
- (no code changes; verifies slices 1-6 together)

This task is run after the user has provisioned an H100 box and added it to `.hosts.toml`.

- [ ] **Step 1: Onboard one H100 box**

Per `docs/host-onboarding.md`:

```bash
cp .hosts.example.toml .hosts.toml
# Edit .hosts.toml: add a single [hosts.h100-1] block
python3 scripts/host_registry.py reachable h100-1
```

- [ ] **Step 2: Trigger setup eagerly to surface errors before the loop**

```bash
./scripts/remote-setup.sh h100-1
```

Expected: prints `[remote-setup] h100-1 ready` after setup completes (~5-15 minutes depending on engine install time).

- [ ] **Step 3: Run a 1-round loop**

```bash
./run_loop.sh --rounds 1 --tag smoke-h100 --model meta-llama/Meta-Llama-3-8B-Instruct
```

Expected:
- log file under `raw/loops/<date>-smoke-h100.log`
- one new commit on the branch with one experiment page
- `wiki/experiments/<date>-<slug>.md` exists, lints clean
- `raw/benchmarks/<date>-<slug>/metrics.json` exists with non-trivial throughput numbers

- [ ] **Step 4: Inspect the result**

```bash
git log --oneline -3
cat wiki/experiments/<date>-<slug>.md | head -40
cat raw/benchmarks/<date>-<slug>/metrics.json
```

- [ ] **Step 5: If all clean, write a slice-6 completion note in `wiki/log.md`**

```markdown
## [<date>] manual | slice-6 smoke green

**Op**: manual
**Pages updated**: wiki/log.md
**Key result**: First end-to-end Mac-driver round green on h100-1 (commit <sha>). Loop, registry, prompts, lint, commit verified.
**Notes**: Mac-driver multi-vendor implementation slices 1-6 complete.
```

```bash
git add wiki/log.md
git commit -m "Slice 6 smoke green on h100-1"
```

---

## Slice 7: Multi-host serial smoke test (GPU required)

No code changes. Verifies the host-matching plumbing works across multiple hosts.

### Task 7.1: Add a second host and run a multi-round loop

- [ ] **Step 1: Add a B200 host to .hosts.toml**

Edit `.hosts.toml`:

```toml
[hosts.b200-1]
ip        = "<your-b200-ip>"
user      = "ubuntu"
ssh_key   = "~/.ssh/<your-key>"
vendor    = "nvidia"
hardware  = "b200"
gpu_count = 8
```

- [ ] **Step 2: Trigger setup**

```bash
./scripts/remote-setup.sh b200-1
```

- [ ] **Step 3: Add a B200-targeted seed hypothesis**

If the open hypotheses don't yet cover B200, create one:

`wiki/hypotheses/fp8-b200-multi-turn-agentic.md`:
```markdown
---
title: "FP8 inference on B200 vs H100 for multi-turn agentic"
type: hypothesis
tags: [fp8, b200, comparison]
model: llama-3-8b
engine: vllm
workload: multi-turn-agentic
hardware: b200
status: open
expected_gain: "1.8-2.5x throughput vs H100 FP8"
confidence: medium
effort: S
origin: human
created: <today>
updated: <today>
---

## Statement

vLLM FP8 inference on B200 yields ≥ 1.8× throughput vs the same configuration on H100 for the multi-turn-agentic workload at concurrency=64.

## Rationale

Blackwell's 5th-gen Tensor Cores deliver 2.0× FP8 peak FLOPs/SM and 2.5× per-chip vs Hopper. HBM bandwidth is 2.4× higher (8 TB/s vs 3.35 TB/s), so even decode-phase bandwidth-bound workloads should benefit.

## Proposed experiment

Run vLLM with `--quantization fp8 --max-num-seqs 64 --enable-prefix-caching` on H100 and B200; compare throughput, TTFT p50, TPOT p50.

## Risks

FP4 path may overshadow FP8 in future builds; pin vLLM commit. Compile-time gaps between Hopper and Blackwell in CUDA 12.8 — confirm correctness on both.
```

- [ ] **Step 4: Run a 4-round loop with both hosts in scope**

```bash
./run_loop.sh --rounds 4 --tag smoke-multihost \
    --hosts h100-1,b200-1 \
    --model meta-llama/Meta-Llama-3-8B-Instruct
```

- [ ] **Step 5: Verify scheduling spread**

After the run:

```bash
grep "host:" wiki/experiments/<date>-*-*.md | sort | uniq -c
```

Expected: at least one experiment dispatched to each of `h100-1` and `b200-1`.

- [ ] **Step 6: Commit a slice-7 completion note**

```markdown
## [<date>] manual | slice-7 multihost serial green

**Op**: manual
**Pages updated**: wiki/log.md
**Key result**: 4-round multi-host loop green; rounds dispatched across both h100-1 and b200-1 per scheduler.
**Notes**: No code changes required — host-registry seams from slice 4 sufficed.
```

---

## Slice 8: AMD/ROCm path (GPU required)

Implement `setup-rocm.sh`, deepen `wiki/hardware/mi300x.md` from the first MI300X experiments, regenerate the compatibility table.

### Task 8.1: Implement setup-rocm.sh

**Files:**
- Modify: `scripts/setup-rocm.sh` (replace stub)

- [ ] **Step 1: Replace the stub**

Replace `scripts/setup-rocm.sh` with the real installer:

```bash
#!/usr/bin/env bash
# AMD/ROCm setup. Mirrors setup-cuda.sh but uses ROCm wheels and skips TRT-LLM.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

# ---------- args ----------
SKIP_TRT=true   # always skip on ROCm
SKIP_HF_WARM=false
DOCKER=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-hf-warm) SKIP_HF_WARM=true ;;
        --docker) echo "ERROR: --docker not supported on ROCm in this slice" >&2; exit 64 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

step() { echo "==== $* ===="; }
fail() { echo "FAIL=$1" >&2; exit 1; }

step "ROCm version check"
if ! command -v rocm-smi >/dev/null 2>&1; then
    fail "no_rocm_smi"
fi
rocm_version="$(rocm-smi --showdriverversion 2>/dev/null | awk '/Driver/{print $NF}')"
echo "ROCm driver: ${rocm_version:-unknown}"

step "Python venv"
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

step "PyTorch ROCm wheels"
pip install --upgrade pip
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.2

step "Install vLLM ROCm build"
# vLLM ROCm install path: from source with VLLM_TARGET_DEVICE=rocm
if [ ! -d /opt/vllm-src ]; then
    sudo git clone https://github.com/vllm-project/vllm /opt/vllm-src
fi
( cd /opt/vllm-src && \
    git checkout v0.6.4.post1 && \
    VLLM_TARGET_DEVICE=rocm pip install -e . ) || fail "vllm_install"

step "Install SGLang ROCm build"
pip install "sglang[all]" || fail "sglang_install"

if [ "$SKIP_HF_WARM" = false ] && [ -n "${MODEL:-}" ]; then
    step "Warm HF cache for $MODEL"
    if [ -n "${HF_TOKEN:-}" ]; then
        export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    fi
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')" || fail "hf_warm"
fi

echo "DONE"
```

```bash
chmod +x scripts/setup-rocm.sh
bash -n scripts/setup-rocm.sh
```

- [ ] **Step 2: Commit**

```bash
git add scripts/setup-rocm.sh
git commit -m "Implement scripts/setup-rocm.sh"
```

### Task 8.2: Onboard MI300X box and run smoke test

- [ ] **Step 1: Add the MI300X to .hosts.toml**

```toml
[hosts.mi300x-1]
ip        = "<your-mi300x-ip>"
user      = "ubuntu"
ssh_key   = "~/.ssh/<your-key>"
vendor    = "amd"
hardware  = "mi300x"
gpu_count = 8
```

- [ ] **Step 2: Trigger setup**

```bash
./scripts/remote-setup.sh mi300x-1
```

Expected: ROCm setup succeeds, prints `[remote-setup] mi300x-1 ready`. If it fails, capture the FAIL reason and add it to `wiki/hardware/mi300x.md` under "Optimization gotchas" before continuing.

- [ ] **Step 3: Add an MI300X-targeted seed hypothesis**

`wiki/hypotheses/vllm-mi300x-baseline-multi-turn-agentic.md`:
```markdown
---
title: "vLLM ROCm baseline on MI300X for multi-turn agentic"
type: hypothesis
tags: [rocm, mi300x, baseline]
model: llama-3-8b
engine: vllm
workload: multi-turn-agentic
hardware: mi300x
status: open
expected_gain: "establish baseline; expected within 0.7-1.2x of H100 throughput"
confidence: high
effort: S
origin: human
created: <today>
updated: <today>
---

## Statement

vLLM ROCm build on MI300X serving Llama 3 8B Instruct for the multi-turn-agentic workload yields throughput within 70%–120% of H100 baseline at concurrency=64.

## Rationale

MI300X has 192 GB HBM (2.4× H100), 5.3 TB/s bandwidth (1.6× H100), but 1.31 PFLOPs BF16 (1.3× H100). The wider/slower compute should approximately match H100 on bandwidth-bound decode while losing on prefill.

## Proposed experiment

Run vLLM with default config (`--max-num-seqs 64`) and the same multi-turn-agentic workload used for H100 baseline; compare throughput, TTFT, TPOT.

## Risks

ROCm flash-attn maturity; HIP graph stability; CK-vs-Triton kernel selection.
```

- [ ] **Step 4: Run a 1-round loop targeting MI300X**

```bash
./run_loop.sh --rounds 1 --tag smoke-mi300x \
    --hosts mi300x-1 \
    --model meta-llama/Meta-Llama-3-8B-Instruct
```

- [ ] **Step 5: Inspect**

```bash
cat wiki/experiments/<date>-vllm-mi300x-baseline-multi-turn-agentic.md
cat raw/benchmarks/<date>-*/metrics.json
```

- [ ] **Step 6: Update wiki/hardware/mi300x.md with measured numbers**

Add the observed MFU, throughput, TTFT, TPOT under "Known performance ceilings". Cite the experiment page.

- [ ] **Step 7: Regenerate the compatibility table**

```bash
python3 scripts/regenerate-compat-table.py
git diff wiki/concepts/engine-hardware-compatibility.md
```

If unchanged, no further commit needed; if engine `supported_hardware` was tightened (e.g., MI300X dropped from SGLang due to install fail), re-commit the table.

- [ ] **Step 8: Commit slice-8 completion note**

```markdown
## [<date>] manual | slice-8 mi300x green

**Op**: manual
**Pages updated**: wiki/hardware/mi300x.md, wiki/log.md
**Key result**: First MI300X experiment green; baseline numbers captured. Cross-vendor loop end-to-end.
**Notes**: Mac-driver multi-vendor design fully implemented through slice 8.
```

---

## Slice 9 (deferred — not in this plan)

Cost guardrails, parallel-host execution, cloud-API provisioning, Docker compose ROCm split. Each becomes its own design + plan once the user requests it.

---

## Plan self-review

**Spec coverage:** Every section of the spec maps to one or more tasks in slices 1-8.

- "Architecture" → slices 4-6 (registry, dispatcher, loop)
- "Host registry" → slice 4
- "Schema additions" → slice 3
- "Orchestration loop" → slice 6
- "Vendor branching" → slices 5, 8
- "Rollout" → all 8 slices
- "Deliverables" — every item in the spec's deliverable list has a task that lands it.
- "Risks" — captured as guardrails inside individual tasks (lint script, halt-on-conflict, FAIL=hf_gated handling, idempotent migration).

**Type/name consistency:**
- `host_registry.py` subcommands match between the script implementation, the test file, and the prompt files (`list`, `get`, `match`, `state`, `schedule`, `reachable`).
- `setup_state` values are consistent across `host_registry.py`, `remote-setup.sh`, and the design spec (`pending|running|ready|failed|unreachable`).
- `HYPOTHESIS=`, `EXPERIMENT=`, `VERDICT=`, `DONE`, `FAIL=` markers consistent across `prompts/pick.md`, `prompts/run.md`, `run_loop.sh`, and `remote-bootstrap.sh`.

**Placeholders:** none. Every step contains the actual code, exact path, or exact command.
