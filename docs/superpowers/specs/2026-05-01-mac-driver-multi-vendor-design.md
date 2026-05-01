# Mac-Driver, Multi-Host, Multi-Vendor Autoresearch Loop — Design

**Branch:** `mac-driver-multi-vendor`
**Status:** design — pending user review
**Date:** 2026-05-01

## Goal

Make Claude Code on the user's Mac the single brain that drives autonomous LLM-serving optimization experiments across multiple long-lived remote GPU boxes spanning both NVIDIA (H100, B200) and AMD (MI300X) hardware. The user provisions boxes manually; everything after that — setup, hypothesis selection, benchmark execution, result ingestion, wiki writeup — is autonomous.

The wiki on the Mac is the only authoritative state. Remote boxes are stateless workers.

## Decisions made during brainstorming

| Decision | Choice |
|---|---|
| Scope | Phase 2 + 3 + 4 in one design (Mac-driver + multi-host + multi-vendor). |
| Provisioning | Hybrid — user provisions long-lived per-vendor boxes; driver never provisions or tears down. |
| Concurrency | Serial first, with seams for parallel later (host registry + per-host state already in place). |
| Orchestration | Bash drives the loop. Each round is a fresh `claude --print` session. |
| Bring-up | Driver-assisted — registry stores `setup_state`; driver runs setup remotely on first dispatch. |
| Host selection | Two-stage — Claude picks the hypothesis; bash matches `hardware:` frontmatter to host registry. |

## Architecture

```
┌─────────────────────────── YOUR MAC (cockpit) ───────────────────────────┐
│                                                                          │
│   .hosts.toml         ← host registry (you edit; gitignored)             │
│   .host-state.toml    ← driver bookkeeping (gitignored)                  │
│       │                                                                  │
│       ▼                                                                  │
│   run_loop.sh         ← bash conductor                                   │
│       │  per round:                                                      │
│       │   1. reachability sweep                                          │
│       │   2. setup pass for any pending/failed hosts                     │
│       │   3. claude --print PICK   → HYPOTHESIS=<slug>                   │
│       │   4. bash matches hyp.hardware × engine.supported_hardware       │
│       │      against registry → pick host                                │
│       │   5. claude --print RUN    → ssh, benchmark, rsync, writeup      │
│       │   6. lint experiment page                                        │
│       │   7. git add wiki/ raw/benchmarks/<run>/ ; commit                │
│       │                                                                  │
│       ▼                                                                  │
│   wiki/   raw/benchmarks/   ← single source of truth, committed to git   │
│                                                                          │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │ ssh (per-command, ControlMaster pooled)
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
       ┌─────────────┐      ┌─────────────┐      ┌──────────────┐
       │ h100-1      │      │ b200-1      │      │ mi300x-1     │
       │ NVIDIA      │      │ NVIDIA      │      │ AMD          │
       │ setup-cuda  │      │ setup-cuda  │      │ setup-rocm   │
       │ vllm,sglang,│      │ vllm,sglang,│      │ vllm,sglang  │
       │ trt-llm     │      │ trt-llm     │      │ (no trt-llm) │
       └─────────────┘      └─────────────┘      └──────────────┘
```

### Invariants

1. **The Mac owns the wiki.** Hosts never read or write wiki state. They emit raw artifacts; the Mac ingests them.
2. **Bash decides where, Claude decides what.** Bash holds the loop counter, the host registry, and the scheduling logic. Claude picks the hypothesis and writes up the result.
3. **Each round is a fresh Claude session.** Cross-round memory is the wiki; in-session memory is one round. A session crash costs at most one round.
4. **Compatibility is data, not prompt content.** Hardware compatibility is enforced via `hypothesis.hardware`, `engine.supported_hardware`, and host-registry tags — bash matches them deterministically. Claude never has to remember which engine works on which GPU.
5. **Setup is mechanical, not reasoned.** Bring-up runs without Claude in the loop; vendor detection is `nvidia-smi` vs `rocm-smi`.

## Host registry

Two files on the Mac, both gitignored.

### `.hosts.toml` — user-edited

```toml
[hosts.h100-1]
ip        = "203.0.113.42"
user      = "ubuntu"
ssh_key   = "~/.ssh/lambda_key"
vendor    = "nvidia"
hardware  = "h100"
gpu_count = 8
notes     = "Lambda Cloud, 8xH100 SXM5"

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
ssh_key   = "~/.ssh/tensorwave_key"
vendor    = "amd"
hardware  = "mi300x"
gpu_count = 8
```

A committed `.hosts.example.toml` documents the schema for new users.

### `.host-state.toml` — driver-written

```toml
[hosts.h100-1]
setup_state    = "ready"        # pending | running | ready | failed | unreachable
setup_started  = "2026-05-01T14:32:00Z"
setup_finished = "2026-05-01T14:48:12Z"
repo_sha       = "f48ff96"
last_dispatch  = "2026-05-01T16:02:00Z"
last_error     = ""
```

Separation of concerns: user file stays free of machine-written timestamps; `rm .host-state.toml` cleanly forces re-setup.

### `scripts/host_registry.py`

Single Python helper (Python 3.11+ stdlib `tomllib`, no external deps). All TOML parsing lives here; bash never parses TOML directly.

```
host_registry.py list                          # all host names, one per line
host_registry.py list --summary                # name, vendor, hardware, state — for prompt context
host_registry.py match --hardware h100         # all host names matching a hyp's hardware field
host_registry.py match --vendor nvidia         # vendor wildcard match
host_registry.py get h100-1 ip                 # single field (ip, user, ssh_key, vendor, hardware, gpu_count)
host_registry.py get h100-1 ssh_target         # synthetic field — returns "<user>@<ip>"
host_registry.py state h100-1 --set ready      # write status to .host-state.toml
host_registry.py state h100-1 --set-error "…"  # set last_error
host_registry.py reachable h100-1              # ssh ping with 10s timeout, exit 0/1
host_registry.py schedule \                    # full scheduling query (used by run_loop.sh)
    --hypothesis-hardware any \                #   intersect: hyp.hardware ∩
    --engine-supported h100,b200,mi300x \      #     engine.supported_hardware ∩
    --exclude h100-1                           #     registry hosts (minus excluded)
                                               #   prints a single host name or "none"
```

### Hardware-matching rules

| `hypothesis.hardware` | Matches host where… |
|---|---|
| `any` | any host |
| `nvidia` | `vendor = "nvidia"` |
| `amd` | `vendor = "amd"` |
| `h100` / `b200` / `mi300x` / etc. | `hardware = "<slug>"` |

### SSH

- Bash invokes `ssh -i "$ssh_key" -o ControlMaster=auto -o ControlPath=~/.ssh/cm-%r@%h:%p -o ControlPersist=10m "$user@$ip" "<cmd>"`.
- ControlMaster pooling means after the first connection per box, subsequent SSH calls reuse the open channel.
- No ssh-agent forwarding. Explicit key path is auditable and stateless.
- Pre-dispatch reachability check: `ssh -o ConnectTimeout=10 ... echo ok`. Failure marks the host `unreachable` and skips it.

## Schema additions

### New page type: `hardware`

Lives at `wiki/hardware/<slug>.md`. Slug matches the tag used in `.hosts.toml` and hypothesis frontmatter (`h100`, `b200`, `mi300x`, etc.).

**Frontmatter:**

```yaml
---
title: "NVIDIA H100"
type: hardware
tags: [hardware, nvidia, hopper]
vendor: nvidia
arch: hopper
created: 2026-05-01
updated: 2026-05-01
---
```

**H2 sections:** Specs, Memory hierarchy, Compute features, Engine support, Known performance ceilings, Optimization gotchas, Connections, Sources.

**Seed pages on landing the schema:** `h100.md`, `b200.md`, `mi300x.md`. `h200.md`, `mi325x.md` as stubs.

### Frontmatter additions on existing page types

| Page type | New field | Values | Required |
|---|---|---|---|
| `hypothesis` | `hardware` | `any`, `nvidia`, `amd`, or specific slug | yes |
| `experiment` | `hardware` | specific slug only — what it actually ran on | yes |
| `experiment` | `host` | name from `.hosts.toml` (e.g., `h100-1`) | yes |
| `model` | `target_hardware` | list of slugs the model is being optimized for | yes |
| `engine` | `supported_hardware` | list — e.g., `[h100, b200, mi300x]` for vLLM | yes |

### Generated compatibility table

`wiki/concepts/engine-hardware-compatibility.md` is regenerated by `LINT` from each engine page's `supported_hardware` field. Single source (engine pages), single truth (lint regenerates), no drift.

### Migration

- Existing 8 hypotheses get `hardware: any` added.
- Existing 19 experiments get `hardware: tpu-v6e` and `host: <inferred>` added; they don't participate in serving scheduling but stay in the wiki as honest history.
- Migration done in slice 3 by a one-shot script committed alongside the schema change.

## Orchestration loop

### Driver invocation

```bash
./run_loop.sh \
  --rounds 5 \
  --hosts h100-1,b200-1 \          # optional whitelist; default = all
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --tag exp-2026-05-01             # log file naming
```

### Per-round flow

1. **Round banner** appended to `raw/loops/<date>-<tag>.log`.
2. **Reachability sweep** — every in-scope host gets a 10s SSH ping; `.host-state.toml` updated.
3. **Setup pass** — for any host where `setup_state ∈ {pending, failed}`, run `scripts/remote-setup.sh <host>`. Pure bash; no Claude turn.
4. **PICK turn** — `claude --print` reads the wiki and prints `HYPOTHESIS=<slug>` (or `HYPOTHESIS=none`).
5. **Schedule** — bash reads three things and intersects them:
   - `hypothesis.hardware` from the picked hypothesis page's frontmatter
   - `engine.supported_hardware` from the engine page named by `hypothesis.engine`
   - reachable hosts from `.hosts.toml` whose `setup_state = "ready"`

   Intersection is computed by `host_registry.py schedule …`. In serial mode, only one host runs at a time, so any reachable+ready+compatible host counts as eligible (the registry is small enough that order doesn't matter; we'll add tie-breaking when parallel mode lands). If the intersection is empty, log `skipped: no compatible host for <hyp>` and re-run PICK with the unschedulable hypothesis added to an exclusion list passed in the user message. Max 3 PICK attempts before giving up the round.
6. **RUN turn** — `claude --print` is given hypothesis slug + host name + run slug + model. Claude does SSH, benchmark, rsync, writeup itself via Bash and Edit tool calls.
7. **Lint** — bash runs the experiment-page linter. Failure aborts the round, leaves the page in place for inspection.
8. **Round commit** — `git add wiki/ raw/benchmarks/<run>/ && git commit -m "round N: <slug> on <host>"`.
9. **Loop or exit.**

### Bash ↔ Claude contract — line-based

Claude prints `KEY=value` lines on stdout that bash greps. Reasoning goes to stderr. No JSON parsing in bash.

**PICK turn:**

- Input: system prompt is `prompts/pick.md`. User message: round number, model, registry summary.
- Output stdout: `HYPOTHESIS=<slug>` or `HYPOTHESIS=none`.

**RUN turn:**

- Input: system prompt is `prompts/run.md`. User message: hypothesis slug, host name, run slug, model.
- Output stdout: `EXPERIMENT=<path>` and `VERDICT=<verdict>`.

The two prompt files live in `prompts/`, are committed, and are the only place loop logic exists in natural language.

### What Claude does inside the RUN turn

Via Bash and Edit tool calls:

```bash
# 1. Resolve host details
ssh_key=$(scripts/host_registry.py get h100-1 ssh_key)
ssh_target=$(scripts/host_registry.py get h100-1 ssh_target)

# 2. Sync branch state to box
ssh -i "$ssh_key" "$ssh_target" \
    "cd ~/llm-serving-autoresearch-wiki && git fetch && git checkout $branch && git pull --ff-only"

# 3. Run benchmark
ssh -i "$ssh_key" "$ssh_target" \
    "cd ~/llm-serving-autoresearch-wiki && \
     python benchmark_harness.py \
       --engine vllm --model $model \
       --workload multi-turn-agentic \
       --config '{\"enable_prefix_caching\": true, \"max_num_seqs\": 128}' \
       --output-dir raw/benchmarks/$run_slug \
       --launch-server"

# 4. Pull artifacts back
rsync -avz -e "ssh -i $ssh_key" \
    "$ssh_target:~/llm-serving-autoresearch-wiki/raw/benchmarks/$run_slug/" \
    "raw/benchmarks/$run_slug/"

# 5. Write experiment page (Edit tool)
# 6. Update wiki/index.md, wiki/log.md, hypothesis status (Edit tool)
```

### Setup-on-first-dispatch — bash only

```bash
# scripts/remote-setup.sh <host-name>
ssh "$target" 'bash -s' < scripts/remote-bootstrap.sh
# remote-bootstrap.sh: clones repo, detects vendor (nvidia-smi vs rocm-smi),
# dispatches to setup-cuda.sh or setup-rocm.sh, prints DONE or FAIL=<reason>.
```

No Claude turn. Setup is mechanical: detect → branch → install. Keeping Claude out makes setup deterministic.

### Failure modes

| Failure | Bash response |
|---|---|
| SSH unreachable mid-round | Mark host `unreachable`. Abort round. No experiment page. Continue to next round. |
| Benchmark crashes (non-zero exit) | Claude in RUN turn captures stderr, writes experiment page with `verdict: invalid`, exits cleanly. |
| Claude prints no `EXPERIMENT=` line | Bash retries the RUN turn once with stricter prompt. If still malformed, abort round. |
| Experiment page fails frontmatter lint | Print lint error. Abort round. Leave page in place for inspection. |
| User edits wiki while loop is running | `git commit` step detects conflict. Halt loop. User resolves and `--resume`s. |
| HF token gates a model on setup | `remote-bootstrap.sh` exits with `FAIL=hf_gated`. `setup_state=failed`. Host pulled from rotation. |
| 3 PICK turns return `none` or unschedulable | Loop exits cleanly with "no further work". |

## Vendor branching: AMD/ROCm path

The diff between NVIDIA and AMD is concentrated in two layers — driver/install and engine availability — and almost nowhere else.

### `setup.sh` becomes a dispatcher

```bash
detect_vendor() {
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L | grep -q GPU; then echo nvidia; return; fi
    if command -v rocm-smi  >/dev/null 2>&1 && rocm-smi  --showid | grep -q GPU; then echo amd;    return; fi
    echo "ERROR: no NVIDIA or AMD GPU detected" >&2; exit 1
}
case "$(detect_vendor)" in
    nvidia) exec bash scripts/setup-cuda.sh "$@" ;;
    amd)    exec bash scripts/setup-rocm.sh "$@" ;;
esac
```

- `scripts/setup-cuda.sh` — current `setup.sh` body, moved verbatim.
- `scripts/setup-rocm.sh` — new, mirrors structure: install ROCm 6.x, install ROCm-flavored PyTorch wheels, install vLLM ROCm build (`VLLM_TARGET_DEVICE=rocm`), install SGLang ROCm build, **skip TensorRT-LLM**, warm HF cache.

### `benchmark_harness.py` — no changes

Both vLLM and SGLang ROCm builds expose the same CLI flags as their CUDA builds. The harness launches whatever engine is installed. TensorRT-LLM exclusion is enforced via engine `supported_hardware`, not the harness.

### Engine `supported_hardware` declarations

```yaml
# wiki/engines/vllm.md
supported_hardware: [h100, h200, b200, mi300x]

# wiki/engines/sglang.md
supported_hardware: [h100, h200, b200, mi300x]

# wiki/engines/tensorrt-llm.md
supported_hardware: [h100, h200, b200]
```

Conservative starting points; observations from first MI300X experiments will revise them.

### `wiki/hardware/mi300x.md` — AMD-specific knowledge

What the agent needs to reason on AMD: ROCm version pins, Flash-Attention path (Triton-on-ROCm or AMD composable-kernels), HIP graph maturity, no TRT-LLM, FP8/INT8 quantization gaps, Infinity Fabric topology. Not exhaustive — only what affects serving optimization decisions.

### Docker compose — deferred

Current `docker-compose.yml` is NVIDIA-pinned. Splitting per vendor is real work (different base images, different device flags). Mac-driver path uses native install on the box, not Docker. Keep Docker NVIDIA-only for now; document in `docs/host-onboarding.md`. Add ROCm Docker only when a real user asks.

## Rollout — eight slices

| # | Slice | What lands | GPU? |
|---|---|---|---|
| 1 | Engine ingestion | `wiki/engines/{vllm,sglang,tensorrt-llm}.md` populated; `supported_hardware` declared. | no |
| 2 | Hardware pages | `wiki/hardware/{h100,b200,mi300x}.md`. | no |
| 3 | Schema additions | `SCHEMA.md` updates; new page type; frontmatter rules; lint rules; migration. | no |
| 4 | Host registry | `scripts/host_registry.py`, `.hosts.example.toml`, `docs/host-onboarding.md`. | no |
| 5 | Setup dispatcher | Split `setup.sh` → dispatcher + `scripts/setup-cuda.sh`. `scripts/remote-bootstrap.sh`. `scripts/setup-rocm.sh` lands as a one-line stub that prints `FAIL=rocm_not_implemented` and exits non-zero (slice 8 fills it in). | no |
| 6 | Mac-driver `run_loop.sh` | New `run_loop.sh` with PICK/RUN. `prompts/pick.md`, `prompts/run.md`. Driver-assisted bring-up. **First end-to-end smoke test on one rented H100.** | yes |
| 7 | Multi-host serial | Add a second host to `.hosts.toml` (B200), run a 4-round loop scheduling spreads across both. No code change. | yes |
| 8 | AMD/ROCm | `scripts/setup-rocm.sh`. Deepen `wiki/hardware/mi300x.md` from real experiments. First MI300X smoke test. | yes |

Slices 1–3 can run in parallel. Slices 4–6 are sequential. Slices 7–8 unlock once 6 is green.

## Deliverables

### Code
- `run_loop.sh` (rewritten)
- `setup.sh` (dispatcher)
- `scripts/setup-cuda.sh` (moved verbatim from current `setup.sh`)
- `scripts/setup-rocm.sh` (new, slice 8)
- `scripts/remote-bootstrap.sh` (new)
- `scripts/remote-setup.sh` (new — bash wrapper that ssh's `remote-bootstrap.sh`)
- `scripts/host_registry.py` (new)
- `prompts/pick.md`, `prompts/run.md` (new)
- `scripts/lint-experiment-page.sh` (new — verifies frontmatter + structure of a written experiment page)

### Config
- `.hosts.example.toml` (committed)
- `.hosts.toml`, `.host-state.toml` (gitignored)
- `.gitignore` updates

### Wiki
- `wiki/hardware/h100.md`, `b200.md`, `mi300x.md`
- `wiki/engines/{vllm,sglang,tensorrt-llm}.md` populated (replaces stubs)
- `wiki/concepts/engine-hardware-compatibility.md` (lint-generated)
- Frontmatter migrations on 8 hypotheses + 19 experiments

### Docs
- `docs/superpowers/specs/2026-05-01-mac-driver-multi-vendor-design.md` (this file)
- `docs/host-onboarding.md`
- `docs/architecture.md`
- `SCHEMA.md` updates
- `README.md` updates

## Risks

1. **No cost guardrails in v1.** A rogue loop could rack up GPU bills. Slice 9 (deferred): `--max-cost` / `--max-hours` flags to `run_loop.sh`.
2. **Per-experiment artifact size.** Profiles can be GB-scale. Mitigation: cap profile size on the box and rsync only metric JSONs by default; fetch large traces on demand.
3. **Wiki write conflicts.** If user edits `wiki/index.md` while a round runs, `git commit` fails and the loop halts cleanly. Documented; acceptable.
4. **Drift in agent writeups across rounds.** Mitigation: strict template in `prompts/run.md` + frontmatter+structure linter as last bash step of every round.
5. **MI300X engine builds are fast-moving.** Slice 8 should pin commits and document them in `wiki/hardware/mi300x.md`. Updates become deliberate.

## Out of scope for this design

- Cost guardrails (`--max-cost`).
- Parallel-host execution (the registry and scheduling already accommodate it; the loop just doesn't dispatch concurrently yet).
- Cloud-API-driven provisioning (Lambda, RunPod, etc.).
- Docker compose ROCm split.
- Image-baked snapshots (pre-installed AMIs).
- Cross-tier hardware wildcards (`nvidia-hopper`, `nvidia-blackwell`).

These are tracked as future slices but not part of v1.
