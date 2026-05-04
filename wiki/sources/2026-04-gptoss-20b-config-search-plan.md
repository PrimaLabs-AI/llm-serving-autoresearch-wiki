---
title: "GPT-OSS-20B serving-config search — 9-config matrix plan (vLLM 0.18 / 0.19)"
type: source
tags: [docs, internal, gpt-oss, vllm-tune, h100, planning, ablation]
created: 2026-05-04
updated: 2026-05-04
---

The pre-execution plan that defines the 9-config ablation matrix the Mac-driver round is now executing on `h100-1`. Specifies each config's flag-diff vs `OPT` (the prior 120B-tuned default), the per-replica equivalent of each workload's 4-GPU peak concurrency, and the v0.18-vs-v0.19 canary decision tree. A single H100 (80 GB SXM, 94 GB NVL, or 141 GB H200) is sufficient because the production stack is 4× independent TP=1 replicas — per-replica behavior reproduces exactly on 1 GPU.

## Overview

The plan was authored before the v019 config search ran; its goal was to find a vLLM flag combo for gpt-oss-20B that wins net across decode, prefill, and sharegpt on H100. The "optimized" config it set out to challenge had been tuned on 120B and only won on decode (prior 4-GPU result: decode 1.53×, prefill 0.97×, sharegpt 0.89×). The plan calls out each row of the matrix with rationale, sets per-replica concurrency (decode c=128, prefill c=512, sharegpt c=256), pins vLLM 0.18.0 by default, and gates a switch to 0.19 on a canary measurement.

After execution, the [findings doc](2026-04-gptoss-20b-v019-findings.md) reports the canary went strongly in v0.19's favor (+61% decode), so the matrix was rerun on v0.19. LEAN won.

## Key claims

1. **A single H100 is sufficient for matrix sweeping.** Production = 4× independent TP=1 replicas; per-replica measurements multiplied by 4 reproduce the 4-GPU peaks within 1.6%.
2. **9 configs are sufficient.** Each is a single-knob ablation from `OPT`, plus the lean compound (`LEAN`). Larger sweeps (P-EAGLE, DFlash, Snowflake Arctic-LSTM) are deferred to v2.
3. **`gpu_memory_utilization=0.85` stays fixed on v0.18** for the OOM finding documented separately. On v0.19 with the budgeting fix the plan adds one extra `GPU_MEM=0.95` cell on the winner.
4. **Per-cell time budget ≈ 5 min** (1 min warmup + 3 min run + 1 min cooldown). 9 × 3 cells + 9 boots × 3 min ≈ 3 h wall-clock. Total +2 h if both v0.18 and v0.19 are swept.
5. **Canary decision tree:** v0.19 / v0.18 ratio ≥ 0.85 → rebase on v0.19 + add GPU_MEM=0.95 cell; 0.75-0.85 → one more cell before deciding; < 0.75 → stay on v0.18.

## Key data points

### The 9-config matrix (verbatim)

All configs keep `--tensor-parallel-size 1`, `--trust-remote-code`, `--gpu-memory-utilization 0.85`. Differences:

| ID | dtype | kv-cache-dtype | block-size | batched-tokens | spec method | spec k | Why this row |
|---|---|---|---:|---:|---|---:|---|
| `BASE` | auto | auto | 16 | (default) | — | — | Stock vLLM. Floor. |
| `OPT` | bf16 | fp8 | 128 | 16384 | Eagle3 (RedHatAI) | 3 | Current "optimized" — control. |
| `K2` | bf16 | fp8 | 128 | 16384 | Eagle3 (RedHatAI) | 2 | Pos-2 accept = 9.5%; side-test showed -25% draft compute, only -2.2% accept length. |
| `NOSPEC` | bf16 | fp8 | 128 | 16384 | — | — | Isolates whether spec itself drives the sharegpt regression. |
| `BLK64` | bf16 | fp8 | **64** | 16384 | Eagle3 (RedHatAI) | 3 | block_size=128 was tuned for 120B's wider GQA. |
| `BATCH8K` | bf16 | fp8 | 128 | **8192** | Eagle3 (RedHatAI) | 3 | 16k matmul flop-bound on 120B; finishes too fast on 20B. |
| `NOFP8KV` | bf16 | **auto** | 128 | 16384 | Eagle3 (RedHatAI) | 3 | KV not the bottleneck on 20B at 80GB; quant cost may exceed capacity gain. |
| `NOBF16` | **auto** | fp8 | 128 | 16384 | Eagle3 (RedHatAI) | 3 | Drop bf16 promotion; let MXFP4 stay native. |
| `LEAN` | bf16 | fp8 | **64** | **8192** | Eagle3 (RedHatAI) | 2 | Compound the lean tweaks. |

### Workload concurrency (per-replica equivalents)

| Workload | Input | Output | 4-GPU peak | Per-replica → c |
|---|---|---|---|---|
| `decode` | 256 | 4096 | c=512 (optimized) | **c=128** |
| `prefill` | 4096 | 512 | c=2048 (baseline) | **c=512** |
| `sharegpt` | ~175 | ≤4096 | c=1024 (baseline) | **c=256** |

### Output layout

```
results/gptoss20b_config_search_<ts>/
├── per_config/<config>.csv      # one row per workload
├── boot_logs/<config>.log       # docker logs of each launch
├── metrics/<config>.txt         # /metrics snapshot at end (spec accept rates)
├── evalscope/<config>/          # raw evalscope cell outputs
├── all.csv                      # aggregated, with config_id column
└── summary.md                   # ranked table per workload + winner pick
```

## Techniques referenced

- One-knob ablation methodology (each config diffs exactly one knob from OPT)
- Per-replica equivalent concurrency derivation (4-GPU peak / 4)
- Eagle3 speculative decoding (RedHatAI draft)
- KV cache fp8 quantization vs auto
- Paged KV cache block_size tuning
- bf16 promotion of MXFP4 MoE
- vLLM CUDA-graph budgeting fix (v0.19 default)
- Canary-gated version switch

## Gaps & caveats

1. **Absolute tok/s reported is per-replica.** Multiply by 4 for the 4-GPU projection. Only the *ratios* across configs are directly comparable to the 4-GPU findings.
2. **nginx-LB overhead (<1%) is not measured here.**
3. **`gpu_memory_utilization=0.85` is fixed for OOM reasons on 0.18.** On 0.19 with the budgeting fix the winner gets one extra GPU_MEM=0.95 cell.
4. **Out-of-scope (deferred to v2):** P-EAGLE (needs vLLM PR #35062), DFlash (needs PR #40898), Snowflake Arctic-LSTM (Arctic Inference plugin + custom vLLM branch). All need custom builds; v1 matrix is OSS-image-only.
5. The plan was written assuming v0.18 would stay; the canary section enumerates v0.19 contingencies.

## Connections

- Successor / executed report: [`gptoss-20b-v019-findings`](2026-04-gptoss-20b-v019-findings.md) (the v0.19 result; LEAN is the winner)
- Related source: [`gptoss-20b-h100-oom`](2026-04-gptoss-20b-h100-oom.md) (why GPU_MEM=0.85 on v0.18)
- Hypotheses driven by this matrix:
  - [`gptoss-20b-opt-baseline`](../hypotheses/gptoss-20b-opt-baseline.md) (supported in [OPT round 7](../experiments/2026-05-04-gptoss20b-h100-opt.md))
  - [`gptoss-20b-base-on-h100`](../hypotheses/gptoss-20b-base-on-h100.md)
  - [`gptoss-20b-k2-on-h100`](../hypotheses/gptoss-20b-k2-on-h100.md)
  - [`gptoss-20b-nospec-on-h100`](../hypotheses/gptoss-20b-nospec-on-h100.md)
  - [`gptoss-20b-blk64-on-h100`](../hypotheses/gptoss-20b-blk64-on-h100.md)
  - [`gptoss-20b-batch8k-on-h100`](../hypotheses/gptoss-20b-batch8k-on-h100.md)
  - [`gptoss-20b-nofp8kv-on-h100`](../hypotheses/gptoss-20b-nofp8kv-on-h100.md)
  - [`gptoss-20b-lean-on-h100`](../hypotheses/gptoss-20b-lean-on-h100.md)
- Codebase (yet to ingest): `wiki/codebases/vllm-tune.md`
- Hardware: `wiki/hardware/h100.md`

## Sources

- `~/vllm-tune/docs/gptoss-20b-config-search-plan.md` (canonical, on-box only — not committed to wiki repo)
