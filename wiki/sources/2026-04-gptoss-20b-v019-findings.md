---
title: "GPT-OSS-20B serving config search — vLLM 0.19 winning config (LEAN)"
type: source
tags: [docs, internal, gpt-oss, vllm-tune, h100, eagle3, serving, vllm-0.19]
created: 2026-05-04
updated: 2026-05-04
---

The decisive vllm-tune findings doc that resolved the v0.18-vs-v0.19 question for gpt-oss-20B and identified the new shipping default config (LEAN). Bench was 1× H100 80 GB SXM, driver 580, GPU_MEM=0.95 (enabled by v0.19's CUDA-graph budgeting fix). LEAN beats stock vLLM 0.19 by +66% decode / +11% prefill / +28% sharegpt at the per-replica peak concurrencies, with no regression anywhere on the full curve (c=1, 8, 128, 512, 1024 across decode / prefill / sharegpt).

## Overview

Single-H100 ablation matrix run on 2026-04-28 covering 9 configs (BASE, OPT, K2, NOSPEC, BLK64, BATCH8K, NOFP8KV, NOBF16, LEAN) on vLLM v0.19.0. Per-replica numbers; multiplied by ~4 for the 4-GPU LB stack used in production (per-replica equivalence holds within 1.6%). Predecessors: prior 4-GPU `baseline-vs-optimized` study on v0.18 (where the OPT config was tuned for 120B), and the `gptoss-20b-config-search-plan.md` defining the 9-config matrix.

The doc also includes:
- **v0.18 → v0.19 canary** — same OPT flags on both versions: v0.19 wins +61% decode, +17% prefill, +60% sharegpt. The 7× MXFP4 MoE regression documented for 120B does not bite 20B.
- **P-EAGLE speculator evaluation** — `amazon/GPT-OSS-20B-P-EAGLE` at k=2, k=3, k=7. Verdict: only k=7 is shippable, and only as an offline-batch-decode SKU (TTFT becomes 40-80 s, prefill regresses 36%).
- **Full curves** — BASE vs LEAN swept across c={1,8,128,512,1024} on all three workloads. LEAN beats BASE at every cell.

## Key claims

1. **vLLM 0.19 is a clean win on 20B.** Even stock 0.19 (no flags) beats the previously-shipped `v0.18 + OPT` config on every workload. Eagle3 acceptance jumps from 28% → 79% (3rd-draft accept goes 10% → 68%).
2. **The previously-shipped OPT flag set hurts prefill on v0.19.** Eagle3 k=3 adds per-step draft-model compute that wastes prefill cycles — regresses prefill by 30% vs stock at v0.19.
3. **LEAN (Eagle3 k=2 + block_size=64 + max-num-batched-tokens=8192 + bf16 + fp8 KV) wins everywhere.** Worst-vs-BASE = 1.00× — only LEAN and NOSPEC don't lose anywhere, and LEAN delivers far more upside.
4. **At high concurrency, the gap widens.** Stock vLLM without fp8 KV collapses at c=512+ on decode (KV cache saturates → preemption → swap thrash). At c=512 decode: 11,279 LEAN vs 4,482 BASE — a 2.5× advantage with half the request latency.
5. **Eagle3 is the prefill regressor on 20B; cost scales with k.** k=3 → 0.70× BASE prefill, k=2 → 0.91×, k=0 → 1.04×. The non-spec flags (bf16+fp8 KV+blk128+batch16k) are fine on their own — NOSPEC is +4% above BASE on prefill.
6. **Run-to-run variance ≈ 10%** at the same c=128 decode cell across three independent measurements (15,131 / 16,660 / 17,656). Cross-config ratios are stable; absolute numbers carry ±10% uncertainty.
7. **`block_size=64` standalone OOMs at GPU_MEM=0.95.** LEAN combines blk=64 with batch=8192, which reduces activation memory enough to fit. Drop GPU_MEM to 0.90 if running blk=64 standalone.

## Key data points

### Ablation matrix — full results (v0.19, GPU_MEM=0.95)

Decode c=128, prefill c=512, sharegpt c=256. tok/s (output).

| Config | Eagle3 k | block_size | batched_tokens | dtype | KV | decode | prefill | sharegpt | worst-vs-BASE |
|---|---|---|---|---|---|---:|---:|---:|---:|
| `BASE` | — | (default) | (default) | auto | auto | 10,338 | 4,826 | 11,071 | 1.00× |
| `OPT` | 3 | 128 | 16384 | bf16 | fp8 | 14,897 | 3,386 | 16,769 | 0.70× |
| `K2` | 2 | 128 | 16384 | bf16 | fp8 | 14,652 | 4,412 | 16,619 | 0.91× |
| `NOSPEC` | — | 128 | 16384 | bf16 | fp8 | 11,288 | 5,000 | 12,038 | 1.04× |
| `BATCH8K` | 3 | 128 | 8192 | bf16 | fp8 | 16,759 | 4,307 | 16,785 | 0.89× |
| `NOBF16` | 3 | 128 | 16384 | auto | fp8 | 16,391 | 3,385 | 16,315 | 0.70× |
| `BLK64` | 3 | 64 | 16384 | bf16 | fp8 | OOM | OOM | OOM | — |
| **`LEAN`** | **2** | **64** | **8192** | **bf16** | **fp8** | **16,660** | **4,808** | **16,242** | **1.00×** |

### Per-workload peaks (LEAN full-curve sweep)

| | base peak | optimized peak | gain |
|---|---|---|---|
| decode | 10,611 (c=128) | **17,656** (c=128) | +66% |
| prefill | 4,862 (c=512) | **5,397** (c=1024) | +11% |
| sharegpt | 14,063 (c=512) | **18,065** (c=1024) | +28% |

### Eagle3 acceptance — v0.18 vs v0.19 (OPT config)

| | v0.18 OPT | v0.19 OPT |
|---|---|---|
| overall accept | 0.28 | **0.79** |
| pos-0 (1st draft) | 0.51 | 0.91 |
| pos-1 (2nd draft) | 0.24 | 0.79 |
| pos-2 (3rd draft) | **0.10** | **0.68** |

### Recommended LEAN config (verbatim)

```
vllm/vllm-openai:v0.19.0
  --tensor-parallel-size 1
  --trust-remote-code
  --gpu-memory-utilization 0.95
  --dtype bfloat16
  --kv-cache-dtype fp8
  --block-size 64
  --max-num-batched-tokens 8192
  --speculative-config '{"method":"eagle3","model":"/models/RedHatAI/gpt-oss-20b-speculator.eagle3","num_speculative_tokens":2}'
```

## Techniques referenced

- Eagle3 speculative decoding (RedHatAI draft, P-EAGLE)
- Chunked prefill scheduler (vLLM)
- KV cache fp8 quantization
- Paged KV cache block_size tuning
- max_num_batched_tokens chunk-cycle tuning
- vLLM 0.19 CUDA-graph budgeting fix (`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`)
- bf16 promotion of MXFP4 MoE weights

## Gaps & caveats

1. **Per-replica → 4-GPU projection assumes nginx LB overhead < 1%** (holds at the published 4-GPU peaks; not re-verified on v0.19).
2. **Boot-time degradation across sequential matrix runs** — boot times grew 111s → 2065s (BASE → LEAN) over 7-config run. Likely Docker daemon / driver state accumulation. Doesn't affect data integrity, but production has no chained-restart pattern.
3. **Eagle3 acceptance metric reformulation** between vLLM versions — pos-2 went 0.10 → 0.68. Could be a real implementation improvement in 0.19 or a metric counting change. Headline throughput is measured externally by evalscope, so it stands either way.
4. **All ablation cells are single-point measurements at the workload's published 4-GPU peak (per-replica equivalent).** The full curves are authoritative for production sizing.
5. **DFlash and Snowflake Arctic-LSTM speculators remain untested** — both need custom vLLM branches.

## Connections

- Hypotheses (this round, all per-replica on 1× H100 80 GB SXM at GPU_MEM=0.85):
  - [`gptoss-20b-opt-baseline`](../hypotheses/gptoss-20b-opt-baseline.md) — already supported in [OPT round 7](../experiments/2026-05-04-gptoss20b-h100-opt.md)
  - [`gptoss-20b-base-on-h100`](../hypotheses/gptoss-20b-base-on-h100.md)
  - [`gptoss-20b-k2-on-h100`](../hypotheses/gptoss-20b-k2-on-h100.md)
  - [`gptoss-20b-nospec-on-h100`](../hypotheses/gptoss-20b-nospec-on-h100.md)
  - [`gptoss-20b-blk64-on-h100`](../hypotheses/gptoss-20b-blk64-on-h100.md)
  - [`gptoss-20b-batch8k-on-h100`](../hypotheses/gptoss-20b-batch8k-on-h100.md)
  - [`gptoss-20b-nofp8kv-on-h100`](../hypotheses/gptoss-20b-nofp8kv-on-h100.md)
  - [`gptoss-20b-lean-on-h100`](../hypotheses/gptoss-20b-lean-on-h100.md)
- Predecessor source: [`gptoss-20b-config-search-plan`](2026-04-gptoss-20b-config-search-plan.md)
- Related source: [`gptoss-20b-h100-oom`](2026-04-gptoss-20b-h100-oom.md) (why GPU_MEM=0.85 on v0.18, lifted to 0.95 on v0.19)
- Codebase (yet to ingest): `wiki/codebases/vllm-tune.md`
- Hardware: `wiki/hardware/h100.md`

## Sources

- `~/vllm-tune/docs/gptoss-20b-v019-config-search-findings.md` (canonical, on-box only — not committed to wiki repo)
