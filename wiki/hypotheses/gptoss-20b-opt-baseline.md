---
title: "gpt-oss-20B OPT config establishes per-replica baseline on 1× H100"
type: hypothesis
tags: [serving, gpt-oss, h100, eagle3, baseline]
model: gptoss-20b
engine: vllm
workload: multi-turn-agentic
hardware: h100
status: supported
expected_gain: "establish per-replica baseline (decode/prefill/sharegpt at the per-replica peak concurrencies)"
confidence: high
effort: S
origin: human
created: 2026-05-04
updated: 2026-05-04
---

## Statement

The `OPT` config from `vllm-tune`'s 9-config search matrix (vLLM 0.19.0 + bf16 + fp8 KV + block_size=128 + max_num_batched_tokens=16384 + Eagle3-RedHatAI k=3) boots cleanly on a single H100 80 GB SXM with `gpu_memory_utilization=0.85` (working around the v0.18 sampler-warmup OOM, fixed in v0.19) and serves gpt-oss-20B successfully across the three workloads at the per-replica peak concurrencies (decode @ c=128, prefill @ c=512, sharegpt @ c=256) with 100% request success.

## Rationale

`OPT` was the working-as-intended baseline tuned for the 4× H100 production stack ([deploy/vllm-optimized/](../codebases/vllm-tune.md)) before the LEAN ablation took over as the workload-agnostic shipping default. Establishing the per-replica numbers on a single H100 80 GB box is the first calibration step before running the rest of the 9-config matrix (`K2`, `NOSPEC`, `BLK64`, `BATCH8K`, `NOFP8KV`, `NOBF16`, `LEAN`).

## Proposed experiment

Run `scripts/experiments/gptoss20b_config_search/run_matrix.sh --vllm-version v0.19.0 --gpu-mem 0.85 --configs OPT` from the on-box `~/vllm-tune` checkout. Capture per-replica throughput for decode @ c=128, prefill @ c=512, sharegpt @ c=256.

Pass condition: all three cells return `success_pct=100.0` with non-zero `output_tok_s` and the matrix script writes `summary.md`.

## Risks

- Single-H100 80 GB is the smallest tier `gpt-oss-20b` runs on. `gpu_memory_utilization=0.9` would OOM under v0.18 (see [`gptoss-20b-80gb-h100-oom-finding.md`](../sources/2026-04-gptoss-20b-h100-oom.md) — pending ingestion). On v0.19 the budgeting fix lets us run safely at 0.85.
- Eagle3 k=3 acceptance varies by workload; baseline only — no claim about which k is best.
