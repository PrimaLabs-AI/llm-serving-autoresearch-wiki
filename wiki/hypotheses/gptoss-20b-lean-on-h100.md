---
title: "LEAN config (k=2 + blk=64 + batch=8k) is the workload-agnostic winner over OPT"
type: hypothesis
tags: [serving, gpt-oss, h100, eagle3, vllm-tune, shipping-default]
model: gptoss-20b
engine: vllm
workload: multi-turn-agentic
hardware: h100
status: open
expected_gain: "≥+5% on every workload vs OPT (no regression anywhere) at per-replica peak concurrencies"
confidence: high
effort: S
origin: human
created: 2026-05-04
updated: 2026-05-04
---

## Statement

The `LEAN` config (Eagle3 RedHatAI **k=2** + bf16 + fp8 KV + **`block_size=64`** + **`max_num_batched_tokens=8192`**) on 1× H100 80 GB SXM at `gpu_memory_utilization=0.85` beats [OPT round 7](../experiments/2026-05-04-gptoss20b-h100-opt.md) by ≥5% on every workload (decode @ c=128, prefill @ c=512, sharegpt @ c=256) with no regression anywhere. LEAN is what `vllm-tune/deploy/vllm-optimized/` already ships; this experiment puts that decision into the wiki audit trail.

## Rationale

LEAN compounds three independent OPT-relative wins identified in the [v019 findings doc](../sources/2026-04-gptoss-20b-v019-findings.md):

1. **k=2 vs k=3** (see [`gptoss-20b-k2-on-h100`](gptoss-20b-k2-on-h100.md)): cuts the Eagle3 per-step draft-compute penalty roughly in half on prefill while preserving most of the decode multiplier (pos-2 acceptance was 62% in OPT round 7).
2. **block_size=64 vs 128** (see [`gptoss-20b-blk64-on-h100`](gptoss-20b-blk64-on-h100.md)): tighter KV blocks on 20B's narrower attention frees usable cache.
3. **max_num_batched_tokens=8192 vs 16384** (see [`gptoss-20b-batch8k-on-h100`](gptoss-20b-batch8k-on-h100.md)): tighter chunk-prefill cycle better matched to 20B's matmul time.

The findings doc reports LEAN at +66% decode, +11% prefill, +28% sharegpt vs BASE on the same hardware (GPU_MEM=0.95), and worst-vs-BASE = 1.00× — the only config besides NOSPEC that doesn't lose anywhere. Compared to OPT specifically the gain on prefill is the largest (LEAN recovers the prefill Eagle3 penalty almost entirely), and decode/sharegpt land near OPT or slightly above.

The [shipping-default question is whether this also holds at GPU_MEM=0.85](../sources/2026-04-gptoss-20b-h100-oom.md): findings ran at 0.95 thanks to the v0.19 budgeting fix; at 0.85 KV capacity is ~3 GiB lower, which should not change the LEAN-vs-OPT ranking but may compress absolute deltas.

## Proposed experiment

Run from the on-box `~/vllm-tune` checkout:

```bash
DOCKER="sudo docker" \
  bash scripts/experiments/gptoss20b_config_search/run_matrix.sh \
    --vllm-version v0.19.0 --gpu-mem 0.85 --configs LEAN
```

Diff vs OPT: `--block-size 64` (was 128), `--max-num-batched-tokens 8192` (was 16384), `num_speculative_tokens: 2` (was 3). bf16 + fp8 KV unchanged.

Pass conditions: `LEAN_<workload>_tok_s / OPT_<workload>_tok_s ≥ 1.05` for all three of decode @ c=128, prefill @ c=512, sharegpt @ c=256; **and** none of the three ratios is below 1.00 (no regression anywhere).

## Risks

- The compound effect is not strictly the product of individual wins — the three knobs interact (e.g. blk=64 + batch=8k together free additional activation memory the sampler workspace needs). The findings doc confirmed LEAN boots at GPU_MEM=0.95 where standalone BLK64 OOMed; at 0.85 LEAN should boot comfortably, but unverified.
- If LEAN doesn't beat OPT cleanly on this driver/CUDA stack we have a real conflict with the deploy/ default — file as a contradiction block on the v019 findings page.
