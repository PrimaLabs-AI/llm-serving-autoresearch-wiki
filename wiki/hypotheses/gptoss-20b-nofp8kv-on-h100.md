---
title: "Dropping fp8 KV-cache quantization gives back capacity but adds quant cost on 20B"
type: hypothesis
tags: [serving, gpt-oss, h100, kv-cache, fp8, vllm-tune]
model: gptoss-20b
engine: vllm
workload: multi-turn-agentic
hardware: h100
status: open
expected_gain: "0-3% loss across all workloads vs OPT at per-replica peak concurrencies (KV not the binding capacity constraint on 80 GB at c≤512)"
confidence: medium
effort: S
origin: human
created: 2026-05-04
updated: 2026-05-04
---

## Statement

The `NOFP8KV` config (Eagle3 RedHatAI k=3 + bf16 + `block_size=128` + `max_num_batched_tokens=16384`, KV-cache-dtype=auto, i.e. native bf16/fp16) on 1× H100 80 GB SXM at `gpu_memory_utilization=0.85` performs within 3% of [OPT round 7](../experiments/2026-05-04-gptoss20b-h100-opt.md) on all three workloads at the per-replica peak concurrencies. fp8 KV is not the bottleneck on 20B at 80 GB up to c=512 — capacity is sufficient without quantization, and the per-token quant/dequant cost may exceed the capacity gain.

## Rationale

The [config search plan](../sources/2026-04-gptoss-20b-config-search-plan.md) flags fp8 KV as inherited from 120B, where capacity drove the choice (KV cache is the dominant memory consumer at 4096-token contexts on 120B). On 20B at 80 GB SXM the [80GB OOM finding](../sources/2026-04-gptoss-20b-h100-oom.md) shows KV cache at GPU_MEM=0.85 is ~50 GiB — far above what the matrix's c=512 prefill cell needs. The per-attention-step fp8 quant of KV pre-write and dequant on read costs ~0.5-1% wall-clock per kernel call; if we don't need the capacity, we lose this overhead.

The [v019 findings doc](../sources/2026-04-gptoss-20b-v019-findings.md) does not include a clean NOFP8KV cell in the published table — this experiment fills that gap. The expected outcome is a small symmetric loss (or wash) compared to OPT, signalling that fp8 KV is essentially free overhead under this workload mix.

## Proposed experiment

Run from the on-box `~/vllm-tune` checkout:

```bash
DOCKER="sudo docker" \
  bash scripts/experiments/gptoss20b_config_search/run_matrix.sh \
    --vllm-version v0.19.0 --gpu-mem 0.85 --configs NOFP8KV
```

Diff vs OPT: drop `--kv-cache-dtype fp8` (KV-cache-dtype defaults to `auto`). All other flags identical.

Pass conditions: `0.97 ≤ NOFP8KV_<workload>_tok_s / OPT_<workload>_tok_s ≤ 1.03` for each of decode @ c=128, prefill @ c=512, sharegpt @ c=256.

## Risks

- KV-cache-dtype=auto on H100 + Hopper FlashAttention path may still pick fp8 if the model config requests it; we should verify from the boot log that KV cache is materialized in bf16/fp16, not fp8.
- If 20B's KV cache at c=512 prefill is tighter than the OOM-finding analysis predicts, NOFP8KV could trigger preemption/swap thrash and lose more than 3% — that would itself be the load-bearing finding for the matrix.
- Boot-time KV cache size differs from OPT (~2× larger if dropped to bf16); allocator fragmentation may differ enough that a marginal OOM appears.
