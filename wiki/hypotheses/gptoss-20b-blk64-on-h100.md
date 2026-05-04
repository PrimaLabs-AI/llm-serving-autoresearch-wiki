---
title: "block_size=64 frees KV-cache budget on 20B vs OPT's 128 (tuned for 120B GQA)"
type: hypothesis
tags: [serving, gpt-oss, h100, kv-cache, vllm-tune]
model: gptoss-20b
engine: vllm
workload: multi-turn-agentic
hardware: h100
status: open
expected_gain: "+~3-8% sharegpt tok/s vs OPT, ≥+5% decode tok/s under high concurrency, prefill within ±5%"
confidence: low
effort: S
origin: human
created: 2026-05-04
updated: 2026-05-04
---

## Statement

The `BLK64` config (Eagle3 RedHatAI k=3 + bf16 + fp8 KV + `block_size=64` + `max_num_batched_tokens=16384`) on 1× H100 80 GB SXM at `gpu_memory_utilization=0.85` matches or exceeds [OPT round 7](../experiments/2026-05-04-gptoss20b-h100-opt.md) on decode and sharegpt by 3-8%, with prefill within ±5%. The OPT default of `block_size=128` was tuned for 120B's wider GQA group size; on 20B the smaller block frees KV budget for more concurrent sequences before swap pressure begins.

## Rationale

The [config search plan](../sources/2026-04-gptoss-20b-config-search-plan.md) identifies block_size as inherited from the 120B optimization round, where 128 made sense given 120B's larger per-block KV footprint. On 20B with 64 KV heads at the smaller hidden size, `block_size=128` over-provisions per-block memory and increases internal fragmentation in the paged KV-cache allocator. block_size=64 cuts per-block allocation in half, raising effective KV capacity and reducing tail latency from preemption/swap.

The [v019 findings doc](../sources/2026-04-gptoss-20b-v019-findings.md) reports BLK64 standalone OOMed at GPU_MEM=0.95 on that environment — the smaller block-table adds overhead the sampler workspace can't absorb at 0.95. **At GPU_MEM=0.85 we have 4 GiB more slack, so this should boot.** If it still OOMs at 0.85, that is itself a load-bearing finding for the matrix.

## Proposed experiment

Run from the on-box `~/vllm-tune` checkout:

```bash
DOCKER="sudo docker" \
  bash scripts/experiments/gptoss20b_config_search/run_matrix.sh \
    --vllm-version v0.19.0 --gpu-mem 0.85 --configs BLK64
```

Diff vs OPT: `--block-size 64` (replaces `--block-size 128`). All other flags identical.

Pass conditions: container boots successfully (no OOM at 0.85 GPU_MEM); `BLK64_sharegpt_tok_s / OPT_sharegpt_tok_s ≥ 1.03`; `BLK64_decode_tok_s / OPT_decode_tok_s ≥ 1.05`; `|BLK64_prefill_tok_s / OPT_prefill_tok_s − 1| ≤ 0.05`.

## Risks

- **OOM at 0.85 GPU_MEM is the highest risk** — findings doc only confirmed boot at GPU_MEM=0.90 standalone. If BLK64 OOMs we file an `invalid` verdict and note the GPU_MEM threshold.
- Smaller blocks can reduce kernel efficiency in attention prefix computation. On 20B's narrower layers this should be small but unverified.
- The matrix only tests the per-replica peak concurrencies; the full curve (where BLK64's KV headroom would matter most at c=512+) is out of scope here.
