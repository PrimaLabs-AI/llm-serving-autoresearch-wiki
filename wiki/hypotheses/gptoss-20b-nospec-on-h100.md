---
title: "Dropping speculative decoding eliminates the 20s prefill TTFT seen in OPT round 7"
type: hypothesis
tags: [serving, gpt-oss, h100, eagle3, vllm-tune]
model: gptoss-20b
engine: vllm
workload: multi-turn-agentic
hardware: h100
status: open
expected_gain: "≥+15% prefill tok/s vs OPT, prefill TTFT p50 below 15s, ≤-25% decode tok/s, ≤-15% sharegpt tok/s"
confidence: high
effort: S
origin: human
created: 2026-05-04
updated: 2026-05-04
---

## Statement

The `NOSPEC` config (bf16 + fp8 KV + `block_size=128` + `max_num_batched_tokens=16384`, no Eagle3 speculation) on 1× H100 80 GB SXM at `gpu_memory_utilization=0.85` improves prefill output_tok_s by ≥15% over [OPT round 7](../experiments/2026-05-04-gptoss20b-h100-opt.md), reduces prefill TTFT p50 from 20.04 s to below 15 s, but loses at least 15% sharegpt and at least 25% decode throughput. NOSPEC isolates whether speculation itself drives the prefill TTFT cost we measured in OPT round 7.

## Rationale

OPT round 7 measured prefill TTFT p50 = 20.04 s at c=512 with 86 ms TPOT — the tightest cell in the matrix. The [v019 findings doc](../sources/2026-04-gptoss-20b-v019-findings.md) ablation reports NOSPEC at 5,000 prefill tok/s vs OPT's 3,386 (1.04× BASE vs 0.70× BASE) on the same workload at GPU_MEM=0.95: Eagle3's per-step draft-model overhead is the prefill regressor. The mechanism is vLLM's chunked-prefill scheduler interleaving prefill chunks with decode steps; with `num_speculative_tokens=3` enabled, every step also runs the draft model k times, and at c=512 prefill the per-step fixed overhead dominates the chunk budget.

The trade is severe in the other direction: NOSPEC loses Eagle3's decode multiplier entirely. Findings doc decode at c=128: OPT 14,897 vs NOSPEC 11,288 — NOSPEC pays ~24% of decode throughput.

## Proposed experiment

Run from the on-box `~/vllm-tune` checkout:

```bash
DOCKER="sudo docker" \
  bash scripts/experiments/gptoss20b_config_search/run_matrix.sh \
    --vllm-version v0.19.0 --gpu-mem 0.85 --configs NOSPEC
```

Diff vs OPT: drop `--speculative-config '...'` entirely. All other flags identical.

Pass conditions: `NOSPEC_prefill_tok_s / OPT_prefill_tok_s ≥ 1.15`, `NOSPEC_prefill_TTFT_p50 < 15 s`, `NOSPEC_decode_tok_s / OPT_decode_tok_s ≤ 0.75`, `NOSPEC_sharegpt_tok_s / OPT_sharegpt_tok_s ≤ 0.85`.

## Risks

- Eagle3 acceptance behavior on this driver/CUDA stack (580/13.0) may differ from the findings doc's environment — if the prefill regression is driver-mediated we'd see a smaller delta.
- NOSPEC frees ~1.6 GB of draft-model + draft KV memory; KV cache will grow correspondingly, so the comparison isn't strictly speculation-isolated. The matrix runs all configs at the same `gpu_memory_utilization=0.85`, so the budget recovers but is allocated differently.
