---
title: "Eagle3 k=2 trades draft compute for net sharegpt tok/s gain vs OPT k=3"
type: hypothesis
tags: [serving, gpt-oss, h100, eagle3, vllm-tune]
model: gptoss-20b
engine: vllm
workload: multi-turn-agentic
hardware: h100
status: open
expected_gain: "+~5% sharegpt tok/s vs OPT, ≤-3% decode tok/s, ≥+10% prefill tok/s"
confidence: medium
effort: S
origin: human
created: 2026-05-04
updated: 2026-05-04
---

## Statement

The `K2` config (Eagle3 RedHatAI draft with `num_speculative_tokens=2`, all other flags identical to `OPT`: bf16, fp8 KV, `block_size=128`, `max_num_batched_tokens=16384`) on 1× H100 80 GB SXM at `gpu_memory_utilization=0.85` improves sharegpt output_tok_s by ~5% over [OPT round 7](../experiments/2026-05-04-gptoss20b-h100-opt.md) and reduces prefill TTFT, while losing at most 3% decode tok/s.

## Rationale

OPT round 7 measured Eagle3 k=3 acceptance ratios of 100% / 76.9% / 62.5% across draft positions 0/1/2 — the third draft accepts only 62% of the time, paying for one extra draft-model forward per step regardless. The [v019 findings doc](../sources/2026-04-gptoss-20b-v019-findings.md) ablation isolates Eagle3 itself as the prefill regressor (k=3 → 0.70× BASE prefill, k=2 → 0.91× BASE) — k=2 trades roughly −9% draft-side compute (one fewer draft pass per step on a ~0.9 B model) for ≤−3% acceptance length. On sharegpt — where output lengths are mixed (~2 K mean) and decode dominates over prefill at c=256 — net throughput should rise.

## Proposed experiment

Run from the on-box `~/vllm-tune` checkout:

```bash
DOCKER="sudo docker" \
  bash scripts/experiments/gptoss20b_config_search/run_matrix.sh \
    --vllm-version v0.19.0 --gpu-mem 0.85 --configs K2
```

Diff vs OPT: `--speculative-config '{"method":"eagle3","model":"/models/RedHatAI/gpt-oss-20b-speculator.eagle3","num_speculative_tokens":2}'` (only the `num_speculative_tokens` field changes).

Pass condition: `K2_sharegpt_tok_s / OPT_sharegpt_tok_s ≥ 1.03`, `K2_prefill_tok_s / OPT_prefill_tok_s ≥ 1.10`, `K2_decode_tok_s / OPT_decode_tok_s ≥ 0.97`.

## Risks

- Acceptance-rate metric reformulation between vLLM versions (see findings doc caveat 6): the position-2 ratio of 62% measured on v0.19 may not directly map to predicted k=2 vs k=3 gains.
- k=2 may give back so much per-step decode gain that net sharegpt loses to OPT's k=3; the prior findings predict a small win but with ±10% run-to-run noise.
