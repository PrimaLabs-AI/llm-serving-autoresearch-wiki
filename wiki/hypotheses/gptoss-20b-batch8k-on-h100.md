---
title: "max_num_batched_tokens=8192 trades small prefill loss for decode gain on 20B"
type: hypothesis
tags: [serving, gpt-oss, h100, batching, vllm-tune]
model: gptoss-20b
engine: vllm
workload: multi-turn-agentic
hardware: h100
status: open
expected_gain: "≤-2% prefill tok/s vs OPT, +~3% decode tok/s, sharegpt within ±3%"
confidence: medium
effort: S
origin: human
created: 2026-05-04
updated: 2026-05-04
---

## Statement

The `BATCH8K` config (Eagle3 RedHatAI k=3 + bf16 + fp8 KV + `block_size=128` + `max_num_batched_tokens=8192`) on 1× H100 80 GB SXM at `gpu_memory_utilization=0.85` improves decode tok/s by ~3% over [OPT round 7](../experiments/2026-05-04-gptoss20b-h100-opt.md) at c=128 and loses at most 2% prefill tok/s at c=512, with sharegpt landing within ±3% of OPT.

## Rationale

The [config search plan](../sources/2026-04-gptoss-20b-config-search-plan.md) hypothesizes that 16k matmul is flop-bound on 120B but finishes too fast on 20B — at the smaller hidden size 16k batched-tokens lets the prefill matmul finish before downstream chunk-prefill scheduling can refill the pipeline, leaving cycles on the table. Halving to 8k tightens the chunk-cycle and gives decode steps interleaved between chunks more frequent slots.

The [v019 findings doc](../sources/2026-04-gptoss-20b-v019-findings.md) ablation supports this asymmetrically: at GPU_MEM=0.95, BATCH8K decode = 16,759 vs OPT 14,897 (+12.5%) and BATCH8K prefill = 4,307 vs OPT 3,386 (+27%) — and importantly **8k is what LEAN ships with**. On the OPT-style k=3 spec config alone (no other LEAN tweaks) the prefill gain is partly a reduction of the Eagle3 prefill penalty, not just a batch-size effect. We expect a smaller absolute swing here because the OPT round-7 baseline at GPU_MEM=0.85 is already tighter on KV than the 0.95 reference.

## Proposed experiment

Run from the on-box `~/vllm-tune` checkout:

```bash
DOCKER="sudo docker" \
  bash scripts/experiments/gptoss20b_config_search/run_matrix.sh \
    --vllm-version v0.19.0 --gpu-mem 0.85 --configs BATCH8K
```

Diff vs OPT: `--max-num-batched-tokens 8192` (replaces `--max-num-batched-tokens 16384`). All other flags identical.

Pass conditions: `BATCH8K_decode_tok_s / OPT_decode_tok_s ≥ 1.03`; `BATCH8K_prefill_tok_s / OPT_prefill_tok_s ≥ 0.98`; `|BATCH8K_sharegpt_tok_s / OPT_sharegpt_tok_s − 1| ≤ 0.03`.

## Risks

- The findings-doc BATCH8K cell ran at GPU_MEM=0.95 and saw +27% prefill, well above our pass-band. Either the OPT-vs-BATCH8K relationship at GPU_MEM=0.85 is much smaller, or our pass-band is too narrow — calibration against this experiment may force an update to expected gain.
- A smaller batch budget can starve the prefill scheduler when many requests arrive simultaneously, raising TTFT tail at c=512.
