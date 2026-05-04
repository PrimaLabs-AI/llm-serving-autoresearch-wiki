---
title: "Stock vLLM 0.19.0 BASE config establishes the floor on 1× H100 80 GB SXM"
type: hypothesis
tags: [serving, gpt-oss, h100, baseline, vllm-tune]
model: gptoss-20b
engine: vllm
workload: multi-turn-agentic
hardware: h100
status: open
expected_gain: "OPT delivers >=1.5x output_tok_s on sharegpt vs BASE at per-replica peak concurrencies"
confidence: high
effort: S
origin: human
created: 2026-05-04
updated: 2026-05-04
---

## Statement

The `BASE` config from `vllm-tune`'s 9-config search matrix — stock vLLM 0.19.0 with no flags beyond `--tensor-parallel-size 1`, `--trust-remote-code`, and `--gpu-memory-utilization 0.85` (no Eagle3 spec, default `block_size=16`, default `max_num_batched_tokens`, `dtype=auto`, `kv-cache-dtype=auto`) — establishes the floor on 1× H100 80 GB SXM. The [OPT round 7](../experiments/2026-05-04-gptoss20b-h100-opt.md) result (16,192 sharegpt tok/s at c=256) should beat `BASE` by at least 1.5x on sharegpt at the per-replica peak concurrencies (decode @ c=128, prefill @ c=512, sharegpt @ c=256).

## Rationale

The findings doc [`gptoss-20b-v019-config-search-findings.md`](../sources/2026-04-gptoss-20b-v019-findings.md) measured BASE at 11,071 sharegpt tok/s (c=256) on the same 1x H100 + v0.19.0 + GPU_MEM=0.95 footprint. BASE is the indispensable reference: it is the only config where every gain attributable to bf16 promotion, fp8 KV cache, block-size tuning, batched-tokens sizing, and Eagle3 speculation is unambiguously visible as a non-zero delta.

## Proposed experiment

Run from the on-box `~/vllm-tune` checkout:

```bash
DOCKER="sudo docker" \
  bash scripts/experiments/gptoss20b_config_search/run_matrix.sh \
    --vllm-version v0.19.0 --gpu-mem 0.85 --configs BASE
```

Capture per-replica throughput for decode @ c=128, prefill @ c=512, sharegpt @ c=256. Pass condition: all three cells return `success_pct=100.0` and `OPT_sharegpt_tok_s / BASE_sharegpt_tok_s >= 1.5` at c=256.

## Risks

- BASE's KV cache, sized by stock vLLM 0.19's auto KV-cache-dtype path on 80 GB at 0.85 GPU_MEM, may saturate at c=512 on decode (the findings doc reports a 2.5x collapse at that point). The matrix runs decode at c=128 only, so we will not directly observe that collapse here, but TPOT may already be elevated due to higher KV pressure.
- Without bf16 promotion the MoE numerics may change relative to the OPT run — semantically equivalent within Eagle3-disabled MXFP4 MoE, but a regression on output quality should be flagged in the experiment write-up before declaring BASE valid.
