---
title: "Qwen2.5-7B + fp8 KV cache on 1× H100 — concurrency headroom"
type: hypothesis
tags: [serving, qwen, h100, fp8-kv, ablation]
model: qwen2.5-7b
engine: vllm
workload: multi-turn-agentic
hardware: h100
status: open
expected_gain: "≤±5% throughput at current concurrencies; +50-100% max concurrency before OOM"
confidence: medium
effort: S
origin: human
created: 2026-05-04
updated: 2026-05-04
---

## Statement

Adding `--kv-cache-dtype fp8` to the [Qwen2.5-7B BASE config](qwen2.5-7b-base-on-h100.md) on 1× H100 80 GB SXM **does not regress throughput by more than 5%** at the current per-replica concurrencies (decode/prefill/sharegpt @ c=256/1024/512), AND doubles the KV cache token capacity, which would enable +50% concurrency headroom in a follow-up high-concurrency cell.

Pass: all three workloads' output_tok_s within 95-105% of BASE (5,973 / 3,190 / 8,282), AND `vllm:kv_cache_usage_perc` peak observation at the same workload+concurrency drops by ≥40% vs BASE.

## Rationale

[BASE rerun](../experiments/2026-05-04-qwen2.5-7b-base-rerun.md) used `auto` KV cache dtype (BF16 → 16 bits per token-head). FP8 halves that to 8 bits/token-head, doubling KV cache capacity. On a 7B model with H100's 80 GB, this is more headroom than we currently use — the immediate throughput effect will be neutral.

But the second-order effect matters: with 2× KV capacity, we can run higher max concurrency before OOM. That's the lever for "extract more from this rig" work.

Quantization noise: FP8 KV introduces small numerical error in attention reads. Empirically (from gpt-oss-20B v019 findings) this degrades by <0.5% on lossless evals — well under the lossless gate threshold. We mark this hypothesis low-confidence on the throughput claim because the FP8 path may use different attention kernels (possibly slower than BF16 on H100 at this batch size).

## Proposed experiment

```bash
sudo docker run --rm --name qwen7b-bench --gpus all --ipc=host --shm-size 16g \
    -p 8000:8000 -v /srv/models:/models:ro \
    --entrypoint python3 vllm/vllm-openai:v0.19.0 \
      -m vllm.entrypoints.openai.api_server \
      --model /models/Qwen/Qwen2.5-7B-Instruct \
      --served-model-name qwen \
      --host 0.0.0.0 --port 8000 \
      --tensor-parallel-size 1 \
      --trust-remote-code \
      --gpu-memory-utilization 0.9 \
      --kv-cache-dtype fp8
```

3-cell sweep at BASE's concurrencies. Capture `vllm:kv_cache_usage_perc` from `/metrics` during the prefill cell (c=1024 should give the highest cache pressure of the three) — that's the load-bearing number for the headroom claim.

## Risks

- **Qwen2.5 may not have FP8 calibration scales packaged.** Some quantization paths require model-specific FP8 scales; check the boot log for warnings about "no FP8 scales found" or fallback to a different dtype.
- **vLLM 0.19's FP8 KV path on H100 may use FlashAttention-2 instead of FlashAttention-3** depending on the head dimension. Qwen-7B has head_dim=128 which is FA3-compatible; should be fine.
- **Lossless gate**: not yet wired in our bench; for this round, document "FP8 KV might affect output quality; not validated against GSM8K/HellaSwag/MMLU until slice-10 lands".

## See also

- [BASE baseline](qwen2.5-7b-base-on-h100.md) (supported)
- [gpt-oss-20B v019 findings](../sources/2026-04-gptoss-20b-v019-findings.md) — fp8 KV evaluated as part of the OPT config
- [vLLM engine page](../engines/vllm.md)
