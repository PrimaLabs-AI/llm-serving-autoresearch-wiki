---
title: "Qwen2.5-7B + prefix-caching on 1× H100 — sharegpt throughput improvement"
type: hypothesis
tags: [serving, qwen, h100, prefix-caching, ablation]
model: qwen2.5-7b
engine: vllm
workload: multi-turn-agentic
hardware: h100
status: refuted
expected_gain: "+10-20% sharegpt output_tok_s vs BASE; ≤noise on decode/prefill"
confidence: medium
effort: S
origin: human
created: 2026-05-04
updated: 2026-05-04
---

## Statement

Adding `--enable-prefix-caching` to the [Qwen2.5-7B BASE config](qwen2.5-7b-base-on-h100.md) on 1× H100 80 GB SXM delivers **≥10% improvement on sharegpt `output_tok_s` at c=512** vs BASE (8,282 tok/s). Decode and prefill cells should change by ≤noise (±5%) since those workloads have no prefix reuse.

Pass: sharegpt output_tok_s ≥ 9,110 (8,282 × 1.10) AND decode + prefill within 95-105% of BASE.

## Rationale

[BASE rerun experiment](../experiments/2026-05-04-qwen2.5-7b-base-rerun.md) established sharegpt = 8,282 output tok/s at c=512. The sharegpt workload draws from `sharegpt_prompts.txt` which contains real chat conversations — many start with similar system prompts and conversation history. Without prefix caching, every request rebuilds the KV cache from scratch even when the first N tokens are byte-identical to a recent request.

vLLM's `--enable-prefix-caching` (off by default in 0.19.0) hashes prefix blocks and reuses the cache across requests with shared prefixes. On sharegpt-style traffic the realized hit rate is typically 5-30% depending on prompt similarity, which translates roughly 1:1 to throughput improvement. We saw a 9.5% prefix cache hit rate logged on the gpt-oss-20B baseline (round 7) — Qwen sharegpt should be in the same ballpark.

Decode and prefill workloads use synthetic random prompts (no prefix reuse), so this knob shouldn't affect them.

## Proposed experiment

Direct `docker run` (no `run_matrix.sh`; that's gpt-oss-20B specific). Same flags as BASE plus `--enable-prefix-caching`:

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
      --enable-prefix-caching
```

Then run the same 3-cell sweep used in BASE: decode @ c=256, prefill @ c=1024, sharegpt @ c=512. Capture `vllm:gpu_prefix_cache_hit_rate` from `/metrics` at end-of-run — that's the load-bearing number for explaining any throughput delta.

## Risks

- **Sharegpt's `random` dataset variant has zero prefix reuse**, in which case the hit rate would be near 0% and we'd see no benefit. Need to confirm the sweep uses `--dataset-name sharegpt` (which loads from the real prompts file) and not the random fallback. Verify by checking `evalscope/.../sharegpt/c512/qwen/parallel_512_number_512/benchmark.log` for "loaded from sharegpt_prompts.txt" type lines.
- **Memory pressure**: prefix cache adds bookkeeping per block; KV cache budget shrinks slightly. Unlikely to matter at 80 GB on a 7B model but worth checking peak HBM.
- **Decode/prefill noise**: per the variance observation in the BASE rerun, single-cell runs have ~18% spread on decode. A 5% regression in decode could be noise rather than a real hit; consider 3 repeats if the BASE comparison is borderline.

## See also

- [BASE baseline](qwen2.5-7b-base-on-h100.md) (supported)
- [Round-7 OPT experiment](../experiments/2026-05-04-gptoss20b-h100-opt.md) — saw 9.5% prefix hit rate on gpt-oss baseline
- [vLLM engine page](../engines/vllm.md)
