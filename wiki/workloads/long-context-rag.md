---
title: "Long Context RAG Workflow"
type: workload
tags: [rag, long-context, retrieval]
created: 2026-04-29
updated: 2026-04-29
---

Retrieval-augmented generation with large retrieved context. The defining characteristic is **long input, short output** — the prompt contains retrieved documents (10K–100K+ tokens) and the model produces a concise answer. This stresses prefill throughput and KV cache memory capacity.

## Request pattern

| Parameter | Value | Notes |
|---|---|---|
| Input length | 10K–128K tokens | Retrieved documents + query |
| Output length | 64–512 tokens | Concise answer |
| Multi-turn | Optional follow-up questions | Reuses prior RAG context |
| Max total context | 10K–128K tokens | Bounded by model's context window |

## Concurrency characteristics

| Parameter | Value | Notes |
|---|---|---|
| Steady-state concurrency | 8–64 concurrent requests | Long prefill limits throughput |
| Burst concurrency | Low-moderate | Users don't batch RAG queries |
| Arrival pattern | Poisson-like | Independent user queries |
| Session affinity | Moderate for follow-up turns | Prior context reuse |
| Prefix reuse | Low across independent queries | Different retrieved docs each time |

## Metrics of interest

- **TTFT**: dominated by prefill time for long inputs; critical UX metric
- **Throughput (input tokens/s)**: prefill throughput ceiling
- **Max concurrent long-context requests**: limited by KV cache memory
- **KV cache memory per request**: scales linearly with input length

## Representative benchmark config

```bash
# vLLM with chunked prefill
python -m vllm.entrypoints.openai.api_server \
  --model <model> \
  --enable-chunked-prefill \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.90

python benchmarks/benchmark_serving.py \
  --backend vllm \
  --model <model> \
  --dataset-name longbench \
  --concurrency 32 \
  --request-rate inf
```

## See also

- [Multi-Turn Agentic](multi-turn-agentic.md)

## Sources
