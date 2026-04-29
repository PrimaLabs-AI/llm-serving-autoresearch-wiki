---
title: "Chain-of-Thought Reasoning Workflow"
type: workload
tags: [reasoning, long-output]
created: 2026-04-29
updated: 2026-04-29
---

Long-form reasoning tasks (math, code generation, planning) where the model produces extended chain-of-thought outputs. The defining characteristic is **short input, long output** — this stresses decode throughput and CUDA graph efficiency since the decode phase dominates.

## Request pattern

| Parameter | Value | Notes |
|---|---|---|
| Input length | 128–2K tokens | Task prompt |
| Output length | 1K–16K tokens | Extended reasoning chain |
| Multi-turn | Sometimes (re-plan, fix errors) | Context grows with prior reasoning |
| Max total context | 2K–32K tokens | Input + accumulated output |

## Concurrency characteristics

| Parameter | Value | Notes |
|---|---|---|
| Steady-state concurrency | 16–128 concurrent requests | Long decode keeps requests in-flight |
| Burst concurrency | Moderate | Batch problem submission |
| Arrival pattern | Poisson to bursty | Depends on usage (homework, batch eval) |
| Session affinity | Low-moderate | Mostly single-turn |
| Prefix reuse | Low | Independent reasoning tasks |

## Metrics of interest

- **TPOT (time per output token)**: decode speed is the bottleneck
- **Throughput (output tokens/s)**: aggregate decode throughput
- **CUDA graph hit rate**: decode benefits heavily from graph capture
- **Speculative decoding speedup**: draft models can accelerate long decode sequences

## Representative benchmark config

```bash
# vLLM with speculative decoding
python -m vllm.entrypoints.openai.api_server \
  --model <model> \
  --speculative-model <draft-model> \
  --num-speculative-tokens 5 \
  --gpu-memory-utilization 0.90

python benchmarks/benchmark_serving.py \
  --backend vllm \
  --model <model> \
  --dataset-name custom \
  --input-len 512 \
  --output-len 4096 \
  --concurrency 64 \
  --request-rate inf
```

## See also

- [Multi-Turn Agentic](multi-turn-agentic.md)

## Sources
