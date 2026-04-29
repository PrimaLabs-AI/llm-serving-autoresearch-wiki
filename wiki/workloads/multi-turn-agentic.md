---
title: "Multi-Turn Agentic Workflow"
type: workload
tags: [agentic, multi-turn, tool-use]
created: 2026-04-29
updated: 2026-04-29
---

Multi-turn agentic workflows where an LLM agent executes a task through repeated cycles of reasoning, tool invocation, and context accumulation. The defining characteristic is **growing context** — each turn appends the tool result to the conversation, so KV cache reuse across turns is critical for throughput.

## Request pattern

| Parameter | Value | Notes |
|---|---|---|
| Input length (turn 1) | 256–1024 tokens | Initial task prompt |
| Input length (turn N) | Previous context + 64–256 tokens | Growing context + new tool result |
| Output length per turn | 64–512 tokens | Reasoning + tool call or final answer |
| Typical turn count | 3–10 turns | Per agent task |
| Context growth rate | ~200–500 tokens/turn | Accumulated tool results |
| Max total context | 4K–32K tokens | Depends on task complexity |

## Concurrency characteristics

| Parameter | Value | Notes |
|---|---|---|
| Steady-state concurrency | 32–256 simultaneous agents | Each agent is a multi-turn session |
| Burst concurrency | 2–5x steady-state | Agent tasks tend to start in batches |
| Arrival pattern | Bursty | Agents spawn sub-agents or parallel tool calls |
| Session affinity | **High** — same session reuses KV cache | Critical for prefix caching |
| Prefix reuse | ~60–80% across turns | Prior conversation turns are identical |

## Metrics of interest

- **TTFT**: critical for user-facing agents (perceived responsiveness)
- **Throughput (req/s)**: total agent tasks completed per second
- **KV cache reuse rate**: percentage of KV cache blocks reused across turns (prefix caching effectiveness)
- **E2E latency per task**: total wall-clock time for all turns of one agent task

## Representative benchmark config

```bash
# vLLM benchmark
python -m vllm.entrypoints.openai.api_server \
  --model <model> \
  --enable-prefix-caching \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.90

# Synthetic workload: 128 concurrent sessions, 5 turns each,
# input growing by ~300 tokens per turn, output ~200 tokens per turn
python benchmarks/benchmark_serving.py \
  --backend vllm \
  --model <model> \
  --dataset-name custom \
  --multi-turn \
  --num-turns 5 \
  --concurrency 128 \
  --request-rate inf
```

## See also

- [Parallel Tool Use](parallel-tool-use.md)
- [Long Context RAG](long-context-rag.md)

## Sources
