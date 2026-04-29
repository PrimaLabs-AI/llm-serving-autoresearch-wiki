---
title: "Structured Output Workflow"
type: workload
tags: [structured-output, json, constrained-decoding]
created: 2026-04-29
updated: 2026-04-29
---

Workloads requiring structured output (JSON, XML, function call schemas) enforced via constrained/grammar-based decoding. The defining characteristic is that the **logit mask applied at each decode step constrains the output distribution**, which can reduce throughput by breaking CUDA graph captures or requiring per-step CPU-GPU synchronization.

## Request pattern

| Parameter | Value | Notes |
|---|---|---|
| Input length | 256–4K tokens | Instructions + schema definition |
| Output length | 64–512 tokens | JSON/tool call output |
| Schema complexity | Simple (key-value) to complex (nested arrays) | Affects constraint overhead |
| Multi-turn | Common — tool calls followed by interpretation | Agent loop pattern |
| Max total context | 1K–16K tokens | Moderate |

## Concurrency characteristics

| Parameter | Value | Notes |
|---|---|---|
| Steady-state concurrency | 32–256 concurrent requests | Similar to agentic workloads |
| Burst concurrency | High | Agent frameworks dispatch tool calls in batches |
| Arrival pattern | Bursty | Batch tool call dispatch |
| Session affinity | High for multi-turn | Tool call → interpretation → next action |
| Prefix reuse | Moderate | Same system prompt, different schemas |

## Metrics of interest

- **Throughput degradation from constraints**: tokens/s with vs. without structured output
- **TTFT**: may increase if constraint compilation blocks prefill
- **CUDA graph compatibility**: whether constrained decoding breaks graph capture
- **Schema compilation overhead**: one-time cost per unique schema

## Representative benchmark config

```bash
# SGLang with JSON schema constraint
python -m sglang.launch_server \
  --model-path <model> \
  --enable-overlap-schedule

# Benchmark with JSON schema enforcement
python -m sglang.bench_serving \
  --backend sglang \
  --model <model> \
  --constrained-decoding \
  --json-schema <schema-file> \
  --concurrency 128
```

## See also

- [Multi-Turn Agentic](multi-turn-agentic.md)
- [Parallel Tool Use](parallel-tool-use.md)

## Sources
