---
title: "SGLang RadixAttention outperforms vLLM prefix caching for multi-turn agentic workloads"
type: hypothesis
tags: [serving, cross-engine, agentic, prefix-caching]
model: ""
engine: sglang
workload: multi-turn-agentic
status: retired
expected_gain: "15-30% higher throughput vs vLLM at concurrency=64"
confidence: medium
effort: M
origin: human
created: 2026-04-29
updated: 2026-05-01
hardware: any
retired_reason: "vLLM and SGLang in the same venv collide on torch/transformers/flashinfer pins. Reopen once setup-cuda.sh splits per-engine venvs (slice 9 follow-up)."
---

## Statement

SGLang's RadixAttention will achieve 15-30% higher throughput than vLLM with prefix caching for multi-turn agentic workloads, because RadixAttention uses a radix-tree structure that handles variable-length prefix matching more efficiently than vLLM's hash-based block matching.

## Rationale

SGLang's RadixAttention maintains a radix tree of KV cache blocks indexed by token prefix. This allows O(k) prefix lookup where k is the tree depth, vs. vLLM's approach which hashes block contents. For multi-turn conversations where prefixes grow incrementally, the radix tree can extend existing entries rather than computing new hashes. Additionally, RadixAttention handles eviction more gracefully — it can evict the least-recently-used leaf without invalidating shared internal nodes.

The [multi-turn agentic workload](../workloads/multi-turn-agentic.md) has growing context across turns, which should benefit from RadixAttention's incremental tree extension.

## Proposed experiment

1. Run the multi-turn agentic workload on SGLang (default RadixAttention config)
2. Run the same workload on vLLM with `--enable-prefix-caching`
3. Same model, same hardware, same concurrency levels [16, 32, 64, 128]
4. Compare: throughput (req/s), TTFT p50/p99, KV cache utilization

Expected delta: SGLang 15-30% higher throughput, especially at higher concurrency where eviction pressure increases.

## Risks

- SGLang's overall scheduler may have different tradeoffs that offset RadixAttention's advantage
- The comparison conflates engine-level differences (scheduler, attention backend) with the prefix caching mechanism
- Results may not generalize across all model sizes

## Dependencies

- Both vLLM and SGLang installed on the same GPU server
- Same model accessible to both engines
