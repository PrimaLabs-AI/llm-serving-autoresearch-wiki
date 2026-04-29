---
title: "Prefix caching improves multi-turn agentic throughput by 20-40%"
type: hypothesis
tags: [serving, prefix-caching, agentic]
model: ""
engine: vllm
workload: multi-turn-agentic
status: open
expected_gain: "20-40% throughput improvement at concurrency=64"
confidence: high
effort: S
origin: human
created: 2026-04-29
updated: 2026-04-29
---

## Statement

Enabling prefix caching in vLLM (`--enable-prefix-caching`) will improve throughput by 20-40% for multi-turn agentic workloads at concurrency=64, because the workload has 60-80% prefix reuse across turns and PagedAttention's block-level KV cache can skip recomputation of shared prefixes.

## Rationale

Multi-turn agentic workflows accumulate context across turns. In a 5-turn agent task, turns 2-5 share the entire prior conversation as a prefix. Without prefix caching, the engine recomputes the KV cache for the shared prefix on every turn. With prefix caching, the engine reuses existing KV cache blocks for the shared portion, eliminating redundant prefill computation.

The [multi-turn agentic workload](../workloads/multi-turn-agentic.md) estimates ~60-80% prefix reuse across turns. vLLM's PagedAttention with `--enable-prefix-caching` should deduplicate these blocks.

## Proposed experiment

1. Launch vLLM with a standard config (no prefix caching) — record baseline throughput at concurrency levels [16, 32, 64, 128]
2. Launch vLLM with `--enable-prefix-caching` — record throughput at same concurrency levels
3. Use the multi-turn agentic workload profile (5 turns, growing context)
4. Measure: throughput (req/s), TTFT p50/p99, KV cache utilization

Expected delta: 20-40% throughput improvement, 30-50% TTFT reduction for turns 2+.

## Risks

- Prefix caching increases KV cache memory overhead (block metadata), potentially reducing max concurrency
- If the workload's prefix reuse is lower than estimated, gains will be smaller
- Block size misalignment with turn boundaries could reduce effective reuse

## Dependencies

- A GPU server with vLLM installed
- The model to test must be specified
