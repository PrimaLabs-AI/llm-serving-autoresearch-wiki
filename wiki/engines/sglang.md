---
title: "SGLang — Structured Generation Language Serving Engine"
type: engine
tags: [stub, serving, gpu]
created: 2026-04-29
updated: 2026-04-29
commit: ""
---

*SGLang is a fast serving engine for LLMs with radix attention for automatic prefix caching and efficient KV cache reuse.*

*Stub — expand when codebase is ingested under raw/code/sglang.*

## Overview

SGLang (Structured Generation Language) is an open-source LLM serving engine developed by the SGLang team (originally from UC Berkeley). Its key innovation is **RadixAttention**, a radix-tree-based KV cache that automatically detects and reuses shared prefixes across requests — critical for multi-turn agentic workloads where conversation context accumulates.

## Architecture

- **Scheduler**: continuous batching with priority-based scheduling
- **KV cache manager**: RadixAttention — radix-tree-based prefix matching for automatic KV cache reuse across requests
- **Batching**: iteration-level scheduling with chunked prefill support
- **Attention backends**: FlashInfer, Triton, custom CUDA kernels

## Serving-relevant surfaces

*To be filled after ingestion. Expected knobs:*

| Category | Knobs | Notes |
|---|---|---|
| Scheduler | `--max-running-requests`, `--max-total-tokens`, `--schedule-policy` | Concurrency control |
| KV cache | `--mem-fraction-static`, `--chunk-prefill-size` | Memory allocation |
| Prefix caching | RadixAttention (automatic) | No explicit toggle — inherent in design |
| Parallelism | `--tp`, `--dp`, `--pp` | Tensor/data/pipeline parallel |
| Quantization | `--quantization`, `--dtype` | FP8, GPTQ, AWQ, etc. |
| CUDA graphs | `--disable-cuda-graph` | Default enabled |
| Speculative decoding | `--speculative-algo`, `--speculative-num-steps` | Eagle/medusa |
| Overlap scheduling | `--enable-overlap-schedule` | Overlap prefill and decode |
| DP attention | `--enable-dp-attention` | Data-parallel attention for higher throughput |

## Known strengths for agentic workloads

- RadixAttention automatically reuses KV cache for shared prefixes across multi-turn conversations
- Native support for structured output (JSON schema) with constrained decoding
- Overlap scheduling can interleave prefill and decode for better GPU utilization under bursty loads

## See also

- [vLLM](vllm.md)
- [TensorRT-LLM](tensorrt-llm.md)

## Sources

- `raw/code/sglang` (to be added)
