---
title: "vLLM — LLM Serving Engine"
type: engine
tags: [stub, serving, gpu]
created: 2026-04-29
updated: 2026-04-29
commit: ""
---

*vLLM is a high-throughput LLM serving engine using PagedAttention for efficient KV cache management.*

*Stub — expand when codebase is ingested under raw/code/vllm.*

## Overview

vLLM is an open-source LLM serving engine developed by UC Berkeley's Sky Computing Lab. It introduced PagedAttention, which manages KV cache memory at the block level (analogous to virtual memory paging), enabling higher throughput under concurrent requests without requiring static memory pre-allocation.

## Architecture

- **Scheduler**: iteration-level continuous batching; decides which requests to preempt, swap, or schedule each step
- **KV cache manager**: block-level allocation via PagedAttention; supports swap-to-CPU and prefix caching
- **Batching**: dynamic batch formation at each scheduling step; prefill and decode can be interleaved (chunked prefill)
- **Attention backends**: xformers, FlashInfer, ROCm, TPU

## Serving-relevant surfaces

*To be filled after ingestion. Expected knobs:*

| Category | Knobs | Notes |
|---|---|---|
| Scheduler | `--max-num-seqs`, `--max-num-batched-tokens`, `--scheduling-policy` | Controls concurrency ceiling |
| KV cache | `--block-size`, `--gpu-memory-utilization`, `--swap-space`, `--enable-prefix-caching` | Memory management |
| Batching | `--enable-chunked-prefill`, `--max-num-batched-tokens` | Prefill/decode interleaving |
| Parallelism | `--tensor-parallel-size`, `--pipeline-parallel-size`, `--data-parallel-size` | Multi-GPU/multi-node |
| Quantization | `--quantization`, `--dtype` | FP8, GPTQ, AWQ, etc. |
| CUDA graphs | `--enforce-eager`, `--num-cudagraph-capture-sizes` | Decode overhead reduction |
| Speculative decoding | `--speculative-model`, `--num-speculative-tokens` | Draft-model acceleration |
| Disaggregated | `--disaggregated-prefill` | Separate prefill/decode workers |

## See also

- [SGLang](sglang.md)
- [TensorRT-LLM](tensorrt-llm.md)

## Sources

- `raw/code/vllm` (to be added)
