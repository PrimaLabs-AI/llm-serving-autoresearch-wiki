---
title: "TensorRT-LLM — NVIDIA LLM Serving Engine"
type: engine
tags: [stub, serving, gpu, nvidia]
created: 2026-04-29
updated: 2026-04-29
commit: ""
---

*TensorRT-LLM is NVIDIA's optimized LLM serving engine built on TensorRT, targeting maximum throughput on NVIDIA GPUs.*

*Stub — expand when codebase is ingested under raw/code/tensorrt-llm.*

## Overview

TensorRT-LLM is NVIDIA's open-source library for deploying LLMs with high throughput on NVIDIA GPUs. It leverages TensorRT's graph optimization, kernel fusion, and GPU-specific tuning. Unlike vLLM and SGLang which are Python-first, TensorRT-LLM compiles the model into an optimized TensorRT engine ahead of time.

## Architecture

- **Engine compilation**: ahead-of-time model compilation via TensorRT; weight-only or full compilation
- **KV cache manager**: Paged KV cache with block-level management; supports CUDA graph capture for decode
- **Batching**: inflight batching (continuous batching) via the TensorRT-LLM runtime; supports context and generation requests interleaving
- **Attention**: fused multi-head attention, multi-query attention, grouped-query attention kernels optimized per GPU architecture

## Serving-relevant surfaces

*To be filled after ingestion. Expected knobs:*

| Category | Knobs | Notes |
|---|---|---|
| Engine build | `--dtype`, `--use_fp8`, `--gpt_attention_plugin`, `--context_fmha` | Build-time optimization flags |
| KV cache | `--max_batch_size`, `--max_seq_len`, `--tokens_per_block` | Memory pre-allocation |
| Batching | `--max_num_tokens` (runtime) | Per-iteration token budget |
| Parallelism | `--tensor_parallel`, `--pipeline_parallel` | Multi-GPU/multi-node |
| Quantization | `--quantization` (FP8, INT8, INT4 weight-only) | Post-training quantization |
| CUDA graphs | enabled by default for decode | Reduces kernel launch overhead |
| Speculative decoding | draft model integration | Supported via executor API |
| KV cache reuse | `--reuse` flag | Prefix caching support |

## Known strengths

- Highest single-GPU throughput for NVIDIA GPUs due to TensorRT kernel fusion
- Architecture-specific kernel auto-tuning (Hopper, Ada, Ampere)
- Tight integration with NVIDIA Triton Inference Server for production deployment
- Excellent FP8 and INT8 quantization support with minimal quality loss

## Known weaknesses for agentic workloads

- Ahead-of-time compilation adds deployment friction vs. Python-first engines
- Less flexible for rapid experimentation (engine rebuild required for some config changes)
- Smaller community and fewer third-party integrations than vLLM/SGLang

## See also

- [vLLM](vllm.md)
- [SGLang](sglang.md)

## Sources

- `raw/code/tensorrt-llm` (to be added)
