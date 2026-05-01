---
title: "vLLM"
type: engine
tags: [engine, vllm, paged-attention]
commit: ""
supported_hardware: [h100, h200, b200, mi300x]
created: 2026-04-29
updated: 2026-05-01
---

vLLM is a high-throughput, memory-efficient LLM serving engine developed at UC Berkeley's Sky Computing Lab. It introduced PagedAttention, a KV cache management scheme that treats GPU memory as virtual pages — eliminating fragmentation and enabling high concurrency without static pre-allocation.

## Overview

vLLM serves as the de-facto open-source reference implementation for production LLM serving. Its core contribution is PagedAttention, which maps key/value cache blocks to non-contiguous physical GPU memory pages, much like OS virtual memory. This prevents the pre-allocation waste that plagued earlier engines (where each request reserved a full worst-case KV buffer) and raises effective GPU memory utilization to 90–95 % under mixed-length workloads.

Beyond memory efficiency, vLLM uses iteration-level continuous batching: every scheduling step, the engine assembles the largest valid batch from all queued requests (both prefilling new tokens and decoding in-flight ones), rather than waiting for a full batch to arrive or a fixed-duration window to expire. This eliminates the head-of-line blocking that degrades tail latency under load.

vLLM exposes an OpenAI-compatible REST API via `vllm.entrypoints.openai.api_server`, making it a drop-in replacement for OpenAI's `/v1/completions` and `/v1/chat/completions` endpoints.

## Architecture

### Scheduler

The scheduler runs once per iteration (decode step) and makes three decisions:

1. **Which waiting requests to prefill** — up to `max_num_batched_tokens` prompt tokens can be processed in one step. New requests are prefilled in chunks when chunked prefill is enabled.
2. **Which in-flight requests to decode** — up to `max_num_seqs` sequences can be decoded simultaneously.
3. **Whether to preempt or swap** — when KV cache capacity is exhausted, the scheduler preempts the lowest-priority in-flight request (evicting its blocks) or swaps it to CPU DRAM (`--swap-space`).

The scheduling policy (`--scheduling-policy`) controls priority ordering: `fcfs` (first-come-first-served, default) or `priority` (user-supplied per-request priority).

### KV Cache Manager (PagedAttention)

The BlockManager maintains a pool of fixed-size physical KV blocks (`--block-size`, default 16 tokens per block). Each sequence holds a virtual block table that maps logical KV positions to physical blocks. Blocks are reference-counted, allowing:

- **Copy-on-write sharing** for parallel sampling (multiple completions of the same prompt share the prompt's KV blocks until divergence).
- **Prefix caching** (`--enable-prefix-caching`): blocks whose token sequence matches a previously seen prefix are reused across requests — critical for agentic workloads with shared system prompts.
- **CPU swap-space eviction**: full KV blocks are moved to pinned CPU RAM when GPU pages run out, then swapped back when the sequence is rescheduled.

### Batching Strategy

vLLM supports two batching modes:

- **Standard continuous batching** (default): prefill and decode requests share a single forward pass. Prefill tokens are processed in one shot; decode steps produce one token per in-flight sequence.
- **Chunked prefill** (`--enable-chunked-prefill`): long prompt prefills are split into fixed-size chunks (`--max-num-batched-tokens` controls the chunk size) and interleaved with decode steps. This bounds TTFT variance at the cost of slightly lower prefill throughput and is the recommended mode for latency-sensitive agentic workloads.

### Attention Backends

vLLM selects the attention backend based on hardware:

| Backend | Platform | Notes |
|---|---|---|
| FlashInfer | NVIDIA (CUDA) | Default on H100/H200/B200; supports paged-KV, FP8 |
| xformers | NVIDIA (CUDA) | Legacy fallback |
| FlashAttention-2 | NVIDIA (CUDA) | Used when FlashInfer not available |
| ROCm Flash Attention | AMD (MI300X) | Default on ROCm |
| TPU Pallas | Google TPU | Via `tpu-inference` backend |

## Key Abstractions

| Class | Location | Role |
|---|---|---|
| `LLMEngine` | `vllm/engine/llm_engine.py` | Synchronous core: scheduler + model executor orchestration |
| `AsyncLLMEngine` | `vllm/engine/async_llm_engine.py` | Async wrapper for the API server; drives the step loop |
| `Worker` | `vllm/worker/worker.py` | Per-GPU process; holds model weights + KV cache |
| `ModelRunner` | `vllm/worker/model_runner.py` | Executes model forward pass; manages CUDA graphs |
| `BlockManager` | `vllm/core/block_manager.py` | PagedAttention block allocator; owns the GPU/CPU block pools |
| `Scheduler` | `vllm/core/scheduler.py` | Iteration-level batch assembly; preemption/swap logic |
| `SamplingParams` | `vllm/sampling_params.py` | Per-request sampling config (temperature, top-p, stop tokens) |
| `SequenceGroup` | `vllm/sequence.py` | A request with all its output sequences; unit of scheduling |

## Entry Points

### OpenAI-Compatible API Server

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 256 \
  --enable-prefix-caching
```

Key server flags beyond the serving surfaces table below:
- `--host` / `--port` — bind address (default `0.0.0.0:8000`)
- `--served-model-name` — alias exposed in `/v1/models`
- `--max-model-len` — override maximum context length
- `--tokenizer-pool-size` — async tokenization workers
- `--disable-log-requests` — suppress per-request logging in high-QPS deployments

### Programmatic (Offline) Inference

```python
from vllm import LLM, SamplingParams
llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct",
          tensor_parallel_size=1, gpu_memory_utilization=0.90)
outputs = llm.generate(["Hello, my name is"], SamplingParams(max_tokens=50))
```

### Benchmark Harness Integration

The `benchmark_harness.py` in this repo launches vLLM via:
```
python -m vllm.entrypoints.openai.api_server
```
with flags driven by the `ENGINE_COMMANDS["vllm"]["flag_map"]` dict (see `benchmark_harness.py` lines 36–55).

## Serving-Relevant Surfaces

All flags below are accepted by `benchmark_harness.py`'s `flag_map` for vLLM (source: `benchmark_harness.py` lines 36–55). CLI spellings use `--` prefix; `benchmark_harness.py` config keys use `snake_case` without dashes.

| Config Key | CLI Flag | Type | Default | Category | Effect |
|---|---|---|---|---|---|
| `max_num_seqs` | `--max-num-seqs` | int | 256 | Scheduler | Maximum number of sequences (requests) in a single iteration; hard ceiling on concurrency. Increase for higher throughput; decrease to bound TTFT. |
| `max_num_batched_tokens` | `--max-num-batched-tokens` | int | 2048 | Scheduler / Batching | Total token budget per iteration across all sequences. With chunked prefill, this controls chunk size. Larger values raise throughput; smaller values lower TTFT variance. |
| `gpu_memory_utilization` | `--gpu-memory-utilization` | float | 0.90 | KV cache | Fraction of GPU HBM reserved for the KV cache block pool (after model weights). Values 0.85–0.95 are typical; too high risks OOM from activation memory spikes. |
| `enable_prefix_caching` | `--enable-prefix-caching` | bool | false | KV cache | Reuse KV blocks for requests sharing an identical prefix (system prompt, few-shot examples). Critical for multi-turn agentic workloads. No latency cost when prefix is not found. |
| `enable_chunked_prefill` | `--enable-chunked-prefill` | bool | false | Batching | Split prefill into `max_num_batched_tokens`-sized chunks interleaved with decode. Bounds TTFT at the cost of slightly lower prefill throughput. Recommended for latency-sensitive agentic workloads. |
| `block_size` | `--block-size` | int | 16 | KV cache | Tokens per KV cache page. Smaller blocks reduce internal fragmentation but increase block-table overhead. 16 is optimal for most shapes; try 32 for very long sequences. |
| `tensor_parallel_size` | `--tensor-parallel-size` | int | 1 | Parallelism | Number of GPUs across which model weights are split via tensor parallelism. Must divide num_attention_heads evenly. Use 2–8 for models that don't fit on a single GPU. |
| `pipeline_parallel_size` | `--pipeline-parallel-size` | int | 1 | Parallelism | Number of pipeline stages across GPUs. Rarely used in serving (adds pipeline bubble latency); useful when a model is too large for TP alone. |
| `data_parallel_size` | `--data-parallel-size` | int | 1 | Parallelism | Number of independent engine replicas sharing the same model; each replica gets its own KV cache and scheduler. Requires a load balancer in front. |
| `quantization` | `--quantization` | str | None | Quantization | Weight/activation quantization method. Supported: `fp8` (H100/H200/B200 native), `awq`, `gptq`, `squeezellm`, `marlin`. FP8 recommended for highest throughput on Hopper/Blackwell GPUs. |
| `dtype` | `--dtype` | str | `auto` | Quantization | Compute dtype for activations and non-quantized weights. `auto` resolves to `bfloat16` on Ampere+. Override with `float16` for older GPUs or `float32` for debugging. |
| `enforce_eager` | `--enforce-eager` | bool | false | CUDA graphs | Disable CUDA graph capture and run in eager mode. Dramatically slows decode (removes graph-replay benefit of ~30–50 % latency savings). Use only for debugging. |
| `swap_space` | `--swap-space` | int (GiB) | 4 | KV cache | CPU RAM reserved for swapping preempted KV blocks. Increase for workloads with high concurrency and variable-length requests where GPU KV cache is frequently exhausted. |

## Tunable Knobs

The following table covers the full set of knobs exposed in `benchmark_harness.py` plus additional serving knobs for manual tuning:

| Flag | Type | Default | Expected Effect on Throughput | Expected Effect on TTFT / TPOT |
|---|---|---|---|---|
| `--max-num-seqs` | int | 256 | +throughput as it rises (up to GPU memory limit) | +TTFT as it rises (more queuing) |
| `--max-num-batched-tokens` | int | 2048 | +throughput with larger values | −TTFT variance with smaller values (chunked prefill) |
| `--gpu-memory-utilization` | float | 0.90 | +throughput with higher values (more KV pages) | neutral |
| `--enable-prefix-caching` | bool | false | +20–40 % throughput on agentic workloads | −TTFT for cached prefixes (near-zero) |
| `--enable-chunked-prefill` | bool | false | neutral to slight − | −TTFT p99 by 30–50 % at high concurrency |
| `--block-size` | int | 16 | minor; 32 slightly better for very long seqs | neutral |
| `--tensor-parallel-size` | int | 1 | +throughput until TP overhead dominates (>8) | −TTFT (parallel prefill) |
| `--pipeline-parallel-size` | int | 1 | −throughput (pipeline bubble); use only when TP insufficient | +TTFT bubble |
| `--data-parallel-size` | int | 1 | linear throughput scaling (independent replicas) | neutral |
| `--quantization fp8` | str | None | +50–80 % more KV pages fit; +throughput at high concurrency | neutral |
| `--dtype bfloat16` | str | auto | reference baseline | reference |
| `--enforce-eager` | bool | false | −30–50 % throughput (no CUDA graph) | debug only |
| `--swap-space` | int | 4 | minor at typical concurrency; prevents preemption at burst | prevents TTFT spikes from re-prefill |
| `--scheduling-policy priority` | str | fcfs | neutral on aggregate throughput | reduces TTFT for high-priority requests |
| `--num-cudagraph-capture-sizes` | int | 10 | warmup latency; more sizes = better decode graph coverage | −TPOT for uncommon decode batch sizes |
| `--speculative-model` + `--num-speculative-tokens` | str + int | None | +output throughput for high-acceptance-rate workloads | −TPOT by 1.5–2× when draft acceptance > 70 % |

## Supported Models

vLLM uses the Hugging Face `transformers` model architecture registry. Any model with a registered `AutoModelForCausalLM` class and a supported attention pattern is served natively. Notable supported families:

| Family | Examples | Notes |
|---|---|---|
| Llama / Llama 2 / Llama 3 | Llama-3-8B, Llama-3-70B, Llama-3.1-405B | Most-tested family; reference benchmarks use Llama-3 |
| Mistral / Mixtral | Mistral-7B, Mixtral-8×7B | MoE via expert-parallel decode |
| Gemma / Gemma 2 / Gemma 3 | Gemma-2-9B, Gemma-3-27B | Supports GQA |
| Qwen / Qwen 2 / Qwen 2.5 | Qwen2.5-72B | Long context (128K) tested |
| DeepSeek / DeepSeek-V2 / DeepSeek-R1 | DeepSeek-V2-Chat, DeepSeek-R1-671B | MLA attention requires FlashInfer MLA path |
| Phi / Phi-3 | Phi-3-mini-4k, Phi-3.5-MoE | Small MoE variant supported |
| Command R / Command R+ | Cohere Command R | RAG-optimized; long context |
| Falcon | Falcon-7B, Falcon-40B | Legacy; supported but not actively tuned |
| OPT / BLOOM | OPT-30B, BLOOM-176B | Reference/legacy |

## Known Strengths / Weaknesses

### Strengths

- **PagedAttention eliminates KV cache fragmentation**: GPU memory utilization reaches 90–95 % vs 40–60 % for static pre-allocation engines. This directly translates to higher max concurrency and throughput at the same hardware.
- **Prefix caching with zero-overhead miss path**: when a request's prefix is not cached, there is no overhead — the system falls through to normal block allocation. The gain is purely additive for agentic workloads with shared system prompts.
- **Broad hardware and model support**: runs on NVIDIA Hopper/Ampere/Ada, AMD MI300X, and (via `tpu-inference` backend) Google Cloud TPUs. Covers nearly every open-weight model family from 1B to 671B parameters.
- **Chunked prefill reduces TTFT P99 under load**: by interleaving prefill chunks with decode steps, no single long-prompt prefill can monopolize the GPU and stall all in-flight decode sequences.
- **Speculative decoding integration**: draft-model or n-gram-based speculation can double output token throughput for long-generation workloads (chain-of-thought, summarization).

### Weaknesses

- **Continuous batching does not disaggregate prefill from decode**: both operations compete for the same GPU. Under high request rates, prefill-heavy workloads increase TTFT for co-located decode sequences. Disaggregated prefill (`--disaggregated-prefill`, experimental) addresses this but requires separate prefill and decode worker pools.
- **KV cache swap overhead at burst traffic**: when the GPU KV pool is exhausted, preempted sequences must re-prefill from scratch (or restore from CPU swap at ~PCIe bandwidth). At sustained high concurrency this causes latency spikes. Mitigated by `--swap-space` and sufficient `--gpu-memory-utilization`, but ultimately a memory capacity constraint.
- **Pipeline parallelism adds bubble latency**: unlike TP, PP introduces idle time at pipeline boundaries. PP > 1 is rarely the right choice for serving; TP + DP is preferred.
- **CUDA graph capture increases startup time and VRAM**: capturing graphs for many decode batch sizes requires a warmup pass for each size and a small persistent VRAM allocation. `--enforce-eager` disables this at significant throughput cost.
- **MoE expert parallelism is less optimized than dedicated MoE engines**: for DeepSeek-V2/V3-scale MoE models, SGLang's expert-parallel dispatch and TensorRT-LLM's fused MoE kernels outperform vLLM on throughput.

## Connections

- [SGLang](sglang.md) — primary comparison engine; RadixAttention provides similar prefix-caching benefits with different scheduling semantics
- [TensorRT-LLM](tensorrt-llm.md) — NVIDIA-optimized alternative; higher peak throughput on A100/H100 for single-model-single-GPU deployments
- [Multi-Turn Agentic workload](../workloads/multi-turn-agentic.md) — primary target workload; prefix caching and chunked prefill are the key levers
- [Parallel Tool Use workload](../workloads/parallel-tool-use.md) — burst prefix-sharing pattern; prefix caching benefit is concentrated here
- [Long Context RAG workload](../workloads/long-context-rag.md) — stress-tests KV cache capacity and prefill throughput
- [Chain-of-Thought workload](../workloads/chain-of-thought.md) — stress-tests decode throughput; speculative decoding is the key lever
- [Structured Output workload](../workloads/structured-output.md) — JSON-constrained decoding via Outlines/Guidance integration
- [Prefix caching for multi-turn agentic hypothesis](../hypotheses/prefix-caching-multi-turn-agentic.md) — hypothesis targeting `--enable-prefix-caching` for agentic workloads
- [FP8 quantization hypothesis](../hypotheses/fp8-quantization-throughput.md) — hypothesis targeting `--quantization fp8` for higher concurrency
- [Chunked prefill hypothesis](../hypotheses/chunked-prefill-high-concurrency.md) — hypothesis targeting `--enable-chunked-prefill` for TTFT reduction
- [Speculative decoding hypothesis](../hypotheses/speculative-decoding-cot.md) — hypothesis targeting `--speculative-model` for chain-of-thought throughput
- [PagedAttention concept](../concepts/continuous-batching.md) — continuous batching and paged KV cache fundamentals
- [KV cache concept](../concepts/kv-cache.md) — KV cache design and memory pressure
- [Continuous batching concept](../concepts/continuous-batching.md) — iteration-level scheduling fundamentals

## Sources

- `benchmark_harness.py` lines 36–55 — canonical `flag_map` for vLLM flags accepted by this repo's harness
- vLLM paper: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023
- vLLM documentation: https://docs.vllm.ai
