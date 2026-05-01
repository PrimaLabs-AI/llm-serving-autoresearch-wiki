---
title: "TensorRT-LLM"
type: engine
tags: [engine, tensorrt-llm, nvidia]
commit: ""
supported_hardware: [h100, h200, b200]
created: 2026-04-29
updated: 2026-05-01
---

TensorRT-LLM is NVIDIA's open-source library for compiling and serving LLMs at maximum throughput on NVIDIA GPUs. It transforms a model into an optimized TensorRT engine ahead of time — fusing kernels, auto-tuning tile sizes per GPU architecture, and pre-allocating memory — then serves requests through an inflight-batching runtime. This compilation step produces the highest single-GPU peak throughput among open-source serving engines for NVIDIA hardware, at the cost of a one-time build phase and tighter coupling to a specific GPU architecture.

## Overview

TensorRT-LLM targets NVIDIA Hopper (H100, H200) and Blackwell (B200) GPUs. It is **NVIDIA-only**: there is no AMD/ROCm or CPU backend. The workflow splits into two phases:

1. **Build phase** — `trtllm-build` (or `tensorrt_llm.commands.build`) consumes HuggingFace or NeMo weights and produces a serialized `.engine` file optimized for a specific GPU SKU, batch size budget, and sequence length budget. The build runs TensorRT's layer-fusion and auto-tuner, selects architecture-specific kernels (e.g., Hopper WGMMA, FP8 tensor core paths on H100/H200, BF16/FP8 on B200), and captures CUDA graphs for the decode phase.

2. **Serve phase** — `tensorrt_llm.commands.run_server` (or Triton Inference Server + the `tensorrtllm_backend`) loads the compiled engine and runs an inflight-batching scheduler. Requests arrive as HTTP/gRPC and are dispatched as `LlmRequest` objects; the scheduler assembles iteration-level batches from a mix of context (prefill) and generation (decode) phases.

Key serving characteristics:
- Paged KV cache with configurable `tokens_per_block`, eliminating worst-case pre-allocation.
- CUDA graph capture for the decode step reduces kernel-launch overhead at the cost of fixed batch/sequence shapes.
- FP8 and INT8 quantization are first-class build options with NVIDIA-calibrated scales.
- Multi-GPU support via tensor parallelism (Megatron-LM style) and pipeline parallelism across nodes.
- Native Triton Inference Server integration for production deployments.

Compared to [vLLM](vllm.md) and [SGLang](sglang.md), TensorRT-LLM typically achieves higher single-GPU throughput on NVIDIA hardware due to TensorRT kernel fusion and architecture-specific auto-tuning. The trade-off is that every meaningful configuration change (dtype, max batch size, max sequence length, quantization) requires a rebuild of the engine.

## Architecture

### Scheduler

TensorRT-LLM's inflight batching scheduler runs once per iteration:

1. **Context queue**: newly arrived requests whose prompt tokens have not yet been processed. Context (prefill) requests are interleaved with generation requests up to `max_batch_size` and `max_num_tokens` token budgets.
2. **Generation queue**: in-flight requests currently emitting tokens. Each decode step appends one token per live sequence.
3. **Priority and eviction**: when KV cache blocks are exhausted, the scheduler pauses lower-priority context requests rather than evicting generation requests (unlike vLLM's preemption model). This favours decode latency at the cost of prefill latency under extreme load.

The scheduler is implemented in C++ (`tensorrt_llm/batch_manager/`) and is driven by the Python executor API or the Triton backend.

### KV Cache Manager

- **Block-based paged KV cache**: analogous to PagedAttention. KV blocks are fixed-size (`tokens_per_block`), allocated from a pre-reserved pool at engine launch.
- **Block reuse (prefix caching)**: optional `--kv_cache_reuse` flag. When enabled, the manager hashes token sequences and re-uses matching KV blocks across requests, yielding free TTFT reduction on shared prefixes.
- No built-in radix-tree prefix optimization (unlike SGLang); reuse is hash-based with a fixed LRU eviction policy.

### Batching

- **Inflight (continuous) batching**: context and generation tokens from multiple requests are packed into a single TensorRT forward pass each iteration.
- **Context chunking** (`--enable_context_fmha_fp32_acc`, build-time): long context requests can be chunked to bound per-iteration token count, analogous to vLLM's chunked prefill.
- **Max tokens per iteration**: controlled at build time via `max_num_tokens`; the scheduler will not exceed this budget per step.

### Attention

TensorRT-LLM ships its own fused attention kernels, selected at build time:
- `gpt_attention_plugin`: custom CUDA kernel for multi-head, multi-query, and grouped-query attention with paged KV.
- `context_fmha`: FlashAttention-style fused context attention for prefill.
- Hopper-specific: FP8 tensor core attention (WGMMA) used when `dtype=fp8` and target is `H100/H200`.
- Blackwell-specific: B200 attention kernel path (WGMMA SM100) when targeting B200.

### Parallelism

- **Tensor parallelism**: weight matrices split across GPUs on the tensor dimension (Megatron-LM style). Configured at build time via `--tp_size`; same value exposed at runtime as `--tensor-parallel`.
- **Pipeline parallelism**: model layers split across GPU ranks. Configured at build time via `--pp_size`; exposed at runtime as `--pipeline-parallel`.
- No native data-parallel serving mode; horizontal scaling is done at the load-balancer level (multiple engine replicas).

## Key Abstractions

| Class / Module | Location | Role |
|---|---|---|
| `ModelConfig` | `tensorrt_llm/models/` | Defines model architecture (layer count, head dims, vocab size, etc.) |
| `BuildConfig` | `tensorrt_llm/builder.py` | Build-time knobs: dtype, max_batch_size, max_seq_len, plugin selection |
| `EngineConfig` | `tensorrt_llm/runtime/engine.py` | Serialized engine metadata bundled with the `.engine` file |
| `GenerationSession` / `GptSession` | `tensorrt_llm/runtime/generation.py` | C++-backed session managing a single engine forward pass; wraps CUDA graph replay |
| `KvCacheManager` | `tensorrt_llm/batch_manager/kvCacheManager.py` | Block allocator; owns the paged KV pool and optional prefix-reuse hash table |
| `LlmRequest` | `tensorrt_llm/batch_manager/llmRequest.py` | In-flight request object tracking token buffers, state machine, and priority |
| `InferenceEngine` / Executor API | `tensorrt_llm/executor.py` | High-level Python API for async request submission and result streaming |
| `TensorRTLLMBackend` | Triton backend plugin | Triton Inference Server integration; maps Triton requests to the executor API |

## Entry Points

### Build an Engine

```bash
trtllm-build \
  --checkpoint_dir ./checkpoints/llama3-8b-hf \
  --output_dir ./engines/llama3-8b-fp16-tp1 \
  --dtype float16 \
  --max_batch_size 64 \
  --max_seq_len 8192 \
  --gpt_attention_plugin float16 \
  --gemm_plugin float16 \
  --use_paged_context_fmha enable \
  --tp_size 1 \
  --pp_size 1
```

For FP8 on H100/H200:
```bash
trtllm-build \
  --checkpoint_dir ./checkpoints/llama3-8b-fp8 \
  --output_dir ./engines/llama3-8b-fp8-tp1 \
  --dtype float16 \
  --quantization fp8 \
  --max_batch_size 128 \
  --max_seq_len 8192 \
  --gpt_attention_plugin float16 \
  --gemm_plugin float16 \
  --tp_size 1
```

### Launch the Serving Runtime

```bash
python -m tensorrt_llm.commands.run_server \
  --model ./engines/llama3-8b-fp16-tp1 \
  --tensor-parallel 1 \
  --pipeline-parallel 1 \
  --dtype float16 \
  --max-batch-size 64 \
  --max-seq-len 8192 \
  --tokens-per-block 64
```

### Triton Inference Server Deployment

For production deployments, TensorRT-LLM integrates with Triton via the `tensorrtllm_backend`. The model repository directory contains `config.pbtxt` files that map Triton model names to engine paths and executor settings.

### Benchmark Harness Integration

`benchmark_harness.py` in this repo launches TRT-LLM via:
```
python -m tensorrt_llm.commands.run_server
```
with flags driven by `ENGINE_COMMANDS["tensorrt-llm"]["flag_map"]` (see `benchmark_harness.py` lines 71–84). The harness maps six config keys:

| Harness key | CLI flag |
|---|---|
| `tensor_parallel` | `--tensor-parallel` |
| `pipeline_parallel` | `--pipeline-parallel` |
| `dtype` | `--dtype` |
| `max_batch_size` | `--max-batch-size` |
| `max_seq_len` | `--max-seq-len` |
| `tokens_per_block` | `--tokens-per-block` |

## Serving-Relevant Surfaces

All flags listed below are tunable for serving experiments. Flags marked **build-time** require a full engine rebuild; flags marked **runtime** can be changed without rebuilding.

| Flag | Phase | Type | Default | Effect |
|---|---|---|---|---|
| `--tensor-parallel` / `--tp_size` | build + runtime | int | 1 | Number of GPUs for tensor parallelism; weight matrices split across TP ranks. Must match between build and serve. |
| `--pipeline-parallel` / `--pp_size` | build + runtime | int | 1 | Number of pipeline stages; model layers distributed across PP ranks. |
| `--dtype` | build + runtime | str | `float16` | Activation and weight dtype: `float16`, `bfloat16`, `float32`, `fp8`. FP8 requires H100/H200/B200. |
| `--max-batch-size` / `max_batch_size` | build + runtime | int | 64 | Maximum number of requests in flight simultaneously. Larger values increase memory and may reduce per-request latency isolation. |
| `--max-seq-len` / `max_seq_len` | build + runtime | int | 2048 | Maximum sequence length (prompt + output). KV cache pool is sized to this limit. |
| `--tokens-per-block` | build + runtime | int | 64 | KV cache block granularity in tokens. Smaller blocks reduce fragmentation; larger blocks reduce allocation overhead. Common values: 32, 64, 128. |
| `--use_paged_kv_cache` | build | bool | true | Enable paged KV cache. Disabling falls back to static allocation; almost always worse unless sequence lengths are perfectly uniform. |
| `--use_inflight_batching` | build | bool | true | Enable continuous/inflight batching. Disabling forces synchronous batch semantics; dramatically reduces throughput. |
| `--gemm_plugin` | build | str | `float16` | dtype for the GEMM TRT plugin (linear layers). Setting to `fp8` with FP8 checkpoints enables tensor-core FP8 paths. |
| `--gpt_attention_plugin` | build | str | `float16` | dtype for the fused GPT attention plugin. Controls which attention kernel path is selected. |
| `--context_fmha` / `--use_paged_context_fmha` | build | bool | false / true | Enable FlashAttention-style fused context attention for prefill. `use_paged_context_fmha enable` is recommended for paged deployments. |
| `--kv_cache_reuse` | runtime | bool | false | Enable prefix KV cache reuse (hash-based). Reduces TTFT when requests share a common prompt prefix. |
| `--max_num_tokens` | build | int | auto | Maximum total tokens (context + generation) across all in-flight requests per iteration. Tighter bound than `max_batch_size * max_seq_len`. |
| `--quantization` | build | str | — | Quantization scheme: `fp8`, `int8_sq` (SmoothQuant), `int4_awq`, `int8_weight_only`. Each requires calibrated checkpoints. |

## Tunable Knobs

Beyond the serving-relevant surfaces table, the following knobs are worth exploring in experiments:

| Knob | Build vs Runtime | Notes |
|---|---|---|
| `tokens_per_block` | build | Sweep 32 / 64 / 128; 64 is the typical sweet spot for long-context; smaller reduces fragmentation for short requests |
| `max_batch_size` | build | Must be set conservatively; setting too high wastes KV pool memory even when fewer requests are active |
| FP8 vs FP16 | build | FP8 roughly halves weight memory footprint; ~10–25 % throughput uplift on H100/H200 for compute-bound decode |
| `gpt_attention_plugin` fp8 | build | Enables FP8 tensor core attention; requires FP8 checkpoint and H100/H200/B200 |
| `use_paged_context_fmha` | build | Recommended for all paged-KV deployments; avoids re-implementing attention for the context path |
| `kv_cache_reuse` | runtime | Low-cost flag; worth enabling for any workload with repeated system prompts or multi-turn contexts |
| Tensor parallel degree | build | H100 SXM5 (80 GB): TP=1 often sufficient for 7–13B models; TP=2 or TP=4 for 70B+ |
| `max_num_tokens` | build | Controls per-iteration token budget; tuning prevents OOM under burst while allowing higher steady-state throughput |
| CUDA graph capture | build (automatic) | Captured for decode by default; set `--use_custom_all_reduce disable` to debug graph issues |

## Supported Models

TensorRT-LLM ships pre-built recipes for major model families. Each requires model-specific Python scripts under `examples/<family>/` to convert weights and run builds.

| Model Family | Quantization Support | Notes |
|---|---|---|
| LLaMA / LLaMA-2 / LLaMA-3 | FP16, BF16, FP8, INT8 SQ, INT4 AWQ | Best-supported family; FP8 recipes for H100 validated |
| Mistral / Mixtral (MoE) | FP16, BF16, FP8, INT4 AWQ | MoE routing fused into plugin |
| Falcon | FP16, BF16, INT8 | Older architecture; GQA variant supported |
| GPT-2 / GPT-J / GPT-NeoX | FP16, INT8 | Legacy support; less optimized than Llama path |
| BLOOM | FP16, INT8 | ALiBi positional encoding handled |
| Baichuan / ChatGLM | FP16, INT8 | Community recipes; less battle-tested |
| Phi-1/2/3 | FP16, BF16 | Small models; useful for fast iteration |
| Gemma / Gemma-2 | FP16, BF16, FP8 | Google models; Gemma-2 sliding-window attention supported |
| Qwen / Qwen-2 | FP16, BF16, FP8 | Alibaba family; large-context variants supported |
| Encoder-decoder (T5, BART) | FP16, BF16 | Separate encoder/decoder engines; cross-attention fused |

All model families are NVIDIA-only — no AMD/ROCm path exists in TensorRT-LLM.

## Known Strengths / Weaknesses

### Strengths

- **Highest single-GPU throughput on NVIDIA**: TensorRT's layer-fusion, kernel auto-tuner, and architecture-specific code paths (Hopper WGMMA, Blackwell SM100) consistently outperform Python-first engines on raw tokens/sec for NVIDIA hardware.
- **Architecture-specific kernel auto-tuning**: the build phase profiles and selects the best tactic for each layer on the exact target GPU, not a generic CUDA kernel.
- **FP8 and INT8 first-class support**: NVIDIA has invested heavily in calibration tools (`modelopt`, `ammo`) and validated FP8 recipes for H100/H200, making quantization lower risk than on other engines.
- **Production-grade Triton integration**: the `tensorrtllm_backend` is used in NVIDIA's own DGX Cloud deployments and is well-tested at scale.
- **Paged KV cache + inflight batching**: matches vLLM/SGLang on core serving features, so the throughput advantage is not purchased by sacrificing concurrency.
- **CUDA graph decode**: capture is automatic; decode latency is consistently lower than non-graph paths.

### Weaknesses

- **NVIDIA-only**: no AMD/ROCm support. Any hypothesis involving TRT-LLM is invalid for MI300X workloads. The scheduler must not dispatch TRT-LLM experiments to MI300X instances.
- **Long build time**: a single `trtllm-build` for a 70B model can take 30–90 minutes on H100. Rapid iteration (e.g., sweeping `max_batch_size`) requires rebuild.
- **Version sensitivity**: TensorRT-LLM engine files are not portable across TRT-LLM releases or TensorRT versions. Upgrading the library requires rebuilding all engines.
- **Static shape constraints**: `max_batch_size` and `max_seq_len` are baked in at build time. An under-provisioned build cannot be hot-patched; an over-provisioned build wastes KV pool memory.
- **Smaller community**: fewer third-party integrations, fewer recipes for non-mainstream models, and less community troubleshooting bandwidth than vLLM.
- **Limited native prefix-caching sophistication**: `--kv_cache_reuse` is hash-based with LRU eviction; no radix-tree structure like SGLang's RadixAttention. Benefits are smaller for workloads with partial prefix overlaps.
- **Deployment complexity**: the two-phase (build + serve) model adds operational surface area compared to vLLM or SGLang which serve directly from HuggingFace checkpoints.

## Connections

- [vLLM](../engines/vllm.md) — primary open-source alternative; supports AMD/ROCm; Python-first; no build phase
- [SGLang](../engines/sglang.md) — RadixAttention for automated prefix caching; also NVIDIA + AMD
- [Multi-Turn Agentic](../workloads/multi-turn-agentic.md) — workload that benefits most from `--kv_cache_reuse` on TRT-LLM
- [Parallel Tool Use](../workloads/parallel-tool-use.md) — burst concurrency workload; tests `max_batch_size` ceiling
- [Chain-of-Thought](../workloads/chain-of-thought.md) — long decode; tests CUDA graph decode path
- [Long Context RAG](../workloads/long-context-rag.md) — stresses `max_seq_len` and `tokens_per_block` choices
- [Structured Output](../workloads/structured-output.md) — constrained decoding; TRT-LLM has limited native support compared to SGLang
- [Hypotheses](../hypotheses/) — open optimization candidates for TRT-LLM

## Sources

- Knowledge-based ingestion (no local source available under `raw/code/tensorrt-llm`); populate with `commit: ""` until local checkout is available.
- NVIDIA TensorRT-LLM documentation: https://nvidia.github.io/TensorRT-LLM/
- NVIDIA TensorRT-LLM GitHub: https://github.com/NVIDIA/TensorRT-LLM
- `benchmark_harness.py` lines 71–84 — harness flag map for `tensorrt-llm`
