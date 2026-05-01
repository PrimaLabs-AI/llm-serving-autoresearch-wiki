---
title: "SGLang"
type: engine
tags: [engine, sglang, radix-attention]
commit: ""
supported_hardware: [h100, h200, b200, mi300x]
created: 2026-04-29
updated: 2026-05-01
---

SGLang is a high-throughput, low-latency LLM serving engine developed at UC Berkeley and LMSYS. Its defining innovation is **RadixAttention** — a radix-tree-based KV cache that delivers automatic, zero-configuration prefix caching across all requests sharing any common token prefix, making it particularly well-suited to multi-turn agentic and parallel-tool-use workloads.

## Overview

SGLang (Structured Generation Language) was originally designed around a structured-generation programming model, but its serving engine has become a first-class open-source inference runtime in its own right. The key architectural differentiator from [vLLM](vllm.md) is how KV cache reuse is managed: where vLLM requires `--enable-prefix-caching` to be explicitly toggled and uses a hash-based block lookup, SGLang's RadixAttention stores KV entries in a radix tree keyed on actual token sequences. This allows it to find the longest common prefix across all live and recently evicted requests automatically, with no per-request hinting required.

SGLang exposes an OpenAI-compatible HTTP API (`/v1/completions`, `/v1/chat/completions`) plus a native `/generate` endpoint. Its server is launched via `python -m sglang.launch_server`. FlashInfer is the default attention backend for CUDA, providing paged-KV attention for decode and chunked-prefill for multi-query prefill.

Additional serving-oriented features include:

- **Overlap scheduling** (`--enable-overlap-schedule`): the CPU scheduling and tokenization for the next batch are pipelined with the GPU forward pass of the current batch, hiding dispatch overhead.
- **Data-parallel attention** (`--enable-dp-attention`): for data-parallel deployments, the attention computation itself is partitioned across the data-parallel axis (in addition to tensor parallelism), reducing per-device KV cache pressure and enabling higher total concurrency.
- **Structured output**: native JSON-schema and regex constrained decoding via XGrammar, with grammar compilation amortized across requests.
- **Speculative decoding**: EAGLE and Medusa draft-model support via `--speculative-algo`.

## Architecture

### Scheduler

SGLang's scheduler runs once per forward pass and operates on a combined ready queue drawn from:

1. **Waiting requests** — not yet prefilled; the scheduler greedily selects up to `--max-running-requests` requests to prefill, subject to KV-token budget.
2. **Running requests** — in-flight decode sequences; one decode token generated per step per sequence.
3. **Retracted requests** — sequences whose KV blocks were evicted; they are re-queued for re-prefill (similar to vLLM preemption).

The scheduler's primary loop uses **chunk prefill** when `--chunk-prefill-size` is set: a long prompt is consumed in `chunk_prefill_size`-token slices per step, interleaved with decode steps for running requests. This bounds TTFT variance without blocking in-flight decodes.

Scheduling policy defaults to FCFS. Unlike vLLM, SGLang does not expose a priority-queue policy as a command-line flag; request priority is managed through the RadixCache eviction policy (LRU over the radix tree, giving recently reused prefixes priority retention).

### KV Cache Manager (RadixAttention)

The KV cache is organized as a **radix tree** mapping token-sequence prefixes to physical KV pages. When a new request arrives, the scheduler walks the radix tree to find the longest prefix match — retrieving those pages for free. Requests sharing the same system prompt, few-shot examples, or conversation history all hit the same radix-tree nodes.

Key properties:

- **Automatic**: no per-request flag or system-prompt ID required; the tree is keyed on raw token IDs.
- **Eviction**: LRU leaf eviction — the radix tree's leaf nodes are evicted first when memory pressure requires it. Shared prefix nodes are never evicted while any request is using them.
- **Page granularity**: pages are the same kind of fixed-size blocks used by PagedAttention; RadixAttention reuses the FlashInfer paged-KV API for the actual attention kernel.
- **No swap space**: SGLang does not implement CPU-side KV swap by default. Evicted KV pages are discarded; preempted requests re-prefill from scratch.

### Batching Strategy

SGLang supports two modes, selectable at launch time:

- **Continuous batching** (default): new requests are admitted when in-flight requests decode their next token. Prefill and decode share the same forward pass.
- **Chunked prefill** (`--chunk-prefill-size N`): long prompt prefills are split into N-token chunks interleaved with decode. Identical semantics to vLLM's `--enable-chunked-prefill` but the chunk size is named differently.

### Attention Backends

| Backend | Platform | Notes |
|---|---|---|
| FlashInfer | NVIDIA (CUDA) | Default on H100/H200/B200; paged-KV decode + grouped-query prefill |
| Triton | NVIDIA (CUDA) | Fallback when FlashInfer not available |
| Custom CUDA kernels | NVIDIA (CUDA) | Hand-written kernels for specific shapes/dtypes |
| ROCm Flash Attention | AMD (MI300X) | ROCm backend; same flag surface |

## Key Abstractions

| Class / Module | Location (approximate) | Role |
|---|---|---|
| `Engine` | `python/sglang/srt/server.py` | Top-level serving engine object; owns the runtime loop |
| `Runtime` | `python/sglang/runtime.py` | Programmatic Python API wrapping Engine |
| `Scheduler` | `python/sglang/srt/managers/scheduler.py` | Batch assembly, RadixCache admission, chunked prefill |
| `TokenizerManager` | `python/sglang/srt/managers/tokenizer_manager.py` | Async tokenization; feeds the Scheduler |
| `TpModelWorker` | `python/sglang/srt/model_worker.py` | Per-GPU process; holds weights, KV cache, and runs forward pass |
| `RadixCache` | `python/sglang/srt/mem_cache/radix_cache.py` | Radix-tree KV index; node-level LRU eviction |
| `ReqToTokenPool` | `python/sglang/srt/mem_cache/memory_pool.py` | GPU memory pool for KV pages (token-level allocator) |
| `ModelRunner` | `python/sglang/srt/model_executor/model_runner.py` | Executes model forward pass; manages CUDA graphs |
| `ChunkPrefillScheduler` | `python/sglang/srt/managers/scheduler.py` | Chunked-prefill admission logic embedded in Scheduler |
| `DataParallelController` | `python/sglang/srt/managers/dp_manager.py` | Routes requests across DP replicas for DP-attention mode |

## Entry Points

### OpenAI-Compatible API Server

```bash
python -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --tp 1 \
  --mem-fraction-static 0.90 \
  --chunk-prefill-size 4096 \
  --enable-overlap-schedule
```

Additional key flags beyond those in the serving-surfaces table:

- `--host` / `--port` — bind address (default `0.0.0.0:30000`; note: different default port from vLLM's 8000)
- `--served-model-name` — alias exposed in `/v1/models`
- `--max-total-tokens` — total KV token budget across all in-flight requests (replaces vLLM's `--max-num-batched-tokens` as the primary memory knob)
- `--max-running-requests` — maximum number of concurrently in-flight requests (decode + prefill)
- `--schedule-policy` — FCFS or custom; defaults to FCFS
- `--speculative-algo` — `EAGLE` or `MEDUSA`; requires `--speculative-draft-model-path`
- `--tokenizer-mode` — `auto` (default) or `slow`
- `--disable-radix-cache` — fall back to LRU-only block allocation without the radix tree (rarely useful)

### Programmatic (Python) API

```python
import sglang as sgl

llm = sgl.Engine(
    model_path="meta-llama/Meta-Llama-3-8B-Instruct",
    tp_size=1,
    mem_fraction_static=0.90,
)
outputs = llm.generate(["Hello, my name is"], sampling_params={"max_new_tokens": 50})
llm.shutdown()
```

### Benchmark Harness Integration

The `benchmark_harness.py` in this repo launches SGLang via:

```
python -m sglang.launch_server
```

with flags driven by `ENGINE_COMMANDS["sglang"]["flag_map"]` (see `benchmark_harness.py` lines 55–68). Health check polls `http://localhost:30000/health`.

## Serving-Relevant Surfaces

All flags below are accepted by `benchmark_harness.py`'s `flag_map` for SGLang (source: `benchmark_harness.py` lines 57–68). Config keys use `snake_case`; CLI spellings use `--` prefix.

| Config Key | CLI Flag | Type | Default | Category | Effect |
|---|---|---|---|---|---|
| `tp` | `--tp` | int | 1 | Parallelism | Tensor-parallel degree. Splits each attention head and MLP column across `tp` GPUs within a node. Must divide `num_attention_heads` evenly. Use 2–8 for large models. Each TP rank holds `1/tp` of the KV cache, so TP also divides per-device KV budget. |
| `dp` | `--dp` | int | 1 | Parallelism | Data-parallel degree. Launches `dp` independent engine replicas sharing weights (if `--enable-dp-attention`) or fully independent (default). A built-in DP controller routes requests. Linear throughput scaling with no head-of-line coupling between replicas. |
| `pp` | `--pp` | int | 1 | Parallelism | Pipeline-parallel degree. Splits model layers across `pp` stages. Introduces a pipeline bubble on every micro-batch; rarely beneficial in serving. Use only when a model exceeds per-node memory capacity beyond what TP alone can handle. |
| `mem_fraction_static` | `--mem-fraction-static` | float | 0.90 | KV cache | Fraction of GPU HBM allocated to the static KV token pool (after model weights). Equivalent role to vLLM's `--gpu-memory-utilization`. Values 0.85–0.92 are typical on H100 SXM; higher values risk OOM from activation spikes during large prefill batches. |
| `chunk_prefill_size` | `--chunk-prefill-size` | int | None (disabled) | Batching | Maximum tokens processed per prefill chunk per scheduling step. When set, long prompts are sliced into chunks of this size and interleaved with decode steps. Bounds TTFT variance at the cost of slightly lower peak prefill throughput. Recommended for latency-sensitive multi-turn workloads. Analogous to vLLM's `--max-num-batched-tokens` with `--enable-chunked-prefill`. |
| `enable_overlap_schedule` | `--enable-overlap-schedule` | bool | false | Scheduling | Pipelines the CPU scheduling pass (batch assembly, tokenization, RadixCache lookup) with the GPU forward pass of the current batch. Hides ~1–5 ms of CPU-side dispatch overhead per step. Effective at high request rates where scheduling itself becomes a bottleneck. |
| `enable_dp_attention` | `--enable-dp-attention` | bool | false | Parallelism | Enables data-parallel attention: each DP rank computes attention only over its own request subset, reducing per-rank KV pressure. Requires `--dp > 1`. Increases total effective KV capacity proportionally to `dp`. Particularly beneficial when the working set of active prefixes exceeds single-GPU KV capacity. |
| `disable_cuda_graph` | `--disable-cuda-graph` | bool | false | CUDA graphs | Disables CUDA graph capture and forces eager execution. Removes the graph-replay speedup on decode (typically 20–40 % latency savings from graph). Should only be set for debugging or when a model has dynamic control flow incompatible with graph capture. Equivalent to vLLM's `--enforce-eager`. |
| `quantization` | `--quantization` | str | None | Quantization | Weight/activation quantization method. Supported: `fp8` (H100/H200/B200 native via FlashInfer FP8 path), `awq`, `gptq`, `marlin`. FP8 is recommended for H100/H200/B200: halves weight-storage footprint, enabling ~2× more KV pages at the same `mem_fraction_static`. |
| `dtype` | `--dtype` | str | `auto` | Quantization | Compute dtype for activations and non-quantized weights. `auto` resolves to `bfloat16` on Ampere/Hopper. Override with `float16` on older GPUs or for compatibility with certain quantization kernels. `float32` is debug-only. |

## Tunable Knobs

The following extends the serving-surfaces table with additional knobs available in `sglang.launch_server` for manual tuning:

| Flag | Type | Default | Expected Effect on Throughput | Expected Effect on TTFT / TPOT |
|---|---|---|---|---|
| `--tp` | int | 1 | +throughput until TP comms dominate (typically > 4 on single-node) | −TTFT (parallel prefill) |
| `--dp` | int | 1 | linear throughput scaling (independent request streams) | neutral per-replica |
| `--pp` | int | 1 | −throughput (pipeline bubble); use only when TP capacity exceeded | bubble adds to TTFT |
| `--mem-fraction-static` | float | 0.90 | +throughput with higher values (more KV pages) | neutral |
| `--chunk-prefill-size` | int | None | neutral to slight −prefill throughput | −TTFT p99 by 30–50 % at high concurrency |
| `--enable-overlap-schedule` | bool | false | +5–10 % throughput at high QPS by hiding CPU dispatch | neutral on TTFT |
| `--enable-dp-attention` | bool | false | +throughput for dp > 1 deployments; reduces per-rank KV pressure | neutral |
| `--disable-cuda-graph` | bool | false | −20–40 % decode throughput | +TPOT (no graph replay) |
| `--quantization fp8` | str | None | +50–80 % more KV pages; +throughput at high concurrency | neutral |
| `--dtype bfloat16` | str | auto | reference baseline | reference |
| `--max-running-requests` | int | auto | +throughput as it rises; OOM risk beyond GPU KV capacity | +TTFT at very high values |
| `--max-total-tokens` | int | auto | primary KV token budget; raise if many long in-flight sequences | −TTFT when raised |
| `--speculative-algo EAGLE` | str | None | +output throughput when draft acceptance > 70 % | −TPOT by 1.5–2× |
| `--disable-radix-cache` | bool | false | −throughput on prefix-heavy workloads; removes tree lookup overhead | +TTFT for new requests (no cache) |
| `--schedule-policy` | str | fcfs | neutral on aggregate throughput | lcfs reduces TTFT for new short requests |

## Supported Models

SGLang uses the Hugging Face `transformers` model architecture registry and supports most architectures vLLM supports, with additional attention to models commonly used in agentic deployments:

| Family | Examples | Notes |
|---|---|---|
| Llama / Llama 2 / Llama 3 | Llama-3-8B, Llama-3-70B, Llama-3.1-405B | Most-tested; all agentic benchmarks default to Llama-3 |
| Mistral / Mixtral | Mistral-7B, Mixtral-8×7B | MoE via expert-parallel decode with FlashInfer |
| Gemma / Gemma 2 / Gemma 3 | Gemma-2-9B, Gemma-3-27B | GQA supported |
| Qwen / Qwen 2 / Qwen 2.5 / Qwen 3 | Qwen2.5-72B, Qwen3-32B | Long context (128K+) tested |
| DeepSeek / DeepSeek-V2 / DeepSeek-V3 / DeepSeek-R1 | DeepSeek-V3-671B, DeepSeek-R1-671B | MLA (multi-head latent attention) supported via FlashInfer MLA path; first-class support |
| Phi / Phi-3 / Phi-4 | Phi-3-mini-4k, Phi-4 | Small models; radix caching still beneficial for shared system prompts |
| Command R / Command R+ | Cohere Command R | RAG workloads; tested with long-context prefix caching |
| InternLM / InternVL | InternLM2-7B | Multi-modal variants supported via SGLang's VLM path |
| LLaVA / Qwen-VL | LLaVA-1.6, QwenVL | Vision-language models supported natively |

## Known Strengths / Weaknesses

### Strengths

- **RadixAttention provides automatic, always-on prefix caching**: unlike vLLM's opt-in `--enable-prefix-caching` (which requires all sharing requests to hit an exact hash of a full KV block boundary), SGLang's radix tree finds the longest common prefix down to individual tokens, regardless of alignment. This yields higher cache hit rates on workloads where prefixes vary slightly across requests (e.g., chat templates with per-user metadata).
- **Higher throughput than vLLM on multi-turn agentic and parallel-tool-use workloads at matched hardware**: the combination of RadixAttention + overlap scheduling eliminates two sources of overhead — redundant KV computation and CPU scheduling bubbles — that vLLM incurs even with prefix caching enabled. Empirical comparisons (SGLang paper, LMSYS benchmarks) show 20–40 % higher throughput on workloads with > 50 % prefix reuse.
- **Overlap scheduling reduces CPU bottleneck at high QPS**: at request rates > 100 req/s with short decode sequences, the per-step CPU scheduling cost becomes visible. Overlap scheduling pipelines this behind the GPU forward pass, sustaining higher throughput without scaling CPU resources.
- **Native structured output (XGrammar) with low overhead**: JSON-schema and regex constrained decoding is compiled once per schema and applied incrementally, avoiding the per-token grammar-automaton overhead of simpler implementations. This is particularly relevant for tool-call workloads.
- **DP attention enables near-linear horizontal scaling**: the `--dp N --enable-dp-attention` combination scales both compute and KV capacity linearly with `N` within a single `launch_server` process, without requiring a separate load balancer (the built-in DP controller handles routing).

### Weaknesses

- **No CPU-side KV swap space**: SGLang does not implement block swapping to CPU DRAM (vLLM's `--swap-space`). When GPU KV pages are exhausted, preempted requests must fully re-prefill. This makes SGLang more sensitive to memory capacity under bursty traffic spikes and workloads with many long sequences.
- **RadixAttention tree overhead at very high token counts**: the radix tree lookup and LRU eviction are O(token sequence length) per request admission. For very long sequences (100K+ tokens), the tree walk adds measurable CPU-side latency. This is rarely a bottleneck in practice but is a theoretical weakness at extreme context lengths.
- **Less mature ecosystem than vLLM**: vLLM has broader hardware support, more extensive documentation, and more production deployments. SGLang's AMD MI300X support and multi-node TP are less battle-tested. Some quantization methods available in vLLM (e.g., SqueezeLLM, AutoAWQ with fused kernels) have partial support in SGLang.
- **Pipeline parallelism is rarely useful**: like vLLM, SGLang's PP support exists but introduces pipeline-bubble latency. For models exceeding single-node TP capacity, the recommended path is multi-node TP rather than PP.
- **Overlap scheduling adds scheduling complexity**: enabling `--enable-overlap-schedule` requires that the scheduler's CPU work finishes in one GPU-step budget. If the scheduler is unexpectedly slow (e.g., very long radix tree on first request of a new session), the pipeline can stall, causing a latency spike on that step.
- **DP attention requires DP > 1 and adds coordination overhead**: the `--enable-dp-attention` path is only useful when `--dp > 1`. The DP controller adds a routing hop before each request reaches a worker, introducing ~0.1–0.5 ms of additional latency per request compared to single-replica deployments.

## Connections

- [vLLM](vllm.md) — primary comparison engine; PagedAttention with opt-in prefix caching; vLLM has broader hardware/ecosystem support while SGLang leads on agentic-workload throughput
- [TensorRT-LLM](tensorrt-llm.md) — NVIDIA-optimized alternative; higher peak throughput for single-model offline workloads but less flexible KV caching
- [Multi-Turn Agentic workload](../workloads/multi-turn-agentic.md) — primary target workload; RadixAttention provides the largest advantage here due to high prefix reuse across turns
- [Parallel Tool Use workload](../workloads/parallel-tool-use.md) — burst prefix-sharing pattern; RadixAttention + overlap scheduling both contribute
- [Long Context RAG workload](../workloads/long-context-rag.md) — stresses KV capacity; `--mem-fraction-static` and FP8 are the key levers
- [Chain-of-Thought workload](../workloads/chain-of-thought.md) — decode-throughput-bound; speculative decoding (EAGLE) is the primary lever
- [Structured Output workload](../workloads/structured-output.md) — native XGrammar support avoids per-token grammar overhead
- [SGLang RadixAttention vs vLLM hypothesis](../hypotheses/sglang-radx-vs-vllm-agentic.md) — head-to-head comparison hypothesis for multi-turn agentic workloads

## Sources

- `benchmark_harness.py` lines 55–68 — canonical `flag_map` for SGLang flags accepted by this repo's harness
- SGLang paper: Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs", NeurIPS 2024
- SGLang documentation: https://docs.sglang.ai
- SGLang GitHub: https://github.com/sgl-project/sglang
- LMSYS throughput benchmarks: https://lmsys.org/blog/2024-01-17-sglang/ (RadixAttention announcement)
