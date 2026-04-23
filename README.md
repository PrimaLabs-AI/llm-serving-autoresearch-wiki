# tpu_performance_autoresearch_wiki

An LLM-maintained knowledge base for **autoresearch-style optimization of TPU model performance**.

The loop:

```
sources + codebases + profiles  →  concepts + models  →  ranked hypotheses
                                                               ↓
                             observations  ←  experiments (config + run + profile)
                                 ↓
                      (priors revised, new hypotheses formulated)
```

Humans curate sources, set optimization targets, and approve experiments. The LLM writes and maintains every file in `wiki/` — page summaries, cross-references, rankings, experiment logs.

## Scope

- **In:** step time, MFU, tokens/sec, memory, and everything that affects them on TPU — compiler flags, parallelism, rematerialization, attention kernels, layout, fusion, scheduling, precision.
- **Out:** model quality/convergence. An optimization that changes model semantics is invalid — noted, not counted.

## Layout

```
SCHEMA.md           single source of truth — read this to understand how the wiki works
CLAUDE.md           @SCHEMA.md pointer (Claude Code)
GEMINI.md           @SCHEMA.md pointer (Gemini CLI)
wiki/               LLM-owned markdown (index, log, page types per schema)
raw/                immutable inputs — sources, code, profiles, assets
  code/             ingested repos (git submodules)
```

## Working with the repo

Clone with submodules:

```bash
git clone --recurse-submodules <url>
```

Or after a plain clone:

```bash
git submodule update --init --recursive
```

Add a new codebase:

```bash
git submodule add <repo-url> raw/code/<slug>
```

Then ask the agent to ingest it — see `SCHEMA.md` → `INGEST-CODEBASE`.

## Ingested codebases

- [jax](raw/code/jax) — JAX library itself (jax-ml/jax): transformations, sharding, `jax.profiler`, `jax.experimental.roofline`, Pallas DSL, and first-party reference TPU kernels (`flash_attention`, `splash_attention`, `paged_attention`, `ragged_paged_attention`, `megablox`, `matmul`, `all_gather`, `threefry`). Ground-truth for every other codebase in this list.
- [xprof](raw/code/xprof) — XProf profiler + TensorBoard plugin (OpenXLA)
- [xprof-mcp](raw/code/xprof-mcp) — MCP server wrapping xprof for agent-driven profile analysis
- [torchax](raw/code/torchax) — PyTorch-on-JAX interop layer (Google)
- [tokamax](raw/code/tokamax) — custom TPU/GPU kernels on JAX + Pallas (OpenXLA)
- [stablehlo](raw/code/stablehlo) — StableHLO operation set + MLIR dialect (OpenXLA)
- [scaling-book](raw/code/scaling-book) — "How To Scale Your Model": TPU scaling / parallelism reference (JAX ML / Google DeepMind)
- [autoresearch](raw/code/autoresearch) — Karpathy's autoresearch reference implementation (the methodology this wiki adapts to TPU perf)
- [learning-machine](raw/code/learning-machine) — Qi Huang's JAX/ML experiments repo; the `jax-huggingface/` subfolder is ingested as the [jax-huggingface](wiki/codebases/jax-huggingface.md) codebase
- [pallas-forge](raw/code/pallas-forge) — auto-tuning framework for Pallas kernels on TPU (block-size sweeps, roofline + xprof capture); ships fused RMSNorm+residual, tiled matmul, fused SwiGLU reference kernels
- [axlearn](raw/code/axlearn) — Apple's public training framework; **only public TPU Pallas Mamba1/Mamba2/RAttention SSM kernels**, plus splash extensions (dropout + logit sink), GPU Triton megablox (`arXiv:2507.05411`)
- [tpu-inference](raw/code/tpu-inference) — vLLM's TPU inference backend; **broadest novel Pallas surface** (RPA v2/v3, MLA v1/v2, fused_moe v1, quantized_matmul blockwise, all_gather_matmul, GDN, SparseCore, structured-sparse); crown-jewel tuning tables
- [maxtext](raw/code/maxtext) — AI-Hypercomputer reference JAX trainer for Gemma/Llama/DeepSeek/Qwen/Mistral/Kimi; splash + ragged-paged-attention + megablox GMM + MLIR-dialect SparseCore
- [maxdiffusion](raw/code/maxdiffusion) — AI-Hypercomputer reference JAX diffusion trainer; **only repo where ring-attention is wired in as first-class splash-integrated kernel** (announced 2026-04-16)
- [ringattention](raw/code/ringattention) — haoliuhl's canonical Pallas TPU ring-attention (Liu et al. 2023 paper companion); unidirectional, no zig-zag
- [alphafold3](raw/code/alphafold3) — pinned to tag **v3.0.1** — only public production-grade **Pallas fused GLU** (GPU via Triton-on-Pallas); kernels removed from `main` after v3.0.1
- [recurrentgemma](raw/code/recurrentgemma) — Google DeepMind's canonical public Mosaic-TPU LRU Pallas scan (Griffin RG-LRU); ancestor of axlearn Mamba
- [ejkernel](raw/code/ejkernel) — single-author community Pallas library (erfanzar); broadest community TPU surface (17 kernels), Apache-2.0
- [EasyDeL](raw/code/EasyDeL) — training/serving framework wrapping ejkernel via an operations registry (same author)
- [sglang-jax](raw/code/sglang-jax) — SGLang's JAX port; mostly vendored from tpu-inference; **novel speculative-decoding tree kernels** (EAGLE) and the ecosystem's largest tuning table (~2,000+ RPA entries)
- [marin](raw/code/marin) — vendors levanter; **deployment-time autotune harness** (kernel-agnostic, shard-aware, compile-cost-aware, GCS-persistent) — the autotune pattern this wiki should emulate
- [graphcast](raw/code/graphcast) — DeepMind weather-forecasting model; wrapper over upstream splash with custom `WeatherMeshMask` (non-LLM block-sparse example)
- [simply](raw/code/simply) — DeepMind experimental serving framework; RPA wrapper that documents the DMA-overhead-bytes autotune heuristic (~0.5 MiB virtual bytes)
- [jaxite](raw/code/jaxite) — FHE (Fully Homomorphic Encryption) Pallas kernels; only non-ML Pallas TPU reference in this wiki (CGGI bootstrap via bf16 byte-split matmul reassembly to u32)
- [qwix](raw/code/qwix) — quantization framework; `QArray`-aware `pallas_call` wrapper; successor to AQT
- [aqt](raw/code/aqt) — deprecated quantization framework; superseded by qwix

## Authoritative contract

[SCHEMA.md](SCHEMA.md) defines page types, operations, frontmatter, naming, and behavioral rules. If anything here conflicts with `SCHEMA.md`, the schema wins.
