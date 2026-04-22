---
title: "Gemma 4 E4B — TPU Autoresearch Optimization"
type: experiment-program
tags: [program, model-under-optimization, gemma, torchax, jax, active]
created: 2026-04-22
updated: 2026-04-22
status: active
hardware: "TPU v6e (primary)"
framework: "torchax (PyTorch-on-JAX); optional native-JAX port"
model_source: "https://huggingface.co/google/gemma-4-E4B"
---

# Gemma 4 E4B — TPU Autoresearch Optimization

A long-running research program that imports Google's **Gemma 4 E4B** via Hugging Face, gets it running on TPU through **torchax** (PyTorch-on-JAX), optionally converts to native JAX, and applies the Karpathy-style **autoresearch loop** to optimize its performance on TPU.

**This is an optimization program, not a quality program.** Per the wiki's scope rule, any change that alters model semantics (output distribution, accuracy, convergence) is invalid — regardless of speedup. We optimize step time, MFU, tokens/sec, and memory only. Outputs of the optimized model must match the reference within bf16 numerical tolerance.

Individual experiments for this program live as dated files in **this folder**: `<YYYY-MM-DD>-<slug>.md`. Working scripts and configs are split by execution path: [`torchax/`](torchax/README.md) (primary — PyTorch-on-JAX) and [`jax/`](jax/README.md) (secondary — native-JAX port). Profiles for each experiment are captured to `raw/profiles/<YYYY-MM-DD>-<slug>/`. See the "Schema note" section at the bottom.

## Target metrics

| Metric | Direction | Notes |
|---|---|---|
| Step time (training) | ↓ | Wall-clock per training step, excluding compile / warmup. |
| Per-token latency (inference) | ↓ | Decode step latency for autoregressive generation. |
| MFU | ↑ | Model FLOPs Utilization against v6e bf16 peak. |
| Tokens/sec | ↑ | Aggregate throughput (training) or generation rate (inference). |
| Tokens/chip/sec | ↑ | Normalizing TPS per chip. |
| Peak HBM | ↓ | Enables larger batch / longer seq / fewer shards. |

## Hardware

- **Primary**: TPU v6e (Trillium) — the target platform.
- **Reference**: TPU v5e / v5p if available (for generation-comparative roofline calibration).
- **Out-of-scope**: GPU (performance claims from GPU sources are calibration-only, not portable).

## Model

- Source: [google/gemma-4-E4B](https://huggingface.co/google/gemma-4-E4B).
- Architecture: Gemma family (GQA attention, SwiGLU FFN, RMSNorm). Confirm exact config on baseline ingest.
- Initial precision: bf16 weights + bf16 activations + fp32 optimizer states; `default_matmul_precision='bfloat16'`.

## How to run

*Baseline command TBD — first experiment will pin this. Scripts live in [`torchax/`](torchax/README.md) (primary) and [`jax/`](jax/README.md) (secondary).*

Minimum requirements for the baseline:
- torchax checkout (current wiki submodule: [torchax](../../codebases/torchax.md)).
- HF access to the Gemma 4 E4B weights.
- XProf capture enabled (see [xprof — capturing profiles](../../sources/2026-xprof-capturing-profiles.md)); profile dumped to `raw/profiles/<YYYY-MM-DD>-baseline/`.
- HLO dump enabled (`XLA_FLAGS --xla_dump_to=... --xla_dump_hlo_as_text`) for later [hlo-dumping-and-diffing](../../concepts/hlo-dumping-and-diffing.md).

## Code

Working scripts, notebooks, and configs are split by execution path:
- [`torchax/`](torchax/README.md) — **primary**: Gemma 4 E4B run via torchax (PyTorch-on-JAX). The baseline and most hypotheses exercise this path.
- [`jax/`](jax/README.md) — **secondary**: a native-JAX port of the model, lit up once the torchax baseline is stable and a port becomes a hypothesis (e.g., to drop dispatch overhead or reach JAX-only tooling more cheaply). A JAX port must reproduce the torchax baseline's outputs within bf16 tolerance before its numbers count.

Every dated experiment in this folder references the exact script + config used by relative path. Code is tracked in the main wiki git repo (not a submodule).

## Baseline

Not yet captured. Baseline experiment is the first hypothesis in the ranked list below.

| Metric | Value | Date | Experiment |
|---|---|---|---|
| Step time | TBD | — | — |
| MFU | TBD | — | — |
| Tokens/sec | TBD | — | — |
| Peak HBM | TBD | — | — |

## Current best

Same as baseline until the first experiment with `verdict: supported` lands.

## Known bottlenecks

*To be identified from the baseline profile. Candidates to look for based on Gemma-family architecture and Wave 1/2 findings:*
- Attention kernel path — whether torchax routes to `tokamax.dot_product_attention` / Splash or falls back to a chunked XLA kernel.
- SwiGLU path — tokamax has no TPU Pallas kernel for [gated-linear-unit](../../concepts/gated-linear-unit.md); almost certainly XLA on TPU.
- RMSNorm path — similarly no TPU Pallas kernel for [layer-norm](../../concepts/layer-norm.md) in tokamax; likely XLA.
- Activation memory pressure — Gemma-family E4B has depth + long-context activations that push HBM; candidate for [rematerialization](../../concepts/rematerialization.md) and [host-offload](../../concepts/host-offload.md).
- Captured-constants bloat — torchax jit'd functions can capture Python-held tensors as constants (see [jax-huggingface part 3](../../sources/2026-jax-huggingface-part-3.md) for a 13.48 GB instance).

## Open hypotheses

Ranked candidates consolidated from Wave 1/2 findings, the [xprof-mcp TPU optimization guide](../../sources/2026-xprof-mcp-tpu-optimization.md), and the [Ultra-Scale Playbook](../../sources/2025-ultrascale-playbook.md). Ranks are preliminary — will be refined once the baseline profile identifies which apply to this model.

| # | Hypothesis | Expected | Conf. | Effort | Origin |
|---|---|---|---|---|---|
| 0 | **Capture baseline** — run 10–20 steps, dump profile, characterize (roofline position, duty cycle, top HLO ops, memory peak). | n/a | high | S | Program bootstrap — first experiment must be this. |
| 1 | Enable `tokamax.dot_product_attention` (Splash Attention) in place of the default torchax attention path. | 15–40% on attention time | high | S | [tokamax splash-attention](../../sources/2026-tokamax-splash-attention.md), [xprof-mcp TPU_OPTIMIZATION §6](../../sources/2026-xprof-mcp-tpu-optimization.md) |
| 2 | Autotune attention block sizes (`block_q`, `block_kv`, `block_kv_compute`) for Gemma 4's head dim + GQA grouping. | 5–15% on attention | med | S | [tokamax autotuning](../../sources/2026-tokamax-autotuning.md), [attention-block-sizes](../../concepts/attention-block-sizes.md) |
| 3 | Expose Splash Attention backward-pass block sizes (`block_q_dkv`, `block_kv_dkv`, `block_kv_dkv_compute`, `block_q_dq`, `block_kv_dq`) to the autotuner — currently hard-wired to 128. | 5–15% on backward attention | med | M | Wave 1 finding |
| 4 | Write/enable a TPU Pallas kernel for SwiGLU (tokamax currently falls back to XLA on TPU). | 10–20% on FFN | med | L | Wave 1 finding, [gated-linear-unit](../../concepts/gated-linear-unit.md) |
| 5 | Write/enable a TPU Pallas kernel for RMSNorm (tokamax currently falls back to XLA on TPU). | 3–8% on norm layers | med | M | Wave 1 finding, [layer-norm](../../concepts/layer-norm.md) |
| 6 | Selective rematerialization via `jax.checkpoint_policies` instead of full activation checkpointing. | ~+2.7% compute for ~70% activation-memory savings — enables larger batch. | high | S | [xprof-mcp TPU_OPTIMIZATION §4.6, §5](../../sources/2026-xprof-mcp-tpu-optimization.md), [rematerialization](../../concepts/rematerialization.md) |
| 7 | Scan-over-layers (`jax.lax.scan` / `ScannedModule`) — compile time O(N)→O(1). | Compile time, not step time. Enables faster iteration on other hypotheses. | high | M | [scan-over-layers](../../concepts/scan-over-layers.md) |
| 8 | Dimension alignment — ensure all batch/hidden dims honor v6e MXU tile (256×256). | 5–15% if any dim is misaligned | med | S | [dimension-alignment](../../concepts/dimension-alignment.md) |
| 9 | Enable XLA async-collective and latency-hiding scheduler flags. | 5–15% if ICI collectives are exposed | high | S | [async-collectives](../../concepts/async-collectives.md), [latency-hiding-scheduler](../../concepts/latency-hiding-scheduler.md) |
| 10 | FSDP mesh design on v6e — tune shard dims to maximize [ici-roofline](../../concepts/ici-roofline.md) headroom. | workload-specific | med | M | [fsdp](../../concepts/fsdp.md), [sharding](../../concepts/sharding.md) |
| 11 | Ring attention for long-context runs (kernel exists in tokamax but not wired to public API). | only if context length is a goal | low | L | Wave 1 finding, [ring-attention](../../concepts/ring-attention.md) |
| 12 | Tile-scaled FP8 (DeepSeek-V3 style: 1×128 activations, 128×128 weights+scales). | 10–30%, requires v6e MXU FP8 support confirmation. | low | L | [Ultra-Scale Playbook](../../sources/2025-ultrascale-playbook.md), [dtype-strategy](../../concepts/dtype-strategy.md) |
| 13 | Host-offload of optimizer state and/or activations. | HBM savings → larger batch or longer context | med | M | [host-offload](../../concepts/host-offload.md) |
| 14 | Eliminate captured-constants bloat in torchax jit'd functions (use `functional_call` pattern). | memory only, but can unblock larger batch | med | S | [jax-huggingface part 3](../../sources/2026-jax-huggingface-part-3.md) |
| 15 | Static-cache / fixed-shape KV cache for decode (if inference is in scope for this program). | ~8× on decode per jax-huggingface part 3 | high (decode only) | M | [static-cache](../../concepts/static-cache.md), [kv-cache](../../concepts/kv-cache.md) |

## Retired hypotheses

*None yet.*

## History

- **2026-04-22**: Program filed. Baseline not yet captured.

---

## See also

- [autoresearch codebase](../../codebases/autoresearch.md) — methodology reference this program adapts (loop shape, experiment discipline, fair-comparison metric).
- [torchax codebase](../../codebases/torchax.md) — the framework bringing Gemma 4 (PyTorch) to TPU via JAX.
- [tokamax codebase](../../codebases/tokamax.md) — kernel library backing several open hypotheses.
- [jax-huggingface codebase](../../codebases/jax-huggingface.md) — Hugging Face model on TPU via torchax prior art.
- [xprof-mcp — TPU optimization guide](../../sources/2026-xprof-mcp-tpu-optimization.md) — primary source for hypothesis rationales.
- [scaling-book](../../codebases/scaling-book.md) — TPU scaling reference (chapter sources pending Wave 3).

## Sources

*Accumulates as experiments are filed.*

---

## Schema note

This folder is an **intentional deviation** from `SCHEMA.md` as originally written. The schema specifies `wiki/experiments/<YYYY-MM-DD>-<slug>.md` (flat, one file per dated experiment). Here the experiments for one long-running optimization program are **nested under a program folder** so the program's plan (this `README.md`), its dated experiments, and optionally local code live together. See the log entry dated 2026-04-22 for the rationale.

Experiment files within this folder still follow the schema's `experiment` page template and naming convention: `<YYYY-MM-DD>-<slug>.md`, with the required frontmatter (`hypothesis:`, `model:`, `commit:`, `verdict:`).
