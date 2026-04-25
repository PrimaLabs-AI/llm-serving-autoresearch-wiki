# TPU Model Performance Auto-optimization — Index
*Last updated: 2026-04-25 — 180 pages (26 codebases + 45 sources + 96 concepts + 1 model-program + 4 analyses + 6 analysis subpages + jax-exp47 rejected page)*

*Methodology: autoresearch (see [README](../README.md) + [SCHEMA](../SCHEMA.md)).*

## ⭐ Featured reference — Pallas kernel directory

**[Pallas kernel directory](analyses/2026-04-23-pallas-kernel-directory.md)** — the single most load-bearing reference document in this wiki. A repo-by-repo catalog of **~200 Pallas kernels across ~30 open-source repositories**, each row with a clickable link to the source code, backend (Mosaic-TPU / Mosaic-GPU SM90/SM100 / Triton / XLA fallback), stability tier, any performance claim quoted verbatim from source, application use case, and known callers. Cross-cutting tables group kernels by function (attention, paged-KV, ring, MoE grouped matmul, normalization, GLU, matmul, collectives, SSM/linear-recurrence, cross-entropy, PRNG, non-ML).

Use this page to answer:
- *"Is there a public Pallas implementation of X?"* — look up X in the [functional-category tables](analyses/2026-04-23-pallas-kernel-directory.md#kernel-inventory-by-functional-category).
- *"Where is the canonical / production version of X?"* — every row has a `[file](https://github.com/…)` link to the definition.
- *"Which repos should this wiki ingest next?"* — see [recommended Wave-4 ingestion order](analyses/2026-04-23-pallas-kernel-directory.md#recommended-next-ingestion-wave-wave-4-proposal) ranked by novelty-vs-tokamax and relevance.
- *"What tuning priors already exist?"* — see [Autotune / perf-tuning artifacts — crown jewels](analyses/2026-04-23-pallas-kernel-directory.md#autotune--perf-tuning-artifacts--crown-jewels) (sglang-jax ~2,000-entry RPA table; tpu-inference 600+ quantized_matmul table; marin/levanter deployment-time autotune harness).

**Six subpages** (split per SCHEMA ~500-line rule) hold the full per-kernel detail:

1. [§1 Upstream JAX + tokamax](analyses/pallas-kernel-directory/01-upstream-jax-tokamax.md) — ~55 kernels. The root of the vendoring graph.
2. [§2 AI-Hypercomputer stacks](analyses/pallas-kernel-directory/02-ai-hypercomputer.md) — MaxText, MaxDiffusion, JetStream (archiving 2026-02-01).
3. [§3 Inference engines](analyses/pallas-kernel-directory/03-inference-engines.md) — vLLM tpu-inference (authoritative author), SGLang-JAX, Aphrodite.
4. [§4 Research labs](analyses/pallas-kernel-directory/04-research-labs.md) — Apple AxLearn (uniquely novel Mamba/SSD/RAttention SSM Pallas), DeepMind repos (RecurrentGemma LRU scan, AlphaFold3 v3.0.1 fused GLU).
5. [§5 Frameworks & quantization](analyses/pallas-kernel-directory/05-frameworks-quant.md) — Tunix, Qwix, AQT, Jaxite (non-ML FHE), PyTorch/XLA, Marin/Levanter autotune harness.
6. [§6 Community & research companions](analyses/pallas-kernel-directory/06-community-research.md) — erfanzar/ejkernel (broadest community surface), haoliuhl/ringattention (canonical Ring Attention), plus flagged GPU-only repos misadvertised as "Pallas".

Companion document: [Pallas kernel source survey](analyses/2026-04-23-pallas-kernel-source-survey.md) — the predecessor repo-level inventory this directory refines to kernel-level.

**Status of open Pallas-related wiki hypotheses after this directory closed**:
- Ring Attention Pallas impl → **found** (maxdiffusion, haoliuhl, ejkernel — three patterns to choose among).
- Fused GLU Pallas reference → **found** (AlphaFold3 @ tag v3.0.1; pin the tag).
- Mamba / SSM Pallas → **found** (axlearn — only public source).
- **Zig-Zag ring attention on TPU → confirmed absent from every surveyed repo.** Remains an open algorithmic port from Brandon et al. 2023.
- TPU Pallas RMSNorm/LayerNorm → **confirmed-absent with external evidence**: maxtext/tpu-inference/axlearn/upstream all skip it — XLA-fusion is sufficient. Consistent with the Gemma 4 exp 33 −8.1% empirical result; don't build.

---

## Models (1)
- [Gemma 4 E4B — TPU autoresearch optimization](experiments/gemma4_autoresearch_optimization/README.md) — program page for `google/gemma-4-E4B` on TPU v6e via torchax/JAX. Status: **active, baseline not yet captured**. 16 open hypotheses consolidated from Wave 1/2 findings. *Note: filed under `experiments/<program>/` rather than `models/` — see schema-note in the page and the 2026-04-22 log entry.*

## Hypotheses — ranked, open only (0)
*None yet.*

## Experiments (0)
*None yet.*

## Sources (45)

### xprof documentation — capture & deployment (6)
- [xprof — capturing profiles](sources/2026-xprof-capturing-profiles.md) — how XProf traces are captured (programmatic, on-demand gRPC, continuous snapshot).
- [xprof — JAX profiling](sources/2026-xprof-jax-profiling.md) — `jax.profiler` API (`start_trace`/`stop_trace`/`trace`/`TraceAnnotation`, `ProfileOptions`, TPU trace modes).
- [xprof — PyTorch/XLA profiling](sources/2026-xprof-pytorch-xla-profiling.md) — `xp.start_trace` / `xp.Trace` surface and mark-step pitfalls.
- [xprof — TensorFlow profiling](sources/2026-xprof-tensorflow-profiling.md) — `tf.profiler` APIs (experimental.Trace, client.trace, Keras callback).
- [xprof — Docker deployment](sources/2026-xprof-docker-deployment.md) — running the XProf UI in Docker with GCS-backed logdirs.
- [xprof — Kubernetes deployment](sources/2026-xprof-kubernetes-deployment.md) — aggregator/worker topology, headless gRPC discovery, round-robin LB.

### xprof documentation — UI & viewers (7)
- [xprof — overview page](sources/2026-xprof-overview-page.md) — high-level performance summary: step-time breakdown, MFU, goodput, precisions, duty cycle.
- [xprof — trace viewer](sources/2026-xprof-trace-viewer.md) — event timeline across host/device tracks (Steps, XLA Ops, Framework Ops, TraceMe, Host Offload, SparseCore, Launch Stats).
- [xprof — memory profile](sources/2026-xprof-memory-profile.md) — HBM allocation/fragmentation analysis with timeline and peak-instant breakdown.
- [xprof — memory viewer](sources/2026-xprof-memory-viewer.md) — per-buffer HBM view with program-order axis and on-chip memory tiers.
- [xprof — graph viewer](sources/2026-xprof-graph-viewer.md) — interactive optimized-HLO graph with cost overlays, fusion toggle, DOT export.
- [xprof — utilization viewer](sources/2026-xprof-utilization-viewer.md) — per-HW-unit utilization: MXU, VPU, XU, RPU, DMA paths, ICI/HIB.
- [xprof — terminology](sources/2026-xprof-terminology.md) — vocabulary: profile, session, run, host, device, step.

### xprof documentation — HLO & op stats (5)
- [xprof — HLO op stats](sources/2026-xprof-hlo-op-stats.md) — ranked HLO ops by time; total vs self time; fusion, collective-op, replica-group breakdowns.
- [xprof — HLO op profile](sources/2026-xprof-hlo-op-profile.md) — per-op detail: FLOPs/bytes, arithmetic intensity, roofline classification, wasted-time sort.
- [xprof — framework op stats](sources/2026-xprof-framework-op-stats.md) — JAX/TF framework-level op breakdown; host vs device time; call-stack attribution.
- [xprof — perf counters](sources/2026-xprof-perf-counters.md) — raw HW performance counters (TPU issue counters, GPU CUPTI kernel fingerprints).
- [xprof — custom call profiling](sources/2026-xprof-custom-call-profiling.md) — profiling Pallas/Mosaic custom calls; LLO utilization tracks.

### xprof documentation — roofline & megascale (4)
- [xprof — roofline model](sources/2026-xprof-roofline-model.md) — roofline tool: per-memory-tier rooflines, OI/FLOPs axes, cost-model vs HW-counter FLOPs.
- [xprof — megascale stats](sources/2026-xprof-megascale-stats.md) — aggregated cross-slice collective stall statistics (slack, stall, required bandwidth).
- [xprof — megascale viewer](sources/2026-xprof-megascale-viewer.md) — Perfetto-embedded per-collective action graph (D2H / NetworkSend / NetworkReceive / H2D).
- [xprof — megascale viewer SQL](sources/2026-xprof-megascale-viewer-sql.md) — PerfettoSQL queries over megascale traces; p99/mean heavy-tail diagnostics.

### xprof-mcp documentation (1)
- [xprof-mcp — TPU optimization guide](sources/2026-xprof-mcp-tpu-optimization.md) — **crown-jewel practical guide**: roofline, dimension alignment, dtype strategy, fusion, rematerialization, KV cache, XLA flags, decision trees.

### tokamax documentation (5)
- [tokamax — supported ops](sources/2026-tokamax-supported-ops.md) — kernel matrix by platform (TPU / GPU / both) and backend (Mosaic / Triton / XLA).
- [tokamax — basic usage](sources/2026-tokamax-basic-usage.md) — API surface: `implementation=`, autotune, jax-export caveats.
- [tokamax — splash attention](sources/2026-tokamax-splash-attention.md) — TPU flash-attention variant: tunables (block sizes, QKV layouts), soft-cap, base-2 softmax.
- [tokamax — autotuning](sources/2026-tokamax-autotuning.md) — autotune API, on-disk cache, non-determinism, all-implementations sweep.
- [tokamax — benchmarking](sources/2026-tokamax-benchmarking.md) — kernel-only timing (`xprof_hermetic`, CUPTI); fwd / fwd_res / vjp / fwd_and_vjp modes.

### jax-huggingface tutorial series (4)
- [jax-huggingface part 1 — single-device forward + jit](sources/2026-jax-huggingface-part-1.md) — Llama-2-7B bf16 forward on TPU v6e: 4.37s cold → 13ms cached via torchax + jax.jit; pytree + static-arg cookbook.
- [jax-huggingface part 2 — 8-way tensor parallelism](sources/2026-jax-huggingface-part-2.md) — NeMo-Megatron TP recipe for Llama on 8 TPU v6e chips; 3.8× cached speedup (13ms → 3.4ms); sub-linear scaling flagged.
- [jax-huggingface part 3 — StaticCache + jax.jit decode](sources/2026-jax-huggingface-part-3.md) — Llama-2-7B 50-token decode 130.9s → 14.77s (8.9×) via StaticCache + functional_call; 13.48GB captured-constants warning.
- [jax-huggingface part 4 — SD2 via torchax.compile](sources/2026-jax-huggingface-part-4.md) — Stable Diffusion 2-base on **A100 GPU** (not TPU): 5.9s → 1.07s/image after fixing `methods_to_compile=['decode']` for VAE.

### External publications (1)
- [Ultra-Scale Playbook (2025)](sources/2025-ultrascale-playbook.md) — Tazi et al., Hugging Face, Feb 2025. GPU-cluster LLM-training playbook (5D parallelism, FlashAttention, FP8, kernel engineering). Ingested with GPU↔TPU contrast emphasis; 90 figures saved under `raw/assets/ultrascale-playbook/`.

### Methodology (1)
- [LLM Wiki (Karpathy)](sources/2026-karpathy-llm-wiki.md) — the idea file this wiki's SCHEMA.md descends from. Raw/wiki/schema layers; ingest/query/lint ops; index+log navigation pair; contradiction-flag convention.

### Scaling-book chapters (11, ingested 2026-04-23 from [scaling-book](codebases/scaling-book.md) @ `6cda371`, book dated 2025-02-04)
- [Ch 1 — Rooflines](sources/2025-scaling-book-ch1-roofline.md) — three-constraint model + critical intensity thresholds (v5e 240, v5p 164, H100 296).
- [Ch 2 — How to Think About TPUs](sources/2025-scaling-book-ch2-tpus.md) — TPU hardware (MXU / VPU / VMEM / HBM / ICI / DCN / PCIe); canonical per-chip specs for v5e/v5p.
- [Ch 3 — Sharded Matrices](sources/2025-scaling-book-ch3-sharding.md) — four-case sharded-matmul taxonomy; collective-cost formulas.
- [Ch 4 — Transformer Math](sources/2025-scaling-book-ch4-transformers.md) — FLOPs/params/KV-cache formulas; attention dominates FLOPs only when T > 8D.
- [Ch 5 — Parallelize for Training](sources/2025-scaling-book-ch5-training.md) — DP/FSDP/TP/PP compute-comm thresholds on v5p.
- [Ch 6 — Training LLaMA-3 on TPUs](sources/2025-scaling-book-ch6-applied-training.md) — LLaMA-3-70B on 8960-chip v5p pod, 44 days @ 40% MFU.
- [Ch 7 — Transformer Inference](sources/2025-scaling-book-ch7-inference.md) — prefill/decode split; decode always bandwidth-bound.
- [Ch 8 — Serving LLaMA-3-70B](sources/2025-scaling-book-ch8-applied-inference.md) — min topology by dtype (bf16 4×4, int8 4×2, int4 2×2).
- [Ch 9 — Profiling TPU Programs](sources/2025-scaling-book-ch9-profiling.md) — HLO/LLO/xprof; matmul roofline matched to 0.4%.
- [Ch 10 — Programming TPUs in JAX](sources/2025-scaling-book-ch10-jax.md) — Auto/Explicit/shard_map; collective matmul 1.27× via ppermute overlap.
- [Ch 11 — How to Think About GPUs](sources/2025-scaling-book-ch11-gpus.md) — H100/B200 vs v5p/v5e; FLOPs/$ ranking.

## Codebases (27)

### Wave 4 follow-up — deferred Pallas ecosystem (5, added 2026-04-23)
- [graphcast](codebases/graphcast.md) — commit `08cf736` — DeepMind weather model; splash wrapper + `WeatherMeshMask` (non-LLM block-sparse reference).
- [simply](codebases/simply.md) — commit `f40b81e` — DeepMind serving framework; RPA wrapper that documents the DMA-overhead-bytes autotune heuristic (0.5 MiB virtual bytes).
- [jaxite](codebases/jaxite.md) — commit `e4a3351` — Google FHE library; only non-ML Pallas TPU kernel in this wiki (CGGI bootstrap).
- [qwix](codebases/qwix.md) — commit `b966dc4` — Google quantization framework; `QArray`-aware `pallas_call`; successor to AQT.
- [aqt](codebases/aqt.md) — commit `9d1667e` — deprecated; superseded by qwix.

### Wave 4 — Pallas kernel ecosystem (11, added 2026-04-23)
- [axlearn](codebases/axlearn.md) — commit `b479714` — Apple's training framework; **only public TPU Pallas Mamba1/Mamba2/RAttention SSM kernels**, plus splash extensions (dropout + logit sink); GPU Triton megablox (`arXiv:2507.05411`).
- [tpu-inference](codebases/tpu-inference.md) — commit `a657060d` — vLLM's TPU backend; **broadest novel Pallas surface** (RPA v2/v3, MLA v1/v2, fused_moe v1, quantized_matmul blockwise, all_gather_matmul, GDN, SparseCore, structured-sparse-matmul); crown-jewel tuning tables (1,200+ RPA v2 + 600+ quantized_matmul entries; v6 96 MiB / v7 48 MiB VMEM).
- [maxtext](codebases/maxtext.md) — commit `532c8b3d8` — AI-Hypercomputer reference JAX trainer for Gemma/Llama/DeepSeek/Qwen/Mistral/Kimi; splash + ragged-paged-attention + megablox GMM + MLIR-dialect SparseCore. Closest public analogue of the gemma4 program.
- [maxdiffusion](codebases/maxdiffusion.md) — commit `c98002fe` — AI-Hypercomputer reference diffusion trainer; **only repo where ring-attention is wired as first-class splash-integrated kernel** (2026-04-16).
- [ringattention](codebases/ringattention.md) — commit `d2ea1af` — haoliuhl's canonical Pallas TPU ring-attention (Liu et al. 2023 paper companion); unidirectional, no zig-zag. Closes ultrascale-playbook Gap #2.
- [alphafold3](codebases/alphafold3.md) — commit `231efc9` (**tag v3.0.1**, removed from main) — only public production-grade **Pallas fused GLU** (GPU via Triton-on-Pallas); reference for a future TPU port.
- [recurrentgemma](codebases/recurrentgemma.md) — commit `2efa84d` — Google DeepMind's canonical public Mosaic-TPU LRU Pallas scan (Griffin RG-LRU); ancestor of axlearn Mamba; real+complex accumulators + multi-shard correction.
- [ejkernel](codebases/ejkernel.md) — commit `f2289a0` — single-author community Pallas library (erfanzar); broadest community TPU surface (17 kernels), Apache-2.0.
- [EasyDeL](codebases/EasyDeL.md) — commit `090a03b2` — training/serving framework wrapping ejkernel via an operations registry (same author).
- [sglang-jax](codebases/sglang-jax.md) — commit `7907875a` — SGLang JAX port; mostly vendored from tpu-inference; **novel EAGLE speculative-decoding tree kernels** + ecosystem's largest tuning table (~2,000+ RPA entries v4/v5/v6e/v7).
- [marin](codebases/marin.md) — commit `7a56e016d` — vendors levanter; **deployment-time autotune harness** (kernel-agnostic, shard-aware, compile-cost-aware, GCS-persistent) — the autotune pattern this wiki should emulate.
- [tpu-recipes](codebases/tpu-recipes.md) — commit `e284e361` — AI-Hypercomputer's per-(model, hardware, topology) reproduction recipes for Trillium (v6e) and Ironwood (v7x); **canonical reference for MaxText `tuning_params` blocks** (`remat_policy`, `decoder_layer_input: offload`, per-projection offload, FSDP sharding) on Llama 3.1 / Gemma 3-4 / Mixtral / DeepSeek 3 / Qwen 3 / GPT-OSS / GPT-3 175B, plus matmul + HBM microbenchmarks.

### Wave 1–3 (10)
- [jax](codebases/jax.md) — commit `feb5ba0` — The JAX library itself: transformations, sharding, `jax.profiler`, **`jax.experimental.roofline`**, Pallas DSL, and the first-party reference TPU kernel tree at `jax.experimental.pallas.ops.tpu.*` (splash_attention, paged_attention, ragged_paged_attention, megablox, flash_attention, matmul, all_gather, threefry). Ground-truth for every other codebase in this wiki.
- [pallas-forge](codebases/pallas-forge.md) — commit `090510b` — Pallas kernel auto-tuning framework (tiled matmul, fused RMSNorm+residual, SwiGLU/GeGLU); **forward-only — no custom_vjp** so unusable in training as-is. Already evaluated via gemma4 exp 20 (rejected).
- [jax-huggingface](codebases/jax-huggingface.md) — commit `93328b2` (subfolder of `qihqi/learning_machine`) — 4-part tutorial + scripts running HuggingFace Llama-2-7B and Stable Diffusion under JAX via torchax.
- [xprof](codebases/xprof.md) — commit `2e33c01` — OpenXLA profiler + TensorBoard plugin; canonical metric vocabulary and profile-capture surface for every experiment.
- [xprof-mcp](codebases/xprof-mcp.md) — commit `9970d65` — MCP server exposing 18 tools for agent-driven profile analysis; wraps a local xprof HTTP server and `.xplane.pb` files.
- [torchax](codebases/torchax.md) — commit `8f957d1` — PyTorch backend that runs torch programs on TPU via JAX; op lowering + graph-compile boundary for torch-origin models.
- [tokamax](codebases/tokamax.md) — commit `54bdd95` — Pallas kernel library (splash/flash attention, GLU, layer_norm, ragged_dot, cross-entropy); **direct optimization toolbox**.
- [stablehlo](codebases/stablehlo.md) — commit `ce5d230` — MLIR op-set + dialect reference; consulted when reading HLO dumps.
- [scaling-book](codebases/scaling-book.md) — commit `6cda371` — "How To Scale Your Model" book (DeepMind); 11 chapters to be ingested as sources in Wave 3.
- [autoresearch](codebases/autoresearch.md) — commit `228791f` — Karpathy's autoresearch reference impl (single H100, `val_bpb`); methodological model for this wiki's loop.

## Concepts (96)

### Pallas-authoring patterns (11, stubs — added 2026-04-23)
- [online-softmax-with-logit-sink](concepts/online-softmax-with-logit-sink.md) — axlearn splash extension; add `exp(sink - m_final)` to normalizer.
- [in-kernel-dropout](concepts/in-kernel-dropout.md) — generate dropout mask from prng_key + block indices (avoids HBM mask materialization).
- [two-level-chunk-recomputation](concepts/two-level-chunk-recomputation.md) — SSD / Mamba2 pattern for linear-recurrence kernels.
- [block-sparse-offset-masks](concepts/block-sparse-offset-masks.md) — precomputed `(n_q_blocks, n_kv_blocks)` offset table for paged-attention sliding-window.
- [multi-shard-sequence-parallel-correction](concepts/multi-shard-sequence-parallel-correction.md) — recurrentgemma pattern for sequence-parallel linear recurrences.
- [custom-splash-masks](concepts/custom-splash-masks.md) — subclass `splash_attention_mask.Mask` for any structured adjacency (LLM, weather, etc.).
- [manual-mlir-dialect-pallas](concepts/manual-mlir-dialect-pallas.md) — drop below `pallas_call` to raw MLIR for SparseCore / TCGEN05 / TMEM features.
- [pallas-on-triton-fused-gemm-activation-gemm](concepts/pallas-on-triton-fused-gemm-activation-gemm.md) — GPU fused-GLU pattern; alphafold3 v3.0.1 reference.
- [grouped-program-ids-for-l2](concepts/grouped-program-ids-for-l2.md) — GPU-side PID reordering for L2 cache reuse (not applicable on TPU).
- [dma-overhead-heuristic](concepts/dma-overhead-heuristic.md) — ~0.5 MiB virtual-bytes DMA-setup cost in block-size autotune.
- [nvidia-weight-tile-bytes-limit](concepts/nvidia-weight-tile-bytes-limit.md) — 101,376-byte per-SM shared-memory weight-tile cap (H100/GB10/A100).

### Autotune-harness patterns (3, stubs — added 2026-04-23)
- [jaxpr-hash-cache-keys](concepts/jaxpr-hash-cache-keys.md) — pin autotune-cache to stringified jaxpr (marin/levanter pattern).
- [compile-time-aware-autotune-filtering](concepts/compile-time-aware-autotune-filtering.md) — discard candidates whose compile cost > 0.20 s over baseline.
- [vmem-oom-fallthrough](concepts/vmem-oom-fallthrough.md) — catch `resource_exhausted … vmem` at autotune, demote candidate.

### Hardware facts (1, added 2026-04-23)
- [vmem-budget](concepts/vmem-budget.md) — per-generation VMEM: v4 32, v5e 32, v5p 95, **v6e 96**, **v7 48** MiB (v6e→v7 halves).

### Architecture & hardware (12)
- [memory-hierarchy](concepts/memory-hierarchy.md) — TPU memory stack (HBM, VMEM, SMEM, CMEM, host RAM).
- [hbm](concepts/hbm.md) — on-package high-bandwidth memory.
- [vmem](concepts/vmem.md) — on-chip vector scratchpad.
- [cmem](concepts/cmem.md) — TPU v4-only on-chip memory tier.
- [mxu](concepts/mxu.md) — systolic matrix unit (128×128 or 256×256 depending on generation).
- [vpu](concepts/vpu.md) — vector programmable unit; gateway to MXU/XU/RPU.
- [sparsecore](concepts/sparsecore.md) — TPU v5p/v6e units for embedding-style sparse lookups.
- [tensor-node](concepts/tensor-node.md) — compute unit on a TPU chip (two per chip).
- [ici](concepts/ici.md) — intra-slice Inter-Chip Interconnect.
- [dcn](concepts/dcn.md) — inter-slice Data Center Network.
- [megascale](concepts/megascale.md) — Google multi-slice collective-communication layer over DCN.
- [multi-slice](concepts/multi-slice.md) — workload spanning multiple TPU slices.

### Performance metrics & roofline (13)
- [mfu](concepts/mfu.md) — Model FLOPs Utilization.
- [step-time](concepts/step-time.md) — wall-clock per training step.
- [mxu-utilization](concepts/mxu-utilization.md) — matrix-unit achieved/peak throughput.
- [memory-bandwidth-utilization](concepts/memory-bandwidth-utilization.md) — HBM bandwidth achieved/peak.
- [tpu-duty-cycle](concepts/tpu-duty-cycle.md) — fraction of wall time the TPU was busy.
- [arithmetic-intensity](concepts/arithmetic-intensity.md) — FLOPs per byte; x-axis of roofline.
- [peak-flops](concepts/peak-flops.md) — device's peak FLOPs/s ceiling.
- [hbm-bandwidth](concepts/hbm-bandwidth.md) — peak HBM throughput per TPU generation; sets the slope of the roofline.
- [roofline-model](concepts/roofline-model.md) — compute-vs-memory-bound performance model.
- [ridge-point](concepts/ridge-point.md) — arithmetic intensity at which peak FLOPs becomes reachable.
- [memory-bound](concepts/memory-bound.md) — BW-limited regime (below ridge point).
- [compute-bound](concepts/compute-bound.md) — FLOPs-limited regime (above ridge point).
- [ici-roofline](concepts/ici-roofline.md) — ICI-bandwidth ceiling for sharded ops.

### Compiler & HLO (12)
- [hlo](concepts/hlo.md) — XLA High-Level Optimizer IR.
- [hlo-op](concepts/hlo-op.md) — single HLO operation node.
- [xla-fusion](concepts/xla-fusion.md) — op-fusion compiler pass.
- [xla-flags](concepts/xla-flags.md) — `XLA_FLAGS` / `LIBTPU_INIT_ARGS` configuration surface.
- [custom-call](concepts/custom-call.md) — XLA's external-kernel mechanism.
- [pallas-kernel](concepts/pallas-kernel.md) — user-authored kernel in Pallas DSL.
- [mosaic-kernel](concepts/mosaic-kernel.md) — TPU kernel lowering backend used by Pallas.
- [autotuning](concepts/autotuning.md) — search over kernel-config space for fastest impl.
- [xla-cost-model](concepts/xla-cost-model.md) — compiler's static FLOPs/bytes estimates.
- [latency-hiding-scheduler](concepts/latency-hiding-scheduler.md) — XLA scheduling pass that overlaps collectives with compute.
- [async-collectives](concepts/async-collectives.md) — XLA flags for async all-reduce/all-gather fusion.
- [hlo-dumping-and-diffing](concepts/hlo-dumping-and-diffing.md) — workflow for inspecting pass-by-pass HLO.

### Kernels (9)
- [flash-attention](concepts/flash-attention.md) — tiled SRAM-resident attention algorithm.
- [splash-attention](concepts/splash-attention.md) — TPU-native flash-family attention (sparse-mask, soft-cap).
- [ring-attention](concepts/ring-attention.md) — ring-of-K/V streaming for long-sequence attention; tokamax kernel exists but not wired to public API.
- [ragged-dot](concepts/ragged-dot.md) — grouped matmul for MoE routing.
- [gated-linear-unit](concepts/gated-linear-unit.md) — fused SwiGLU/GEGLU/REGLU (TPU falls back to XLA).
- [layer-norm](concepts/layer-norm.md) — LayerNorm/RMSNorm fused kernel (TPU falls back to XLA).
- [memory-efficient-cross-entropy](concepts/memory-efficient-cross-entropy.md) — fused linear+log-softmax+NLL avoiding `[B,V]` logits.
- [attention-block-sizes](concepts/attention-block-sizes.md) — `block_q`/`block_kv` tunables for flash/splash attention.
- [base2-softmax](concepts/base2-softmax.md) — `exp(x) = 2^(x·log2 e)` rewrite for TPU's native base-2 exp.

### Optimization techniques (7)
- [rematerialization](concepts/rematerialization.md) — recompute activations to save HBM.
- [scan-over-layers](concepts/scan-over-layers.md) — `jax.lax.scan` pattern; O(N)→O(1) compile time.
- [dimension-alignment](concepts/dimension-alignment.md) — batch/hidden shape divisibility rules for MXU.
- [dtype-strategy](concepts/dtype-strategy.md) — bf16/fp32/fp8/int8 mixing; fp32 weights force per-matmul cast.
- [int8-quantization](concepts/int8-quantization.md) — AQT weight-only / full-int8; shifts critical batch.
- [host-offload](concepts/host-offload.md) — async host↔accelerator transfer to reduce HBM pressure.
- [training-memory-budget](concepts/training-memory-budget.md) — rule of thumb: ~16 bytes/param for bf16 + AdamW.

### Parallelism & collectives (12)
- [fsdp](concepts/fsdp.md) — Fully Sharded Data Parallelism (all-gather + reduce-scatter on ICI).
- [tensor-parallelism](concepts/tensor-parallelism.md) — op-level split; cheap within ICI island (≤8).
- [sequence-parallelism](concepts/sequence-parallelism.md) — TP companion that shards LayerNorm/Dropout along sequence; converts internal all-reduce to reduce-scatter + all-gather.
- [context-parallelism](concepts/context-parallelism.md) — parallelism axis that splits the sequence dim for attention.
- [pipeline-parallelism](concepts/pipeline-parallelism.md) — layer-stage parallelism; AFAB/1F1B/interleaved/zero-bubble/DualPipe schedule family; rarely used on a single TPU pod.
- [expert-parallelism](concepts/expert-parallelism.md) — MoE axis; all-to-all dispatch + tokamax `ragged_dot` compute; ICI-rich, DCN-cliff.
- [sharding](concepts/sharding.md) — GSPMD mesh partitioning.
- [collective-communication](concepts/collective-communication.md) — cross-replica ops umbrella.
- [all-gather](concepts/all-gather.md) — assembles one tensor from per-replica shards.
- [all-reduce](concepts/all-reduce.md) — sums and broadcasts across replicas.
- [reduce-scatter](concepts/reduce-scatter.md) — reduces across replicas and distributes the result as shards; FSDP's gradient-reduction half.
- [send-recv-done](concepts/send-recv-done.md) — XLA's four-op async-collective pattern on TPU.

### Inference (5)
- [kv-cache](concepts/kv-cache.md) — per-token key/value cache; makes decode bandwidth-bound at small batch.
- [static-cache](concepts/static-cache.md) — fixed-shape KV cache enabling `jax.jit`.
- [continuous-batching](concepts/continuous-batching.md) — paged-attention serving pattern.
- [decode-profile-signature](concepts/decode-profile-signature.md) — HBM-bound decode diagnostic pattern (dynamic-slice dominant, duty cycle <60%).
- [serving-warmup](concepts/serving-warmup.md) — pre-compile all shapes before serving traffic.

### Profiling (11)
- [profile-capture](concepts/profile-capture.md) — umbrella for XProf capture APIs.
- [jax-trace](concepts/jax-trace.md) — `jax.profiler` capture surface.
- [pytorch-xla-trace](concepts/pytorch-xla-trace.md) — `xp.start_trace` / `xp.Trace` surface.
- [custom-trace-annotations](concepts/custom-trace-annotations.md) — `xp.Trace` / `jax.profiler.TraceAnnotation` / `tf.profiler.experimental.Trace`.
- [trace-me](concepts/trace-me.md) — low-level annotation API underlying the framework-specific trace wrappers.
- [mark-step-sync](concepts/mark-step-sync.md) — PyTorch/XLA lazy-tensor sync boundary.
- [trace-viewer](concepts/trace-viewer.md) — xprof timeline UI.
- [trace-event-categories](concepts/trace-event-categories.md) — Trace Viewer track taxonomy.
- [perfetto](concepts/perfetto.md) — embedded trace-viewing UI used by megascale tools.
- [perf-counters](concepts/perf-counters.md) — raw HW performance counters.
- [llo-utilization](concepts/llo-utilization.md) — custom-call HW-resource usage track in the trace viewer.

## Observations (0)
*None yet.*

## Analyses (5)
- [2026-04-24 Gemma 4 E4B — JAX fp32-master + seq=8192 regime ceiling (exp 52–53)](analyses/2026-04-24-gemma4-jax-fp32master-seq8k-regime.md) — new-regime baseline (exp 52 at 26,807 TPS seq=2048 b=1 fp32-master) + seq=8192 infeasibility probe on v6e-4. Legacy bf16 also OOMs at seq=8192 (memory wall is not AMP-specific). XLA compile-time peak HBM non-monotonic in seq_len (seq=6144 = 49.66 GiB, seq=8192 = 35.18 GiB). Three-branch forward path documented.
- [2026-04-23 Pallas kernel directory](analyses/2026-04-23-pallas-kernel-directory.md) — repo-by-repo catalog of ~200 Pallas kernels across ~30 repos, with source-code refs, stability, perf claims, use cases, and callers. Cross-cutting functional-category tables + 6 subpages ([§1 JAX+tokamax](analyses/pallas-kernel-directory/01-upstream-jax-tokamax.md), [§2 AI-Hypercomputer](analyses/pallas-kernel-directory/02-ai-hypercomputer.md), [§3 Inference engines](analyses/pallas-kernel-directory/03-inference-engines.md), [§4 Research labs](analyses/pallas-kernel-directory/04-research-labs.md), [§5 Frameworks & quant](analyses/pallas-kernel-directory/05-frameworks-quant.md), [§6 Community](analyses/pallas-kernel-directory/06-community-research.md)). Confirms Zig-Zag ring attention absent everywhere; identifies AlphaFold3 @ v3.0.1 fused GLU and apple/axlearn SSM kernels as the key novel content.
- [2026-04-23 Pallas kernel source survey](analyses/2026-04-23-pallas-kernel-source-survey.md) — web-research inventory of every public source of Pallas kernel code. Identifies 5 top ingest candidates (maxtext, tpu-inference, maxdiffusion, axlearn, sglang-jax) and updates 3 open hypothesis candidates on the Ultra-Scale Playbook page with reference implementations found in the wild.
- [2026-04-23 Gemma 4 E4B on v6e-4 — optimization ceiling (exp 1–33, torchax stack)](analyses/2026-04-23-gemma4-v6e4-optimization-ceiling.md) — synthesis of the 33-experiment torchax-stack loop. Best = exp 25 (33,372 TPS, +9.2% over baseline). Loop at diminishing returns; identifies Pallas-fuses-into-matmul lesson from exp 33 and proposes next levers (scan-over-layers Option B, compile cache, hardware scale-up).
- [2026-04-24 Gemma 4 E4B — JAX-stack ceiling + process retrospective (exp 34–42)](analyses/2026-04-24-gemma4-jax-ceiling-and-process-retrospective.md) — JAX-stack analog. Best = exp 36 (34,614 TPS / 23.05 % MFU, **+3.7 % over torchax session-best**). Explains why JAX beats torchax on same hardware (compile-time HBM ~1.25 GiB lower; no torchax dispatch overhead). Includes cross-stack generalizable lessons + **process retrospective** with concrete `program.md` / `SCHEMA.md` rule additions.
