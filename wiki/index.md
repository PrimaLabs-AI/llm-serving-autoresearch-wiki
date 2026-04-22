# TPU Performance Autoresearch Wiki — Index
*Last updated: 2026-04-22 — 122 pages (8 codebases + 33 sources + 81 concepts)*

## Models (0)
*None yet.*

## Hypotheses — ranked, open only (0)
*None yet.*

## Experiments (0)
*None yet.*

## Sources (33)

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

## Codebases (8)
- [jax-huggingface](codebases/jax-huggingface.md) — commit `93328b2` (subfolder of `qihqi/learning_machine`) — 4-part tutorial + scripts running HuggingFace Llama-2-7B and Stable Diffusion under JAX via torchax.
- [xprof](codebases/xprof.md) — commit `2e33c01` — OpenXLA profiler + TensorBoard plugin; canonical metric vocabulary and profile-capture surface for every experiment.
- [xprof-mcp](codebases/xprof-mcp.md) — commit `9970d65` — MCP server exposing 18 tools for agent-driven profile analysis; wraps a local xprof HTTP server and `.xplane.pb` files.
- [torchax](codebases/torchax.md) — commit `8f957d1` — PyTorch backend that runs torch programs on TPU via JAX; op lowering + graph-compile boundary for torch-origin models.
- [tokamax](codebases/tokamax.md) — commit `54bdd95` — Pallas kernel library (splash/flash attention, GLU, layer_norm, ragged_dot, cross-entropy); **direct optimization toolbox**.
- [stablehlo](codebases/stablehlo.md) — commit `ce5d230` — MLIR op-set + dialect reference; consulted when reading HLO dumps.
- [scaling-book](codebases/scaling-book.md) — commit `6cda371` — "How To Scale Your Model" book (DeepMind); 11 chapters to be ingested as sources in Wave 3.
- [autoresearch](codebases/autoresearch.md) — commit `228791f` — Karpathy's autoresearch reference impl (single H100, `val_bpb`); methodological model for this wiki's loop.

## Concepts (81)

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

## Analyses (0)
*None yet.*
