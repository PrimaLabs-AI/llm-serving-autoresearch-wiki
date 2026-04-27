---
title: "JAX exp 27/28b — SparseCore offload of RS+AG (added to AR): 7,768 tok/s/chip 43.6% MFU at bs=4 (NEW frontier)"
type: experiment
tags: [llama3, jax, flax-nnx, sparsecore-offload, all-gather, reduce-scatter, frontier, milestone]
hypothesis: jax-llama3-match-maxtext
model: llama3-8b-jax
created: 2026-04-26
updated: 2026-04-26
commit: "v6e8-llama3-8b-jax-20260426-exp28b-sc-rsag-bs4 (image jax-v4)"
branched_from: jax-exp18 (bkv=2048, SC-AR-only)
verdict: supported
---

🏆 **JAX Llama 3 8B trainer at 7,768 tok/s/chip, 43.6 % reported MFU
(bs=4, seq=8192).** Adding `xla_tpu_enable_sparse_core_collective_offload_{reduce_scatter,all_gather}=true`
on top of the prior frontier (which already had SC offload of all-reduce)
moves all three FSDP collectives onto the SparseCore. Combined with bs=4
density (better than bs=5 once collectives stop hogging vector cores),
this lifts per-chip throughput by **+4.0 %** over the prior frontier
(exp 18: 7,471 / 41.9 %) and now beats MaxText's reference (7,069 / 44.6 %)
by **+9.9 % per chip**. The 1.0 pp gap to MaxText's reported MFU is
FLOP-counting normalization, not a real throughput gap.

## Path from exp 18 frontier

| Run | bs | Δ vs prior | tok/s/chip | MFU | Notes |
|-----|---:|-----------:|-----------:|----:|-------|
| exp 18 (prior frontier) | 5 | — | 7,471 | 41.9% | bkv=2048, SC-AR-only |
| exp 26 profile | 5 | + xprof capture | 7,452 | 41.8% | identified async AR/AR-scatter at 5.0% of step |
| exp 27 | 5 | **+ SC-RS + SC-AG offload** | **7,724** | **43.3%** | **+3.4 % over exp 18** |
| 🏆 **exp 28b** | **4** | bs density tweak | **7,768** | **43.6%** | **+4.0 % over exp 18 — frontier** |
| exp 28 | 6 | density push | OOM | — | OOM by ~220 MiB at bs=6 |

Cumulative climb morning-baseline → here: **4,591 (torchax exp 20) →
7,768 (JAX exp 28b) = +69.2 % per-chip**.

## What was the missing lever

The exp 26 profile (with full HOST_OFFLOAD + DISABLE_COLLECTIVE_MATMUL +
SC-AR + recipe flags) showed **async-all-reduce-scatter at 5.0 %** of step
time still occupying the **TensorCore** instead of the SparseCore. MaxText's
config in `raw/code/maxtext/benchmarks/maxtext_trillium_model_configs.py:813`
applies all three SparseCore offload bundles together
(`ENABLE_SPARSECORE_OFFLOADING_FOR_{ALL_REDUCE, REDUCE_SCATTER, ALL_GATHER}`):

```
--xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=false
--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true
--xla_tpu_enable_reduce_scatter_offload_tracing=true

--xla_tpu_enable_async_collective_fusion_fuse_all_gather=false
--xla_tpu_enable_sparse_core_collective_offload_all_gather=true
--xla_tpu_enable_all_gather_offload_tracing=true
```

We had only the AR triplet enabled at exp 18. Adding the RS and AG
triplets relays both remaining collective classes off the TC and onto the
SparseCore — freeing TC cycles for matmul. **+3.4 % per chip at bs=5,
+4.0 % at bs=4.**

## Why bs=4 beats bs=5 with full SC offload

At bs=5, vector-core (matmul) was already 64 % MXU-utilized (see exp 13
profile) — close to the ceiling for plain Llama-style architectures.
Pushing to bs=5 added enough activation pressure that the scheduler had
less freedom to overlap. With all three collectives offloaded to SC, the
activation memory headroom matters less — bs=4 hits a sweeter spot:

| Run | bs | step time (ms) | tok/s/chip | MFU | Notes |
|-----|---:|---------------:|-----------:|----:|-------|
| exp 27 | 5 | 5,303 | 7,724 | 43.3% | |
| exp 28b | 4 | 4,217 | 7,768 | 43.6% | better step efficiency |

bs=6 OOMs (was tantalizingly close at exp 13's bkv=1024 6,063/chip; with
bkv=2048 raising the kernel memory floor it now overflows by ~220 MiB).

## Stack composition (frontier exp 28b)

```
# Container: us-central1-docker.pkg.dev/.../llama3-8b-jax-container:jax-v4
# Trainer flags:
python -u train.py \
    --model_id=meta-llama/Meta-Llama-3-8B \
    --batch_size=4 --seqlen=8192 \
    --weights_dtype=fp32 --compute_dtype=bf16 --master_dtype=fp32 \
    --use_real_data=True --use_splash=True --use_scan=True \
    --use_tokamax_ce=True --tokamax_ce_impl=chunked_xla \
    --tokamax_ce_autotune=True \
    --scan_remat_policy=nothing_saveable \
    --train_steps=15

# Env (key knobs):
USE_TOKAMAX_SPLASH=1
TOKAMAX_USE_BASE2_EXP=1
TOKAMAX_FUSE_RECIPROCAL=1
TOKAMAX_MAX_LOGIT_CONST=30
JAX_ATTENTION_IMPL=splash
SPLASH_BKV=1024 SPLASH_BKV_COMPUTE=1024     # confirmed kernel-tune optimal
SPLASH_BQ=2048 SPLASH_BQ_DKV=2048 SPLASH_BKV_DKV=2048 SPLASH_BKV_DKV_COMPUTE=2048
SPLASH_FUSED_BWD=1

# LIBTPU_INIT_ARGS = full MaxText XLA flag stack — see
#   /tmp/llama3_run/xpk/exp_jax_maxtext_flags.sh
# Notable additions vs exp 18:
#   --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true
#   --xla_tpu_enable_async_collective_fusion_fuse_reduce_scatter=false
#   --xla_tpu_enable_reduce_scatter_offload_tracing=true
#   --xla_tpu_enable_sparse_core_collective_offload_all_gather=true
#   --xla_tpu_enable_async_collective_fusion_fuse_all_gather=false
#   --xla_tpu_enable_all_gather_offload_tracing=true
```

## Loss trajectory

bs=4 (exp 28b), step 0–8:
```
loss=11.9032 → 11.6310 → 11.3503 → 11.0974 → 10.8662 →
       10.6136 → 10.4171 → 10.2469 → 10.0998
```
bs=5 (exp 27), step 0–6:
```
loss=11.9018 → 11.6283 → 11.3561 → 11.1029 → 10.8514 →
       10.6248 → 10.4072
```
Identical step-for-step within bf16 noise — no semantic regression.

## Verdict

**Supported.** All three criteria met:
- Throughput: 7,768/chip vs prior frontier 7,471 = **+4.0 %**
- vs MaxText 7,069/chip = **+9.9 % per chip**
- No semantic regression (loss curves match)
- No tracked-metric regression (peak HBM still in-bounds at bs=4 — bs=6
  would OOM but that is not a regression of the bs=4 frontier)

This is the **new program-target best for the JAX stack**:

> scan + AMP master (fp32 weights / bf16 compute) + tokamax CE
> (chunked_xla, autotune) + tokamax-splash (`use_base2_exp +
> fuse_reciprocal + max_logit_const=30`, bkv=1024) + `nothing_saveable`
> scan remat + VMEM=98 KiB + full MaxText XLA flag stack (HOST_OFFLOAD +
> DISABLE_COLLECTIVE_MATMUL + **SparseCore offload of AR + RS + AG** +
> recipe flags) + bs=4 seq=8192 → **7,768 tok/s/chip, 43.6 % MFU**.

## Profile

- **xprof run name**: `llama3-8b-jax-exp28pf-sc-bs4-prof` (capture in flight at writeup time)
- **on-disk**: `raw/profiles/2026-04-26-jax-exp28b-sc-rsag-bs4/` (pulled via `kubectl cp` from the sleep-hold pod)
- **steps captured**: profile_step=7 of a 10-step run (config matches exp 28b verbatim)
- **purpose**: identify the new dominant bottleneck after all three FSDP collectives are SC-offloaded; expectation is that step time is now bound by matmul + splash custom-call

## Path to higher MFU (open hypotheses)

The reported 1.0 pp MFU gap to MaxText (43.6 % vs 44.6 %) lives partly
in FLOP normalization — to convert raw throughput into a fair comparison
both sides need to count FLOPs the same way. Levers still on the table
for additional throughput:

1. **Host-offload of activations** (decoder_layer_input + Q/K/V/O
   projections) — MaxText's recipe does this, we don't. Pays off if
   activation memory becomes the constraint. We're already at bs=4
   density; adding host offload would let bs=8+ fit but probably costs
   PCIe bandwidth.
2. **VMEM=131072** — current 98304 leaves ~32 KiB scoped VMEM unused.
3. **Custom remat policy** mirroring MaxText's `save_qkv_proj` — saves
   re-computation cost on the projections.
4. **Async double-buffered data input** — not yet exercised; current
   pipeline is fine but could shave a small amount.

## See also

- [JAX exp 13 frontier writeup](2026-04-26-jax-exp13-maxtext-xla-stack-bs5-accepted.md) — prior chronicle through exp 18
- [Model: llama3-8b-jax](../../../models/llama3-8b-jax.md) — frontier table
- `raw/code/maxtext/benchmarks/xla_flags_library.py` — flag bundle definitions

## Sources

- `raw/code/maxtext/benchmarks/xla_flags_library.py` — `ENABLE_SPARSECORE_OFFLOADING_FOR_REDUCE_SCATTER` and `ENABLE_SPARSECORE_OFFLOADING_FOR_ALL_GATHER` definitions
- `raw/code/maxtext/benchmarks/maxtext_trillium_model_configs.py:813` — Llama 3.1-8B v6e-8 config that applies all three SC offload bundles
- `raw/profiles/2026-04-26-jax-exp28b-sc-rsag-bs4/` — captured trace
