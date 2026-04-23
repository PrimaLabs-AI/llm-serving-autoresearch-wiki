---
title: "Exp 24 — splash QKVLayout.SEQ_MINOR (ACCEPTED, +0.5% marginal new best)"
type: experiment
tags: [experiment, gemma4, pallas, splash-attention, qkv-layout, tps-win-marginal]
hypothesis: seq-minor-improves-hbm-streaming
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: 8cec59e
verdict: supported
---

> *Backfilled from `RESULTS.tsv` + commits `8cec59e`, `0fabe79`.*

Flipped splash's Q/K/V memory layout from the default `HEAD_DIM_MINOR` to `SEQ_MINOR`. **+0.5 % TPS: 33,193 vs 33,016 (exp 18).** Marginal but reproducible.

## Hypothesis

Tokamax's `pallas_mosaic_tpu` exposes `q_layout / k_layout / v_layout` as autotune knobs. `SEQ_MINOR` places the sequence dim adjacent in memory — may improve HBM streaming pattern when the batch dim is being streamed. Picking one of the well-known alternatives in tokamax's search space.

## Setup

```diff
     block_sizes = sa_kernel.BlockSizes(
         ...
         use_fused_bwd_kernel=True,
-        # default: HEAD_DIM_MINOR
+        q_layout=sa_kernel.QKVLayout.SEQ_MINOR,
+        k_layout=sa_kernel.QKVLayout.SEQ_MINOR,
+        v_layout=sa_kernel.QKVLayout.SEQ_MINOR,
     )
```

## Results

| Metric | Exp 18 (HEAD_DIM_MINOR) | **Exp 24 (SEQ_MINOR)** | Δ |
|---|---|---|---|
| TPS | 33,016 | **33,193** | **+0.5 %** (+8.6 % vs baseline) |
| Loss | clean | match | identical |

## Verdict

**SUPPORTED.** Merged (commit `0fabe79`).

## See also

- [exp 18 — previous best (default layout)](2026-04-23-exp18-fused-bwd-batch3-accepted.md).
- [exp 25 — splash block=1024 on top of SEQ_MINOR](2026-04-23-exp25-splash-block1024-accepted.md) — the next ratchet.
- [tokamax splash attention doc](../../sources/2026-tokamax-splash-attention.md).

## Sources

- `RESULTS.tsv` row `exp24`.
- Commits `8cec59e`, `0fabe79`.
