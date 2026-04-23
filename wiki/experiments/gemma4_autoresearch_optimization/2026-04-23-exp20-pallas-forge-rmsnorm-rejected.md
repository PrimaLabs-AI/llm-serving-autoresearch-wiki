---
title: "Exp 20 — pallas-forge RMSNorm integration (REJECTED — no custom_vjp)"
type: experiment
tags: [experiment, gemma4, pallas, rmsnorm, invalid, pallas-forge]
hypothesis: pallas-forge-rmsnorm-beats-xla
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: "1b0276f (integrate), eb4db37 (crash)"
verdict: invalid
---

> *Backfilled from `RESULTS.tsv` + commits `1b0276f`, `eb4db37`.*

Monkey-patched `Gemma4RMSNorm.forward` to call pallas-forge's Pallas RMSNorm kernel via `torchax.interop.call_jax`. **Crashed on backward: pallas-forge's kernel has no `custom_vjp` registered, so `jax.grad` cannot differentiate through it. Error: "Linearization failed to produce known values".** Parked.

## Hypothesis

pallas-forge advertises a "Fused RMSNorm + Residual" kernel with 3.44× speedup vs XLA on v5e. If the same holds on v6e-4 and we can swap it into Gemma 4's RMSNorm call sites, the memory-bound `loop_fusion` bucket should shrink.

## Why it failed

pallas-forge exposes a forward kernel but not a backward. For a training workload, forward-only is unusable — the jit trace fails at `jax.value_and_grad` time.

## Verdict

**REJECTED / INVALID.** Not merged. The kernel itself may be correct; it just can't be used for training without a hand-rolled backward. That's what [exp 33](2026-04-23-exp33-pallas-rmsnorm-rejected.md) later delivered (and refuted for a different reason — XLA was already fusing RMSNorm with neighbor matmuls).

## See also

- [exp 33 — hand-written Pallas RMSNorm with custom_vjp](2026-04-23-exp33-pallas-rmsnorm-rejected.md) — the follow-through that revealed the fusion-boundary tax.
- [pallas-forge submodule](../../../raw/code/pallas-forge) — the upstream library.
- [pallas-kernel concept](../../concepts/pallas-kernel.md).

## Sources

- `RESULTS.tsv` row `exp20`.
- Commits `1b0276f` (integrate), `eb4db37` (crash on bwd).
