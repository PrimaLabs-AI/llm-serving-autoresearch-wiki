# Gemma 4 E4B — native-JAX stack experiments

Experiment writeups + ledger for the **native-JAX (Flax NNX)** path through the Gemma 4 autoresearch program. Sibling of [`../../torchax/experiments/`](../../torchax/experiments/README.md) (torchax stack).

## Why two stacks?

The torchax stack was built first (it reuses HuggingFace's PyTorch model via torchax). The JAX stack is a from-scratch Flax NNX port (see [exp 34](2026-04-23-exp34-jax-baseline-accepted.md)) and reveals whether torchax's dispatch overhead, custom-call boundaries, and HF-shaped quirks are bottlenecks. Both stacks share hardware, mesh conventions, and the verdict-suffix / profile-link discipline defined in [`../../program.md`](../../program.md).

## Contents

- `2026-04-23-exp{N}-<slug>-<verdict-suffix>.md` — per-experiment pages. **exp 34** is the JAX-port baseline; numbering continues from the global exp counter so commit history stays linear across both stacks.
- `OBSERVATIONS.md` — skim-and-reason aggregation log, jax-stack-scoped.
- `RESULTS.tsv` — machine-readable ledger (gitignored).

## Current state

**Exp 34** at **30,285 TPS** (seq=1024, batch=1, fsdp=4, bf16, XLA SDPA) — matches the torchax baseline-seq1024 (30,570 TPS) within noise. The −9.2 % gap vs torchax exp 25 (33,372 TPS, the session-best) is entirely explained by missing splash attention + missing batch=3 + missing bf16 CE + missing fused_bwd. Each of those is a queued experiment on this stack.

## Queued experiments (highest-expected-gain first)

- **exp 35** — splash Pallas attention in JAX (mirrors torchax exp 8). Expected: ~+2.7 % → closes the biggest chunk of the gap.
- **exp 36** — scan-over-layers. Easier in native JAX (no torchax kwargs/assertion constraints — see [torchax exp 26 parked blockers](../../torchax/experiments/2026-04-23-exp26-scan-over-layers-potential.md)).
- **exp 37** — tokamax memory-efficient cross-entropy.
- **exp 38** — step-1 recompile root-cause (out_shardings / donation).

## See also

- [`../train.py`](../train.py) — the native-JAX trainer.
- [`../model/`](../model/) — Flax NNX port.
- [Shared program protocol](../../program.md).
- [Session ceiling analysis (torchax, exp 25 era)](../../../../analyses/2026-04-23-gemma4-v6e4-optimization-ceiling.md).
