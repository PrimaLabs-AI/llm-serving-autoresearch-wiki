# Gemma 4 E4B — native-JAX stack experiments

Experiment writeups + ledger for the **native-JAX (Flax NNX)** path through the Gemma 4 autoresearch program. Sibling of [`../../torchax/experiments/`](../../torchax/experiments/README.md) (torchax stack).

## Why two stacks?

The torchax stack was built first (it reuses HuggingFace's PyTorch model via torchax). The JAX stack is a from-scratch Flax NNX port (see [exp 34](2026-04-23-exp34-jax-baseline-accepted.md)) and reveals whether torchax's dispatch overhead, custom-call boundaries, and HF-shaped quirks are bottlenecks. Both stacks share hardware, mesh conventions, and the verdict-suffix / profile-link discipline defined in [`../../program.md`](../../program.md).

## Contents

- `2026-04-23-exp{N}-<slug>-<verdict-suffix>.md` — per-experiment pages. **exp 34** is the JAX-port baseline; numbering continues from the global exp counter so commit history stays linear across both stacks.
- `OBSERVATIONS.md` — skim-and-reason aggregation log, jax-stack-scoped.
- `RESULTS.tsv` — machine-readable ledger (gitignored).

## Current state

**Exp 35** at **30,386 TPS** (seq=1024, batch=1, fsdp=4, bf16, splash) — flat (+0.33 %) over exp 34's XLA-SDPA baseline. Splash kernel is correct (bit-match loss at step 19: 2.2969) and HLO diff confirms it swaps matmul time from XLA convolution-fusion into Mosaic custom-fusion with a ~49 ms / 3-step net saving — but splash's per-call launch overhead at batch=1 seq=1024 offsets most of it. Peak HBM dropped 16.85 → 16.43 GiB, creating headroom for **exp 36 (batch=3)** where splash's asymptotic win is expected to materialize.

## Queued experiments (highest-expected-gain first)

- **exp 36** — **splash + batch=3** (direct analog of torchax exp 18, +8.0 %). HBM 52.6 % leaves ~2-3 GiB headroom. Confidence high.
- **exp 37** — splash + bf16 CE (tokamax or hand-roll). ~+1-3 % and frees ~1.5 GiB of fp32 logits.
- **exp 38** — scan-over-layers. Easier in native JAX (no torchax kwargs/assertion constraints — see [torchax exp 26 parked blockers](../../torchax/experiments/2026-04-23-exp26-scan-over-layers-potential.md)). Compile-time win primarily.
- **exp 39** — step-1 recompile root-cause (out_shardings / donation).

## See also

- [`../train.py`](../train.py) — the native-JAX trainer.
- [`../model/`](../model/) — Flax NNX port.
- [Shared program protocol](../../program.md).
- [Session ceiling analysis (torchax, exp 25 era)](../../../../analyses/2026-04-23-gemma4-v6e4-optimization-ceiling.md).
