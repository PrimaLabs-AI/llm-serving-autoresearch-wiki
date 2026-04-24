# Gemma 4 E4B — native-JAX stack experiments

Experiment writeups + ledger for the **native-JAX (Flax NNX)** path through the Gemma 4 autoresearch program. Sibling of [`../../torchax/experiments/`](../../torchax/experiments/README.md) (torchax stack).

## Why two stacks?

The torchax stack was built first (it reuses HuggingFace's PyTorch model via torchax). The JAX stack is a from-scratch Flax NNX port (see [exp 34](2026-04-23-exp34-jax-baseline-accepted.md)) and reveals whether torchax's dispatch overhead, custom-call boundaries, and HF-shaped quirks are bottlenecks. Both stacks share hardware, mesh conventions, and the verdict-suffix / profile-link discipline defined in [`../../program.md`](../../program.md).

## Contents

- `2026-04-23-exp{N}-<slug>-<verdict-suffix>.md` — per-experiment pages. **exp 34** is the JAX-port baseline; numbering continues from the global exp counter so commit history stays linear across both stacks.
- `OBSERVATIONS.md` — skim-and-reason aggregation log, jax-stack-scoped.
- `RESULTS.tsv` — machine-readable ledger (gitignored).

## Current state

**Exp 36** remains the JAX-stack best at **34,614 TPS** (seq=1024, batch=3, fsdp=4, bf16, splash) — **+13.9 %** over exp 35 and **+3.7 % over the torchax session-best** ([exp 25, 33,372 TPS](../../torchax/experiments/2026-04-23-exp25-splash-block1024-accepted.md)). Step time 355.0 ms/step, peak HBM **27.11 GiB / 31.25 GiB = 86.75 %** (fits comfortably with 4.14 GiB of headroom). HLO-op diff vs exp 35 (b=1): splash `custom fusion` near-constant (169 → 175 ms, ×1.03) while matmul `convolution fusion` grew ×2.75 and `loop fusion` ×3.81 — per-call-overhead amortization mechanism per exp 35's predictions. New bottleneck surfaces at b=3: `loop fusion` (28.1 % of step) and `collective-permute-done` (12.2 %, didn't exist at b=1).

**Exp 37** (bf16 CE env-var gate on top of exp 36) landed flat at **34,629 TPS (+0.04 %, within noise)** — the native-JAX port was already running bf16 CE by construction since exp 34, so the torchax-exp-12-style win was a no-op-by-construction. Durable artifact: the `JAX_CE_DTYPE={bf16,fp32}` gate in `train.py`, useful for regression guards on future LM-head refactors. Peak HBM 27.45 GiB / 87.84 % (unchanged heap, +0.34 GiB stack — free headroom 3.80 GiB).

## Queued experiments (highest-expected-gain first)

- **exp 38** — **collective-permute-done investigation**. 12.1 % of step time at b=3 (549 ms/3-step); `in_shardings` / `out_shardings` audit on the jitted step might reclaim half. Expected +5–6 %. Confidence medium. **Now highest-expected-value open hypothesis.**
- **exp 39** — **Pallas RMSNorm kernel** (210 calls/step, single-HBM-pass). Expected +3–8 % on `loop fusion`. Effort M.
- **exp 40** — scan-over-layers. Easier in native JAX. Compile-time win primarily (step 0: 167 s → ~5 s).
- **exp 41** — b=4. Gated on exp 38 landing; 3.80 GiB free today, b=4 adds ~3.5 GiB.

## See also

- [`../train.py`](../train.py) — the native-JAX trainer.
- [`../model/`](../model/) — Flax NNX port.
- [Shared program protocol](../../program.md).
- [Session ceiling analysis (torchax, exp 25 era)](../../../../analyses/2026-04-23-gemma4-v6e4-optimization-ceiling.md).
