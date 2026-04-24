# Gemma 4 E4B — native-JAX stack experiments

Experiment writeups + ledger for the **native-JAX (Flax NNX)** path through the Gemma 4 autoresearch program. Sibling of [`../../torchax/experiments/`](../../torchax/experiments/README.md) (torchax stack).

## Why two stacks?

The torchax stack was built first (it reuses HuggingFace's PyTorch model via torchax). The JAX stack is a from-scratch Flax NNX port (see [exp 34](2026-04-23-exp34-jax-baseline-accepted.md)) and reveals whether torchax's dispatch overhead, custom-call boundaries, and HF-shaped quirks are bottlenecks. Both stacks share hardware, mesh conventions, and the verdict-suffix / profile-link discipline defined in [`../../program.md`](../../program.md).

## Contents

- `2026-04-23-exp{N}-<slug>-<verdict-suffix>.md` — per-experiment pages. **exp 34** is the JAX-port baseline; numbering continues from the global exp counter so commit history stays linear across both stacks.
- `OBSERVATIONS.md` — skim-and-reason aggregation log, jax-stack-scoped.
- `RESULTS.tsv` — machine-readable ledger (gitignored).

## Current state

**Exp 36** at **34,614 TPS** (seq=1024, batch=3, fsdp=4, bf16, splash) — **+13.9 %** over exp 35 and **+3.7 % over the torchax session-best** ([exp 25, 33,372 TPS](../../torchax/experiments/2026-04-23-exp25-splash-block1024-accepted.md)). Step time 355.0 ms/step, peak HBM **27.11 GiB / 31.25 GiB = 86.75 %** (fits comfortably with 4.14 GiB of headroom). The JAX stack now leads both stacks, and it got there without bf16 CE or fused_bwd-specific tuning yet. HLO-op diff vs exp 35 (b=1): splash `custom fusion` near-constant (169 → 175 ms, ×1.03) while matmul `convolution fusion` grew ×2.75 and `loop fusion` ×3.81 — exactly the per-call-overhead amortization mechanism predicted in exp 35's writeup. New bottleneck surfaces at b=3: `loop fusion` (28.1 % of step) and `collective-permute-done` (12.2 %, didn't exist at b=1).

## Queued experiments (highest-expected-gain first)

- **exp 37** — **splash + b=3 + bf16 CE** (hand-roll first, tokamax variant second). Frees ~1.5 GiB + trims one pass over logits. Expected +1–3 %. Confidence medium.
- **exp 38** — **collective-permute-done investigation**. 12.2 % of step at b=3 is a huge new bucket; `in_shardings` / `out_shardings` audit might reclaim half. Expected +5–6 %. Confidence medium.
- **exp 39** — **Pallas RMSNorm kernel** (210 calls/step, single-HBM-pass). Expected +3–8 % on `loop fusion`. Effort M.
- **exp 40** — scan-over-layers. Easier in native JAX. Compile-time win primarily (step 0: 167 s → ~5 s).
- **exp 41** — b=4. Gated on exp 37 landing; 4.14 GiB free today, b=4 adds ~3.5 GiB.

## See also

- [`../train.py`](../train.py) — the native-JAX trainer.
- [`../model/`](../model/) — Flax NNX port.
- [Shared program protocol](../../program.md).
- [Session ceiling analysis (torchax, exp 25 era)](../../../../analyses/2026-04-23-gemma4-v6e4-optimization-ceiling.md).
