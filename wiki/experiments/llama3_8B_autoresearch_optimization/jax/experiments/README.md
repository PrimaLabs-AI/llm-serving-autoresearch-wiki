# Llama 3 8B — native-JAX stack experiments

Experiment writeups for the **native-JAX (Flax NNX)** path through the Llama 3 8B autoresearch program. Sibling of [`../../torchax/experiments/`](../../torchax/experiments/README.md) (the validated primary stack).

## Why a second stack?

Once the torchax stack reached its MFU plateau (currently ~37–38 % depending on seq), the natural next question is: how much of the remaining gap is torchax overhead vs. fundamental? The native-JAX stack lets us answer that empirically — same model math, same kernels (splash via tokamax, tokamax CE), same sharding plan, but no PyTorch op-lowering layer between the trainer and XLA.

Initial hypothesis: native-JAX is **0–5 %** faster on per-step time at the bs=3 seq=8192 sweet spot. If the gap is bigger, torchax compile-time overhead is significant; if smaller (as we suspect), the work happening on-chip is identical and torchax adds no run-time cost.

## Contents

- [2026-04-26 — exp 13 chronicle (baseline → MaxText XLA stack → exp 18 frontier)](2026-04-26-jax-exp13-maxtext-xla-stack-bs5-accepted.md)
- 🏆 [2026-04-26 — exp 27/28b SparseCore RS+AG offload frontier](2026-04-26-jax-exp27-28-sparsecore-rs-ag-offload-frontier.md) — **7,768 tok/s/chip 43.6 % MFU** (mean across reruns ≈ 7,700/43.3 %)
- 🧪 [2026-04-27 — exp 65/66/67 loss-validation 100 steps](2026-04-27-jax-exp65-67-loss-validation-100steps.md) — full optimization stack is bit-equivalent to minimal-flags baseline (max |Δ| = 0.0003 / 100 steps, bf16 noise floor)

## Current best (trunk)

| Metric | Value | Source |
|--------|------:|--------|
| Throughput | **7,768 tok/s/chip** (62,142 tok/s global) | [exp 28b](2026-04-26-jax-exp27-28-sparsecore-rs-ag-offload-frontier.md) |
| Reported MFU | **43.6 %** (v6e bf16 peak 918 TFLOPs/sec) | exp 28b |
| Step time | 4,217 ms (bs=4 seq=8192) | exp 28b |
| Loss step 0→8 | 11.90 → 10.10 | exp 28b |
| vs MaxText reference | **+9.9 % per-chip throughput** (MaxText: 7,069/chip 44.6 %) | — |
| Stack | scan + AMP master + tokamax CE (chunked_xla) + tokamax-splash (base2/fuse_recip/mlc=30) + nothing_saveable + bs=4 + full MaxText XLA flag stack incl. SC offload of AR/RS/AG | — |

## See also

- [`../`](../) — JAX trainer source.
- [`../../torchax/experiments/`](../../torchax/experiments/) — torchax sibling experiments.
- [`../../program.md`](../../program.md) — shared experiment protocol.
