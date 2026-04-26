# Llama 3 8B — native-JAX stack experiments

Experiment writeups for the **native-JAX (Flax NNX)** path through the Llama 3 8B autoresearch program. Sibling of [`../../torchax/experiments/`](../../torchax/experiments/README.md) (the validated primary stack).

## Why a second stack?

Once the torchax stack reached its MFU plateau (currently ~37–38 % depending on seq), the natural next question is: how much of the remaining gap is torchax overhead vs. fundamental? The native-JAX stack lets us answer that empirically — same model math, same kernels (splash via tokamax, tokamax CE), same sharding plan, but no PyTorch op-lowering layer between the trainer and XLA.

Initial hypothesis: native-JAX is **0–5 %** faster on per-step time at the bs=3 seq=8192 sweet spot. If the gap is bigger, torchax compile-time overhead is significant; if smaller (as we suspect), the work happening on-chip is identical and torchax adds no run-time cost.

## Contents

- `2026-04-NN-baseline-native-jax.md` — first reference run on v6e-8 (planned).
- `2026-04-NN-exp{N}-<slug>-<verdict-suffix>.md` — per-experiment pages. Verdict suffixes: `-accepted` / `-rejected` / `-potential` (see [program.md § Experiment verdict suffix](../../program.md)).

(No experiments filed yet — this stack ships scaffold-only at the first commit.)

## Current best (trunk)

Pending first run.

## See also

- [`../`](../) — JAX trainer source.
- [`../../torchax/experiments/`](../../torchax/experiments/) — torchax sibling experiments.
- [`../../program.md`](../../program.md) — shared experiment protocol.
