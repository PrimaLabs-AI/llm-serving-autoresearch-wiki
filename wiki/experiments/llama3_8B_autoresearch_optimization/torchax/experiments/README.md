# Llama 3 8B — torchax stack experiments

Experiment writeups for the **torchax** (PyTorch-on-JAX) path through the
Llama 3 8B autoresearch program. Sibling of
[`../../jax/experiments/`](../../jax/experiments/README.md) (native-JAX
stack) and [`../../maxtext/experiments/`](../../maxtext/experiments/README.md)
(MaxText reference baseline).

## Contents

- `2026-04-25-baseline.md` — initial reference run on v6e-8 (HF transformers
  + JittableModule + torchax shard_map FSDP).
- `2026-04-25-exp{N}-<slug>-<verdict-suffix>.md` — per-experiment pages.
  Verdict suffixes: `-accepted` / `-rejected` / `-potential` (see
  [program.md § Experiment verdict suffix](../../program.md)).
- (No `OBSERVATIONS.md` or `RESULTS.tsv` yet — to be added when this stack
  matures past first-batch experiments.)

## Current best (trunk)

**Exp 3** at **57,154 TPS** (bs=4 seq=1024 fsdp=8 v6e-8) — **35.7 % MFU**,
**+55.6 % vs baseline**. Stack: splash Pallas (production block sizes from
[`../splash_attn.py`](../splash_attn.py) — to be re-tuned per
[exp 8](2026-04-25-exp8-splash-kernel-autotune-potential.md)) + bf16 +
remat=`nothing_saveable`. See
[2026-04-25-exp3-splash-bs4-accepted.md](2026-04-25-exp3-splash-bs4-accepted.md)
and the [exp 5 (seq=2048 trunk)](2026-04-25-exp5-splash-seq2k-accepted.md)
alternate path. MaxText reference ceiling on the same hardware shape is
44.6 % MFU — gap of ~9 pp remaining.

## Latest finding — kernel-only autotune (exp 8)

[Exp 8](2026-04-25-exp8-splash-kernel-autotune-potential.md) is the first
**kernel-only** experiment in this program. Sweeps splash `BlockSizes` ×
`q_layout` × `use_fused_bwd_kernel` for the (B, Hq, L, hd) shapes the
trainer feeds the kernel — single TPU v6e chip, no training loop. Found a
config that beats production by **+30-32 % on fwd+bwd kernel time** at every
shape we use. Verdict: `potential`; full-training validation queued as exp 9.

The harness ([`../tune_splash.py`](../tune_splash.py)) is reusable for any
future Pallas kernel sweep — its 171-config 3-shape run took **3 minutes
wall-clock**, vs ~45-90 min for an equivalent set of full XPK training runs
(and resolves sub-1 % deltas that get lost in step-time noise).

## See also

- [`../train.py`](../train.py) — the torchax trainer (primary runner).
- [`../splash_attn.py`](../splash_attn.py) — production splash kernel wrapper
  (the autotune target for exp 8).
- [`../tune_splash.py`](../tune_splash.py) — kernel-only autotune harness.
- [`../model/`](../model/) — sharding wiring.
- [Shared program protocol](../../program.md).
