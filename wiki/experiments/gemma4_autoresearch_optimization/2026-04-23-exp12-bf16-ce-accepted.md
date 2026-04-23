---
title: "Exp 12 — bf16 cross-entropy at seq=1024 b=2 + splash (ACCEPTED, +5.8% new best)"
type: experiment
tags: [experiment, gemma4, bf16, cross-entropy, tps-win]
hypothesis: bf16-ce-safe-with-final-logit-softcap
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: "n/a (consolidated into 71a45ae)"
verdict: supported
---

> *Backfilled from `RESULTS.tsv` row + session memory.*

Removed the `.to(torch.float32)` cast on `flat_logits` before `log_softmax` — cross-entropy now runs in bf16 end-to-end. **Result: 32,340 TPS, +5.8 % over baseline, new best at the time.** Loss trajectory identical to exp 8 (fp32 CE), despite bf16 log-softmax.

## Hypothesis

Gemma 4's `final_logit_softcapping = 30.0` bounds logits to ±30 before the LM head softmax. At that range, bf16's ~8e-3 relative precision is enough for stable log-softmax. Cost: bf16 CE saves the fp32 `[B, S, V]` logits materialization (~1.5 GiB at B=2 seq=1024 V=262144).

## Results

| Metric | Exp 8 (splash, fp32 CE) | **Exp 12 (splash, bf16 CE)** | Δ |
|---|---|---|---|
| TPS | 31,387 | **32,340** | **+3.0 %** (+5.8 % vs baseline) |
| Step time | 261 ms | 253.3 ms | −3.0 % |
| Peak HBM | 25.85 GiB | 24.79 GiB | −4.1 % |
| Loss | 3.82 → 1.55 | match | identical |

## Verdict

**SUPPORTED.** Stacks cleanly with splash, free memory, no correctness impact. Merged.

## See also

- [exp 8 — splash attention (fp32 CE)](2026-04-23-exp8-splash-attention-accepted.md) — predecessor.
- [exp 14 — bf16 CE at seq=2048 b=1](2026-04-23-exp14-splash-seq2048-bf16ce-accepted.md) — long-seq variant.
- [memory-efficient-cross-entropy concept](../../concepts/memory-efficient-cross-entropy.md).

## Sources

- `RESULTS.tsv` row `exp12`.
