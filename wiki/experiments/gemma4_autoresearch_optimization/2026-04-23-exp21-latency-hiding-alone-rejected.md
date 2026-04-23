---
title: "Exp 21 — latency-hiding scheduler alone on splash+fused_bwd best (REJECTED, −16.6%)"
type: experiment
tags: [experiment, gemma4, xla-flags, scheduler, regression]
hypothesis: scheduler-alone-is-the-good-bit
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: b26e75c
verdict: refuted
---

> *Backfilled from `RESULTS.tsv` + commit `b26e75c`.*

Enabled `--xla_tpu_enable_latency_hiding_scheduler=true` on the splash+fused_bwd best-config (batch=2). **−16.6 % step time.** Same regression pattern as the full bundle in exp 1 / exp 13.

## Hypothesis

Maybe the exp 1 / exp 13 regression was driven by the scheduler alone, not the `async_collective_fusion*` flags. Isolating might reveal the innocent.

## Result

Refuted — flag alone still regresses at this workload + this config.

> **Later correction**: [exp 30](2026-04-23-exp30-latency-hiding-solo-potential.md) ran the same flag at the newer exp 25 stack (batch=3, SEQ_MINOR, block=1024) and found it **flat**. The exp 21 regression is therefore specific to the batch=2 stack snapshot — possibly interacting with some other diff in the config. The authoritative verdict on `latency_hiding_scheduler` is "flat in isolation" per exp 30. Keep exp 21 parked as a curious data point.

## Verdict

**REJECTED** at the time; effectively **INCONCLUSIVE** in light of exp 30's later flat result. Not merged.

## See also

- [exp 30 — latency-hiding at exp 25 stack (flat)](2026-04-23-exp30-latency-hiding-solo-potential.md) — authoritative.
- [exp 1 — original bundle regression](2026-04-23-exp1-async-collective-flags-rejected.md).
- [exp 31 — overlap_compute_collective_tc alone (flat)](2026-04-23-exp31-overlap-compute-collective-tc-potential.md).

## Sources

- `RESULTS.tsv` row `exp21`.
- Commit `b26e75c`.
