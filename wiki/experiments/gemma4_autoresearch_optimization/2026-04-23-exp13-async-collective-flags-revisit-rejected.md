---
title: "Exp 13 — async-collective flag bundle revisited at splash+bf16 CE (REJECTED, +12.7% step)"
type: experiment
tags: [experiment, gemma4, xla-flags, async-collectives, regression-reconfirmed]
hypothesis: larger-workload-changes-bundle-outcome
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: "n/a (consolidated into 71a45ae)"
verdict: refuted
---

> *Backfilled from `RESULTS.tsv` + `OBSERVATIONS.md`.*

Re-ran [exp 1](2026-04-23-exp1-async-collective-flags-rejected.md)'s async-collective-fusion flag bundle on the new-best stack (splash + bf16 CE + batch=2 + selective remat). **Same regression as before: +12.7 % step time, 253 → 285.4 ms, TPS 32,340 → 28,700.** Confirms the regression is structural to this workload size, not an artifact of the pre-splash baseline.

## Hypothesis

With splash attention and larger batch, scheduling has more work to hide collectives behind — maybe the bundle that regressed at exp 1 will now win.

## Result

Refuted — the mechanism is the same: scheduler reorder breaks compute-fusion locality, convolution and loop fusion bytes blow up, the modest collective-overlap savings don't compensate.

## Verdict

**REJECTED.** Permanently park the `async_collective_fusion*` flags at this scale. Exp 30 + 31 later confirmed `latency_hiding_scheduler` and `overlap_compute_collective_tc` are not the culprits in this bundle.

## See also

- [exp 1 — original refutation](2026-04-23-exp1-async-collective-flags-rejected.md).
- [exp 30 — latency_hiding alone (flat)](2026-04-23-exp30-latency-hiding-solo-potential.md).
- [exp 31 — overlap_compute_collective_tc alone (flat)](2026-04-23-exp31-overlap-compute-collective-tc-potential.md).

## Sources

- `RESULTS.tsv` row `exp13`.
