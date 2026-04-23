---
title: "Exp 23 — seq=2048 b=2 with fused_bwd (REJECTED — same 1.25 GiB ceiling)"
type: experiment
tags: [experiment, gemma4, oom, long-seq, batch-growth, hbm-ceiling]
hypothesis: fused-bwd-unblocks-seq2048-b2
model: gemma4-e4b-torchax-jax
created: 2026-04-23
updated: 2026-04-23
commit: 4164f12
verdict: crash
---

> *Backfilled from `RESULTS.tsv` + commit `4164f12` message.*

Tried seq=2048 batch=2 with fused_bwd. **OOM at compile time — same 1.25 GiB threshold as exp 10, exp 11, exp 22.**

## Hypothesis

fused_bwd reduces backward-pass memory; maybe enough to unblock seq=2048 b=2 (the spot that's OOM'd with every other config).

## Result

No. The 1.25 GiB gap is structural.

## Verdict

**REJECTED — crash.** Not merged. The same lesson as exp 22: batch/seq ratchet is blocked by XLA compile-time accounting; changing which kernel handles the bwd doesn't help.

## See also

- [exp 10](2026-04-23-exp10-seq2048-batch2-bf16ce-rejected.md), [exp 11](2026-04-23-exp11-offload-remat-rejected.md), [exp 22](2026-04-23-exp22-batch4-fused-bwd-rejected.md) — same 1.25 GiB ceiling.

## Sources

- `RESULTS.tsv` row `exp23`.
- Commit `4164f12`.
