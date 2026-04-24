---
title: "Gemma 4 E4B — JAX-stack optimization ceiling + process retrospective"
type: analysis
tags: [analysis, gemma4, v6e4, jax, ceiling, retrospective, process-improvement]
created: 2026-04-24
updated: 2026-04-24
---

Synthesis of the JAX-stack arc (exp 34 → 42) plus **process retrospective** from ~42 experiments across both stacks. Sibling of [2026-04-23-gemma4-v6e4-optimization-ceiling.md](2026-04-23-gemma4-v6e4-optimization-ceiling.md) which covered only the torchax stack.

## JAX-stack ceiling

**Current JAX-stack best = exp 36: 34,614 TPS, 23.05 % MFU** at seq=1024 batch=3 fsdp=4 bf16 with splash Pallas + selective remat + bf16 CE. This is **+3.7 % over the torchax session-best (exp 25, 33,372 TPS / 22.22 % MFU)** on the same hardware.

### Trajectory

```
30,285  exp 34  JAX port baseline (XLA SDPA)    (matches torchax baseline-seq1024 within noise)
30,386  exp 35  + splash attention              (+0.33 %, flat at b=1)
34,614  exp 36  + batch=3                       (+13.9 %, NEW JAX BEST, overtakes torchax)
34,629  exp 37  + bf16 CE (no-op, already was)  (+0.04 %)
34,634  exp 38  + latency-hiding scheduler      (+0.06 %, flat)
32,402  exp 39  b=4 instead of b=3              (−6.4 %, memory pressure)
31,809  exp 40  seq=2048 b=2 splash             (accepted as long-seq ref, −8.1 % vs b=3 s=1024)
29,832  exp 41  VMEM limit 512 MiB              (−13.8 %, over-commits)
34,629  exp 42  async_collective_fusion alone   (+0.04 %, flat)
```

Exp 37–42: **6 consecutive experiments, zero TPS gains**. Same diminishing-returns ceiling pattern as the torchax exp 26–33 arc.

### Why JAX beats torchax on same hardware

The native-JAX port removes overhead that torchax adds between PyTorch and JAX:

1. **Compile-time HBM ~1.25 GiB lower** — the `CompileTimeHbmOom` pattern torchax hit at four configs (exp 10, 11, 22, 23 — all OOM by 1.24-1.25 GiB) **does not recur on JAX**. Exp 40 (seq=2048 b=2) proves this directly: same config that compile-OOM'd on torchax runs cleanly on JAX. The gap is torchax's `JittableModule` dedup during compile trace + HF-cache-class pytree inflation. Flax NNX's pytree is leaner.

2. **Dispatch overhead per call** — torchax adds a `torch_view / jax_view / call_jax` bridge per op. This is invisible at batch=1 (amortized over one large jitted step) but accumulates at the per-layer granularity: ~250 RMSNorm calls + ~42 attention + ~42 MLP × 2 = ~500 ops/step × ~2-5 µs bridge overhead = ~1-3 ms/step. At b=1 s=1024 (~135 ms step), that's <3 %. At b=3 s=1024 (~355 ms step), proportionally smaller. So this is a ceiling-raiser, not a primary win.

3. **Direct control of forward** — the JAX port owns `Gemma4TextModel.__call__` directly (Flax NNX). No HF `ALL_ATTENTION_FUNCTIONS` registration, no kwargs-forwarding layer. Makes scan-over-layers tractable (torchax exp 26 parked on 5 independent blockers; JAX port has none of them).

### What's still on the table for the JAX stack

| Experiment | Cost | Expected gain | Priority |
|---|---|---|---|
| Scan-over-layers (exp 43) | 300–500 LOC | +0 % TPS (compile-time +30× only) | High for iteration speed |
| Pallas RMSNorm custom_vjp retest (exp 44) | copy torchax kernel | −5 to −8 % (fusion-boundary likely same) | Low — repeat of torchax exp 33 |
| Persistent JAX compile cache (exp 45) | 1-line env var | 0 % TPS, 30× first-run cached | Trivial infra |
| tokamax fused `linear_softmax_cross_entropy_loss` (exp 46) | ~50 LOC integration | +1–3 % TPS, frees ~1.5 GiB | Medium |
| VMEM **shrink** to 64 MiB (symmetric to exp 41) | 1 env var | unknown (complement to exp 41) | Low |
| Quantization fp8/int8 for matmuls | out of scope (semantic change) | — | N/A |

Likely practical ceiling for this workload on v6e-4 is **≈24 % MFU** — another ~1 MFU point from here. Beyond that requires hardware change (v6e-8 / v5p-4) or structural model change.

## Cross-stack generalizable lessons

Lessons that hold on BOTH stacks:

1. **Splash attention is the single biggest compiler-independent win** — +2.7 % on torchax (exp 8), enables every subsequent gain via HBM savings. The win is in the softmax mechanism XLA can't express, not batched matmul kernels.
2. **bf16 CE is free on Gemma 4** because `final_logit_softcapping=30.0` bounds logits to bf16-safe range. (JAX port had this correct by construction; torchax needed exp 12 to remove an upcast.)
3. **Batch=3 is the sweet spot on v6e-4**. Both stacks converge here. Beyond that, memory pressure degrades per-token cost (torchax exp 7 at HBM 97.6 %, JAX exp 39 at 86.75 % still has some slowdown).
4. **Pallas kernels only win when XLA can't already fuse**. Splash wins because online-softmax is non-fusable; RMSNorm loses because XLA already fuses it with neighboring matmuls (torchax exp 33, expected to repeat on JAX).
5. **XLA default flags are already well-tuned for this workload**. Latency-hiding, overlap_compute_collective_tc, async_collective_fusion all flat in isolation on both stacks. The torchax bundle regression is combinatorial (fuse_all_gather + multiple_steps together).
6. **The ~1.25 GiB `CompileTimeHbmOom` margin on torchax is a scaffolding artifact, not a hardware limit** — the JAX port doesn't hit it, proving the margin is in the JittableModule/HF-cache pytree.
7. **2D mesh (tp=2) is a net loss at 4 chips** for this model (torchax exp 32 at 2.4× slower) — TP collectives on 2 chips/axis swamp the savings. Revisit at ≥8 chips/axis.
8. **ICI collective-permute `~12 %` is a hardware floor on v6e-4 fsdp=4**, not an XLA-flag lever (confirmed by exp 30 / 31 / 38 / 42 isolation).

## Process retrospective — what worked, what didn't

Running for ~42 experiments across two stacks, I learned several things about running this loop effectively. Writing these down so future sessions (or other agents picking this wiki up) can do better.

### What worked well

1. **Branch-per-experiment rollback discipline** — every rejected/parked experiment has its branch preserved. Made cross-referencing easy (e.g., exp 36 could be sanity-checked against exp 32's 2D-mesh code, exp 33's Pallas kernel). Implemented in `program.md` after being violated early (exp 1–16 were mashed into one big consolidation commit, losing granular history).

2. **Verdict-suffix naming** (`-accepted` / `-rejected` / `-potential`) — makes `ls` of the experiments folder self-summarizing. Took until exp 33+ to adopt, but has been pure upside since. Worth propagating to any future autoresearch wiki.

3. **Per-stack folder split** — introduced after exp 34 when the torchax/jax dichotomy became real. Immediately resolved "which OBSERVATIONS.md applies?" ambiguity. Recommend **starting any multi-stack program with `<stack>/experiments/` structure from day one**.

4. **Direct xprof browser URLs in Profile sections** — added at the user's request, took <1 hour to backfill across 34 writeups via a Python script. Now every experiment page is one click from the trace viewer. Should be a day-one protocol rule.

5. **Consistent TSV schema** (`exp_id | date | tps | mfu | step_ms | hbm | config | status | description`) — gitignored (binary-log-friendly) but carries the full session state in grep-friendly form. The `awk '/step  2/,/step 19/'` + `grep dt=` + mean idiom for computing steady-state TPS is used on every experiment — worth formalizing as a helper script in `torchax/` and `jax/`.

6. **Subagent delegation for structural changes** — the JAX port (~2000 LOC) and the exp 33 Pallas RMSNorm (~300 LOC) were both delegated to subagents and returned clean work. The pattern that worked: provide **(a)** the target file + existing reference impl, **(b)** the exact test criterion (numerical parity + training loss match), **(c)** the decision tree (accepted/rejected thresholds), **(d)** deliverables (writeup + TSV row + commit). Cheap-experiment iteration (VMEM bump, flag isolation) is better done directly — no benefit from the subagent overhead.

### What didn't work / things to change

1. **Subagents occasionally exited early** while a training run was still in progress (they scheduled a wakeup then hit an end-of-budget before the wake fired). Lost ~10 min on exp 38 doing the rerun. **Fix**: when delegating to a subagent for a multi-minute training run, tell it to use `run_in_background: true` + `Monitor` pattern that I use directly — don't let it sleep-then-check.

2. **Exp 1-16 were committed in bulk as one consolidation commit** (`71a45ae`) — lost per-experiment git-log granularity. Caused the "writeups for exp 9-25 missing" gap discovered at exp 26. **Fix**: commit each experiment individually from day one, even the ones you think are trivial. The branch model (added after the fact) enforces this now.

3. **RESULTS.tsv header was inconsistent** across stacks initially (torchax had slightly different columns). Normalizing is tedious. **Fix**: define the schema in `program.md`'s Fixed bindings table and don't let individual experiments drift from it.

4. **I wasted effort on 3+ "likely-null" experiments** (exp 38, 41, 42) before starting the retrospective. The pattern (6 consecutive flat/rejected experiments) was the ceiling signal — should have shifted to writing the analysis around exp 40. **Fix**: after 3 consecutive non-wins at the current best, switch to analysis/structural work, don't grind flag isolation.

5. **xprof MCP was often timed out** on GCS-backed logdirs (slow to list runs). The workaround (upload to GCS via `gsutil rsync`, grep local trace) works but is manual. **Fix**: automate "after training run finishes, upload profile to GCS" in `train.py` directly (both stacks' trainers). Adds 30 s wallclock but makes subsequent xprof queries instant.

6. **Ceiling analyses should be a first-class artifact, not an afterthought**. Both times I wrote one (torchax → this session, and now JAX) it turned out valuable for planning follow-ups. **Fix**: add an explicit protocol step in `program.md`: "when 3 consecutive experiments produce no new best, pause the loop and write a ceiling analysis. The analysis's 'Next hypotheses' list is the new ranked queue."

7. **Cross-references broke on folder moves** (the torchax/jax reorg broke ~35 relative links; fixing them was a 1-hour sed + Python job). **Fix**: SCHEMA.md's link convention already says "relative paths" — consider extending to "use shortest-possible paths, and document expected link depths so that future moves fail loudly instead of silently." Or: stop worrying about moves and use a link-checker in `LINT`.

8. **The Pallas kernel directory analysis** and **Wave-4 ingestion** happened interleaved with experiments, polluting both. **Fix**: in sessions that do both ingest and optimize, alternate in larger blocks (e.g., "ingest week" → "experiment week") rather than mixing.

### Suggested additions to `program.md` / `SCHEMA.md`

Concrete changes worth filing:

- **New protocol rule**: "After 3 consecutive experiments with `status ∈ {discard, inconclusive, parked}` since the last `keep` on the current best, pause and write a ceiling analysis before the next run."
- **New Fixed-bindings row**: "Steady-state TPS computation" — standardize on `mean of steps 2 through (steps - 1)` and the helper awk idiom.
- **New Fixed-bindings row**: "Profile auto-upload" — trainer should `gsutil rsync` after the trace closes so xprof MCP sees it.
- **New lint check**: "Every accepted experiment referenced by `main` commits must have a writeup whose filename ends in `-accepted.md`."
- **SCHEMA.md addition**: a `ceiling_analysis` page type (template: trajectory / what-worked / what-didn't / generalizable-lessons / next-hypotheses / cross-stack-compare).

## Full experiment table (both stacks)

See [2026-04-23-gemma4-v6e4-optimization-ceiling.md](2026-04-23-gemma4-v6e4-optimization-ceiling.md) for the exhaustive torchax table, and the per-experiment writeups under [`../experiments/gemma4_autoresearch_optimization/torchax/experiments/`](../experiments/gemma4_autoresearch_optimization/torchax/experiments/) and [`../experiments/gemma4_autoresearch_optimization/jax/experiments/`](../experiments/gemma4_autoresearch_optimization/jax/experiments/).

Abbreviated comparison — the two best-of-stack + baselines only:

| Config | TPS | MFU | Stack |
|---|---:|---:|---|
| baseline (XLA SDPA, b=1 s=1024) | 30,570 | 20.36 % | torchax |
| JAX port baseline (XLA SDPA, b=1 s=1024) | 30,285 | 20.17 % | jax |
| exp 25 (splash + b=3 + SEQ_MINOR + block=1024 + fused_bwd + bf16 CE + selective remat) | **33,372** | **22.22 %** | **torchax best** |
| exp 36 (splash + b=3 + bf16 CE + selective remat) | **34,614** | **23.05 %** | **jax best, +3.7 % over torchax** |

## See also

- [Torchax ceiling analysis](2026-04-23-gemma4-v6e4-optimization-ceiling.md)
- [Shared program protocol](../experiments/gemma4_autoresearch_optimization/program.md)
- [Torchax experiments](../experiments/gemma4_autoresearch_optimization/torchax/experiments/README.md)
- [JAX experiments](../experiments/gemma4_autoresearch_optimization/jax/experiments/README.md)

## Sources

- All experiments exp 34 – exp 42 under `wiki/experiments/gemma4_autoresearch_optimization/jax/experiments/`.
- Profiles at `raw/profiles/2026-04-23-gemma4-jax-exp{34..42}-*/` (gitignored, mirrored to `gs://tpu-pytorch-alekseyv-us-central2/autoresearch/`).
- All commits on branch `main`: `49d43e9` (port) → `d9db1bc` (exp 42).
