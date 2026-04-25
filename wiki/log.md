# Log

## [2026-04-25] run-experiment | MaxText Llama3.1-8B v6e-8 reference baseline (SUPPORTED, 409.4 TFLOP/s/device, 44.6% MFU)

**Op**: run-experiment.
**Pages created**: `wiki/experiments/llama3_8B_autoresearch_optimization/maxtext/README.md`; `wiki/experiments/llama3_8B_autoresearch_optimization/maxtext/experiments/README.md`; `wiki/experiments/llama3_8B_autoresearch_optimization/maxtext/experiments/2026-04-25-maxtext-llama3-1-8b-v6e8-baseline.md`; `raw/profiles/2026-04-25-maxtext-llama3-1-8b-v6e8-baseline/` (mirror of `gs://tpu-pytorch-alekseyv-us-central2/maxtext/v6e8-20260425-01-llama3-1-8b/ale-llama3-1-8-1-042516-y1s/`).
**Pages updated**: `wiki/experiments/llama3_8B_autoresearch_optimization/README.md` (added maxtext stack pointer with MFU number).
**Key result**: Wired the MaxText stack to the autoresearch v6e-8 GKE cluster `alekseyv-tpu-v6e8-spot-xpk` (which `program-gke.md` had marked TODO across all three GKE clusters) and ran the official `tpu-recipes-v0.1.4` Llama3.1-8B v6e-8 recipe (`llama3_1_8b_8192_no_collective_matmul`, fsdp=8 seq=8192 bs=3) end-to-end through xpk + `benchmark_runner xpk`. Steady-state median over steps 11–14 + 16–19: **409.42 TFLOP/s/device, 7,069.66 Tokens/s/device, 44.6 % MFU** (per-chip vs Trillium 918 TFLOP/s bf16 peak); aggregate 56.6k tok/s across 8 chips. Reproduces recipe README's published 413.4 / 7,138.9 within **−1.0 %** on both metrics — well inside run-to-run noise. Verdict: `supported`.
**Notes**: Infrastructure decisions worth carrying forward: (a) MaxText worktree at `tpu-recipes-v0.1.4` placed at `/mnt/disks/persist/maxtext-tpu-recipes-v0.1.4` per SCHEMA rule 1 (raw/code/maxtext stays on main); (b) local maxtext venv at `/mnt/disks/persist/venv-maxtext-v0.1.4` (uv + py3.12), `MAXTEXT_README`'s `uv pip install -e .[tpu] --resolution=lowest` path doesn't work as written (absl-py 0.1.0 build failure under py3.12; tensorboardx version conflict in jax-stable-stack-0.6.1 reqs file) — the working recipe is `uv pip install -e .[tpu] --prerelease=allow` then patch in `omegaconf grain ruamel.yaml jaxtyping aqtp ml-goodput-measurement google-cloud-aiplatform google-cloud-monitoring google-jetstream cloud-tpu-diagnostics`; (c) `maxtext_base_image` is **1.81 GiB** (much smaller than the torchtitan image, ~11 GiB) so docker build + push is fast (2–3 min total); (d) xpk needs to be cloned to `~/xpk` exactly (path required by `XPK_README.md` and benchmark_runner default `xpk_path`). HBM peak was 10.66 GiB / 31.25 GiB (34 %) — large headroom for `bs=4` and projection-offload ablations as next hypotheses. Loss trajectory monotone-decreasing (12.264 → 1.792); model is training. xprof URL: `http://localhost:8791/?run=2026-04-25-maxtext-llama3-1-8b-v6e8-baseline` (after `xprof --logdir=raw/profiles --port=8791`). New stack folder `wiki/experiments/llama3_8B_autoresearch_optimization/maxtext/` distinguishes "reference baseline (don't optimize, just reproduce)" runs from the optimization-target `torchax/` and `jax/` stacks; mirrors the gemma4 program's per-stack split.

## [2026-04-25] ingest-codebase | tpu-recipes (AI-Hypercomputer/tpu-recipes @ e284e361)

**Op**: ingest-codebase.
**Pages created**: `wiki/codebases/tpu-recipes.md`.
**Pages updated**: `wiki/index.md` (Codebases count 26→27; new entry under Wave 4 ecosystem); `README.md` (new bullet under "Reference trainers & inference engines"); `.gitmodules` (new submodule entry).
**Submodule added**: `raw/code/tpu-recipes` → `https://github.com/AI-Hypercomputer/tpu-recipes.git` @ commit `e284e3613721882ce3e15c533a76c691443ea60f`.
**Key result**: AI-Hypercomputer's curated per-(model, hardware, topology) reproduction recipes on Cloud TPU. Each recipe pins a MaxText tag (e.g. `tpu-recipes-v0.1.4`), a `jax-stable-stack` Docker image (e.g. `jax0.6.1-rev1`), the launch command, and the **MaxText `tuning_params` block** used to hit the published throughput target. Coverage: Trillium (v6e-32/64/128/256) + Ironwood (v7x 4×4×4 / 4×8×8) for Llama 3.1 (8B/70B/405B), Gemma 3-12B, Gemma 4-26B/31B, Mixtral 8×7B / 8×22B, DeepSeek 3-671B, Qwen 3-235B, GPT-OSS 120B, Wan 2.1-14B, GPT-3 175B; legacy v5p tier. Plus `microbenchmarks/` (matmul TFLOPS + HBM bandwidth on v6e-1) and `utils/profile_convert.py` (`.xplane.pb` → text).
**Notes**: Single most direct public source for "what flags Google's perf team picked for this combination." Closest external precedent for the gemma4 program here: the `Gemma3-12B-MaxText/v6e-*` recipes (family + scale) and the published `Llama3.1-70B-MaxText/v6e-32` `tuning_params` block (`remat_policy: custom`, `decoder_layer_input: offload`, `query_proj/key_proj/value_proj: offload`, `per_device_batch_size: 2`). Future hypotheses touching `remat_policy` or per-projection offload should cite the closest matching recipe as the prior. Microbenchmark harness usable to establish v6e-1 chip-level rooflines (matmul ~827 TFLOPS bf16 8192³ with `xla_tpu_scoped_vmem_limit_kib=65536`; HBM ~1359 GB/s 32 MiB copy) — useful when an op-profile roofline classification is in dispute. No subpages created (recipe `READMEs` are themselves the leaves; no perf-relevant subsystems warrant their own wiki page).

## [2026-04-24] analyze + run-experiment | jax-exp53: splash block-size sweep under new regime (REJECTED, flat) + new-regime ceiling analysis

**Op**: run-experiment + analyze.
**Pages created**: `wiki/experiments/gemma4_autoresearch_optimization/jax/experiments/2026-04-24-exp53-jax-splash-block-sweep-fp32master-rejected.md`; `wiki/analyses/2026-04-24-gemma4-jax-fp32master-seq8k-regime.md` (new-regime ceiling analysis); `raw/profiles/2026-04-24-gemma4-jax-exp53-splash-block512-seq2k-fp32master/` + `gs://tpu-pytorch-alekseyv-us-central2/autoresearch/2026-04-24-gemma4-jax-exp53-splash-block512-seq2k-fp32master/`.
**Pages updated**: `.../jax/experiments/RESULTS.tsv` (exp53 discard row); `.../jax/experiments/OBSERVATIONS.md` (exp 53 block + ceiling pointer).
**Key result**: **exp 53 REJECTED** — splash block-size sweep at seq=2048 b=1 fp32-master is flat. Block=2048 (full-tile) hits `CompileTimeScopedVmemOom` (32.14/32.00 MiB, 144 KiB over). Block=512 runs at 26,807 TPS, dead flat vs exp 52's default block=1024 (−0.0 %). Confirms the old-regime exp 48 plateau transfers to the new regime. **Ceiling analysis filed** covering seq=8192 infeasibility on v6e-4 (independent of AMP — legacy bf16 also OOMs), XLA non-monotonic compile-time peak, `nothing_saveable`/offload counter-intuitive regression, and the three-branch forward path (A: optimize at seq=2048; B: v6e-8 mesh; C: memory-saving code changes).
**Notes**: New durable heuristic for program.md: "Splash block size is flat across all measured Gemma 4 E4B shapes on v6e-4 with the fused_bwd kernel + SEQ_MINOR layout — don't open a new block-size experiment unless the shape or kernel changes materially." Still-open exp 52 follow-ups queued in the new-regime ceiling page: exp 54 (pure-AMP isolation at b=1 s=1024), exp 55 (scan_layers at new regime), exp 56 (2D mesh dp=2 tp=2), exp 57 (PLE embedding host-offload). xprof browser URL for exp 53: http://localhost:8791/?run=2026-04-24-gemma4-jax-exp53-splash-block512-seq2k-fp32master.

## [2026-04-24] run-experiment | jax-exp52: fp32-master + bf16-compute AMP new-regime baseline (seq=8192 OOMs on v6e-4; seq=2048 b=1 accepted at 26,807 TPS)

**Op**: run-experiment (regime change + new-baseline establishment).
**Pages created**: `wiki/experiments/gemma4_autoresearch_optimization/jax/experiments/2026-04-24-exp52-jax-fp32master-seq2k-accepted.md`; `raw/profiles/2026-04-24-gemma4-jax-exp52-baseline-seq2k-fp32master/` (on-disk xprof profile, gitignored) + `gs://tpu-pytorch-alekseyv-us-central2/autoresearch/2026-04-24-gemma4-jax-exp52-baseline-seq2k-fp32master/` (GCS mirror).
**Pages updated**: `.../jax/experiments/RESULTS.tsv` (exp52 keep row); `.../jax/experiments/OBSERVATIONS.md` (exp 52 block); `.../jax/experiments/README.md` (new "Current state" section split between old-regime and new-regime baselines); `.../jax/train.py` (new `--weights-dtype` / `--compute-dtype` CLI, seq_len default 2048 → 8192, `JAX_REMAT_POLICY` env var, reordered apply_sharding before load_hf_weights); `.../jax/model/modeling_gemma4.py` (all modules accept split weights_dtype / compute_dtype kwargs; Linear downcasts weight to compute_dtype at matmul; embed_tokens casts output to compute_dtype; layer_scalar casts to hidden dtype); `.../jax/model/weight_loader.py` (optional `shardings=` scatter-shards fp32 directly into sharded NNX params, avoiding a 10.5 GiB fp32 PLE on-device-0 init OOM); `.../jax/model/scan_layers.py` (_matmul_amp helper + layer_scalar-cast for AMP correctness); `.../torchax/train.py` (flag parity stubs; warns + falls back to --dtype on split); commits 517a689 + 176fd2c + flag-wiring squash.
**Key result**: **KEEP as new-regime baseline**. User asked for `seq_len=8192 b=1 fp32-master + bf16-compute` and the optimization loop to run there. seq=8192 b=1 fp32-master is **INFEASIBLE on v6e-4**: compile OOM at 35.18 GiB / 31.25 GiB per-chip HBM (exceeded by 3.93 GiB). Even legacy bf16-only at seq=8192 b=1 OOMs (36.16 GiB, exceeded 4.91 GiB). Intermediate seq_lens are WORSE than seq=8192 due to XLA scheduling non-monotonicity (seq=4096 = 39.58 GiB; seq=6144 = 49.66 GiB). Largest-feasible new-regime config: **seq=2048 b=1 fp32-master = 26,807 TPS, 305.6 ms/step**, loss descent 3.25 → 2.30 clean. −15.7 % vs exp 40 (bf16-only b=2 s=2048 at 31,809 TPS; b=2 is itself infeasible under fp32 master — 39.37 GiB OOM). Exp 36 (bf16 single-dtype, s=1024 b=3 at 34,614 TPS) remains the **old-regime** JAX best; exp 52 is the new-regime baseline.
**Notes**: AMP implementation is JAX-only; torchax gets CLI-parity stubs that warn + fall back to legacy `--dtype` (HF PyTorch takes a single torch_dtype; full torchax wiring is out of scope). Flag wiring: modules accept kwargs (weights_dtype for storage, compute_dtype for matmul); Linear does `x @ W.astype(compute_dtype).T` at call time — single fp32→bf16 cast per forward, XLA folds into the dot when it can. NO hand-casts beyond that needed (no need for `jax.lax.dot_general` with explicit dtype promotion). `nothing_saveable` and `offload_dot_with_no_batch_dims` remat policies **increase** compile-time peak HBM (39.66 and 38.17 GiB at seq=8192 respectively, vs default's 35.18 GiB) — XLA planner does not credit offload as HBM-freed; nothing_saveable serializes more live tensors to avoid recomputing. JAX_SCAN_LAYERS=1 saves 0.35 GiB at fp32-master seq=8192 and 3.7 GiB at bf16-legacy seq=8192, not enough to close the gap. Weight-loader refactor: `shardings={path: NamedSharding}` kwarg scatter-shards each tensor through a host numpy buffer → per-device HBM directly, so the fp32 PLE embedding (11 GiB) never materializes on device 0; needed because NNX inline init runs on device 0 by default. Apply_sharding moved before load_hf_weights so the shardings lookup is available. Init-in-bf16 workaround: even though weights_dtype=fp32, the random-init pass is in bf16 to fit device 0; the loader overwrites with fp32 sharded values; a `_fixup_dtype_meta` pass retargets per-module weights_dtype attributes for introspection. Follow-up queue documented in experiment page: (1) pure-AMP isolation at b=1 s=1024, (2) splash block sweep at seq=2048 fp32-master, (3) scan_layers at seq=2048 fp32-master, (4) 2D mesh dp=2,tp=2 at seq=8192 (K/V replication on tp=2 permitted since num_kv_heads=2), (5) PLE embedding host-offload (risky code change). xprof_mcp access is currently pointing at a different GCS logdir; direct URL for the uploaded profile: http://localhost:8791/?run=2026-04-24-gemma4-jax-exp52-baseline-seq2k-fp32master (resolves once server is re-pointed at the autoresearch bucket).

## [2026-04-24] run-experiment | jax-exp47: marin/levanter fused linear+softcap+CE Pallas kernel — REJECTED (−5.61 % TPS, custom-call + all-gather tax)

**Op**: run-experiment.
**Pages created**: `wiki/experiments/gemma4_autoresearch_optimization/jax/experiments/2026-04-24-exp47-jax-levanter-ce-rejected.md`; `wiki/experiments/gemma4_autoresearch_optimization/jax/model/kernels/__init__.py` and `.../kernels/fused_ce/__init__.py` (import shim); `wiki/experiments/gemma4_autoresearch_optimization/jax/tools/parity_levanter_ce.py` (correctness harness); `raw/profiles/2026-04-24-gemma4-jax-exp47-levanter-ce/` (xprof profile, gitignored).
**Pages updated**: `.../jax/experiments/RESULTS.tsv` (exp47 discard row), `.../jax/experiments/OBSERVATIONS.md` (exp 47 block), `.../jax/train.py` (`JAX_CE_IMPL=levanter` branch + shard_map wrapper + sharded levanter call), `.../jax/model/modeling_gemma4.py` (`Gemma4ForCausalLM.__call__(return_hidden=True)` kwarg + `lm_head_weight()` helper).
**Key result**: **REJECTED**. TPS 34,614 → **32,671 (−5.61 %)**; step time 355.0 → **376.1 ms (+5.95 %)**; parity **PASS** (|diff| 0.048 vs tol 0.05 in bf16); smoke step-4 loss 2.1875 → 2.1979 (+0.47 %, within 5 % bar). The softcap gap from exp 43 is closed — levanter's `fused_cross_entropy_loss_and_logsumexp_penalty` applies `sc * tanh(logits/sc)` inline on each VMEM logits-tile before the streaming softmax (`raw/code/marin/lib/levanter/src/levanter/kernels/pallas/fused_cross_entropy_loss/reference.py:11`), so `[B, S, V]` logits never materialize in HBM. Regression root cause is the same Pallas-custom-call tax as torchax exp 33 (Pallas RMSNorm, −36 ms/step): CE in exp 36 was <3 % of step time and XLA-fused with lm_head + softcap + log_softmax; replacing a tight 10-ms XLA fusion with a 3.5-ms Pallas kernel adds ~15 ms of custom-call boundary + the 1.31-GiB `w_hv` all-gather that the shard_map wrapper forces (lm_head weight is FSDP-sharded on V, kernel requires replicated `[H, V]`). Exp 36 remains JAX-stack best at 34,614 TPS.
**Notes**: Pre-run gates done per SCHEMA run-experiment op: parity harness before smoke, smoke before benchmark, profile captured during benchmark. Block sizes hand-picked (`b=1024, h=256, v=512`): default `(1024, 512, 1024)` overruns 32 MiB VMEM on v6e; `b_block` must be ≥1024 because per-shard flat batch B*S=3072 ≥ 1024 (kernel invariant in `pallas_tpu.py:227`); Gemma 4's shape (V=262144, H=2560) lands in no TPU-tuned bucket (`tuned_block_sizes.py` TPU buckets top out at V=131072). Durable artifacts: `JAX_CE_IMPL=levanter` gate + `jax/model/kernels/fused_ce/` import shim (levanter kernel importable without installing equinox/draccus/rigging via `sys.modules` stubs); `jax/tools/parity_levanter_ce.py` correctness harness; `Gemma4ForCausalLM.__call__(return_hidden=True)` + `lm_head_weight()` seam for future CE-replacement variants. Follow-ups: retry at seq=2048 b=2 (logits pressure higher, cost/benefit may flip); V-sharded kernel variant (removes the w_hv all-gather, kernel-edit effort M); `collective-permute-done` audit unchanged from exp 36's queue. Profile local only (`raw/profiles/2026-04-24-gemma4-jax-exp47-levanter-ce/`, 328 MiB) + GCS mirror at `gs://tpu-pytorch-alekseyv-us-central2/autoresearch/2026-04-24-gemma4-jax-exp47-levanter-ce/` (326.6 MiB uploaded). xprof browser access blocked in this account's logdir (same as other recent JAX-stack experiments). Promotes the "Pallas custom-call tax" heuristic to a general rule: swapping a Pallas kernel in for an XLA-fused op that is <5 % of step time tends to lose fusion more than it gains kernel speed.

## [2026-04-24] run-experiment | jax-exp43: tokamax.linear_softmax_cross_entropy_loss — INVALID (API-precondition failure, no softcap support)

**Op**: run-experiment.
**Pages created**: `wiki/experiments/gemma4_autoresearch_optimization/jax/experiments/2026-04-23-exp43-jax-tokamax-ce-rejected.md`.
**Pages updated**: `.../jax/experiments/RESULTS.tsv` (exp43 discard row), `.../jax/experiments/OBSERVATIONS.md` (exp 43 block), `.../jax/experiments/README.md` (Current state paragraph adds exp 43 line; exp 36 remains best).
**Key result**: **INVALID** — no run performed. tokamax `v0.0.12` public API `tokamax.linear_softmax_cross_entropy_loss(x, labels, weights, *, reduction, precision, implementation)` has **no** `logits_soft_cap` (or equivalent) kwarg (verified across `api.py`, `base.py`, `reference.py`, and the 690-line `pallas_mosaic_tpu_kernel.py`). Gemma 4's `final_logit_softcapping=30.0` is a non-linear `sc * tanh(logits / sc)` element-wise on the `[B, S, V]` logits — cannot be folded into `hidden` or `W` (algebraically impossible for non-linear post-matmul ops), and applying post-kernel requires materializing `[B, S, V]` which defeats the kernel's sole purpose. Skipping softcap violates program.md "What you CANNOT do". Zero lines of code merged. Exp 36 remains JAX-stack best at 34,614 TPS / 86.75 % HBM.
**Notes**: Decision-tree exit per task brief: "Softcap incompatible with tokamax API → can't apply cleanly → `-invalid`. Don't merge." The result **empirically confirms** the build-target already catalogued in [program.md § "Pallas kernels to BUILD" → "Fused final logit softcap + log-softmax + NLL"](experiments/gemma4_autoresearch_optimization/program.md) — a Gemma-aware kernel fork that applies `sc * tanh(logits_tile / sc)` inline on each VMEM block before `log_softmax` is the only correct path to the ~1.5 GiB HBM save on this model. Estimated effort M (extend mosaic_tpu_kernel.py with a softcap pre-op). Secondary API mismatches also noted: no `ignore_index`; `reduction="mean"` divides by `B*S` instead of `mask.sum()`; `x` wants `[B, H]` flat (easy reshape). These are ~15-line glue, non-blocking — softcap is the hard blocker. Note: torchax exp 27's prior tokamax integration attempt failed on **attention** (`dot_product_attention` / sliding-window masks) — exp 43 is the first tokamax-CE attempt on either stack, and fails at a different API seam.

## [2026-04-23] run-experiment | jax-exp37: bf16 CE env-var gate on top of exp 36 — POTENTIAL (flat, no-op-by-construction)

**Op**: run-experiment.
**Pages created**: `wiki/experiments/gemma4_autoresearch_optimization/jax/experiments/2026-04-23-exp37-jax-splash-b3-bf16ce-potential.md`.
**Pages updated**: `.../jax/experiments/RESULTS.tsv` (exp37 parked row), `.../jax/experiments/OBSERVATIONS.md` (exp 37 block), `.../jax/experiments/README.md` (Current state + queued exp list — exp 37 resolved, exp 38 now highest priority), `.../jax/train.py` (added `JAX_CE_DTYPE` gate).
**Key result**: **POTENTIAL (flat, within noise)**. TPS 34,614 → **34,629 (+0.04 %)**; step time 355.0 → 354.85 ms; peak HBM 27.11 → 27.45 GiB (heap unchanged at 10.67 GiB, stack +0.34 GiB). Loss matches exp 36 within 0.3 %. Exp 36 remains JAX-stack best at 34,614 TPS.
**Notes**: The torchax exp 12 analog (+3.0 % TPS / −1.5 GiB HBM) replicated as a **no-op** on the JAX port: the native-JAX port never had the `flat_logits.to(fp32)` upcast that torchax inherited — `forward_loss` in `jax/train.py` has been bf16 log_softmax by construction since the exp-34 port. The durable artifact from exp 37 is the `JAX_CE_DTYPE={bf16,fp32}` env-var gate, which enables controlled A/B on future refactors (e.g., if exp 39 Pallas RMSNorm changes upstream dtype flow). HLO graph bit-identical to exp 36 aside from one dropped implicit-cast in the CE reduction. Profile uploaded to `gs://tpu-pytorch-alekseyv-us-central2/autoresearch/2026-04-23-gemma4-jax-exp37-splash-b3-bf16ce/`; xprof at http://localhost:8791/?run=2026-04-23-gemma4-jax-exp37-splash-b3-bf16ce. Next priority: exp 38 (collective-permute-done sharding audit, 12.1 % of step time).

## [2026-04-23] run-experiment | jax-exp36: splash + batch=3 — new JAX-stack best, BEATS torchax session-best (+3.7 %)

**Op**: run-experiment.
**Pages created**: `wiki/experiments/gemma4_autoresearch_optimization/jax/experiments/2026-04-23-exp36-jax-splash-batch3-accepted.md`.
**Pages updated**: `.../jax/experiments/RESULTS.tsv` (exp36 keep row), `.../jax/experiments/OBSERVATIONS.md` (exp 36 block), `.../jax/experiments/README.md` (Current state + queued exp list).
**Key result**: **SUPPORTED / accepted**. TPS 30,386 → **34,614 (+13.9 %)**; step time 134.8 → 355.0 ms (sub-linear at 3× batch — per-token cost −13.9 %); peak HBM 16.43 → **27.11 GiB / 86.75 %** (4.14 GiB free). The JAX stack at b=3 splash **surpasses torchax session-best** (exp 25: 33,372 TPS) by **+3.7 %** without bf16 CE. HLO-op diff validates the per-call-amortization mechanism: splash `custom fusion` near-constant (169 → 175 ms, ×1.03) while matmul/loop fusions grew 2.75–3.81×.
**Notes**: No code change — only `--batch_size 1` → `--batch_size 3` and `JAX_ATTENTION_IMPL=splash` env var unchanged from exp 35. Loss descends cleanly 3.81 → 1.84 (b=3 batch content differs from b=1, not bit-comparable; trajectory healthy). New bottleneck surfaces at b=3: `loop fusion` 28.1 % (RMSNorm Pallas kernel now worth building), `collective-permute-done` 12.2 % (SPMD re-shard audit). Next: exp 37 (bf16 CE), exp 38 (sharding audit), exp 39 (RMSNorm kernel). Profile uploaded to `gs://tpu-pytorch-alekseyv-us-central2/autoresearch/2026-04-23-gemma4-jax-exp36-splash-batch3/`; xprof browser at http://localhost:8791/?run=2026-04-23-gemma4-jax-exp36-splash-batch3.

## [2026-04-23] ingest-batch + lint | Wave 4 follow-up — 15 concept stubs, 5 deferred codebases, 11 scaling-book chapters, lint pass

**Op**: ingest-source (11 scaling-book chapters) + ingest-codebase (5 deferred repos) + manual (15 concept stubs) + lint.

**Pages created (31)**:

- **15 concepts** (per the 2026-04-23 Pallas kernel directory's "Reusable Pallas-authoring patterns" + hardware-fact list):
  - 11 Pallas-pattern stubs: `online-softmax-with-logit-sink`, `in-kernel-dropout`, `two-level-chunk-recomputation`, `grouped-program-ids-for-l2`, `dma-overhead-heuristic`, `multi-shard-sequence-parallel-correction`, `block-sparse-offset-masks`, `custom-splash-masks`, `manual-mlir-dialect-pallas`, `pallas-on-triton-fused-gemm-activation-gemm`, `nvidia-weight-tile-bytes-limit`.
  - 3 autotune-harness stubs (from marin/levanter harness): `jaxpr-hash-cache-keys`, `compile-time-aware-autotune-filtering`, `vmem-oom-fallthrough`.
  - 1 hardware-fact concept: `vmem-budget` (v4 32 / v5e 32 / v5p 95 / **v6e 96** / **v7 48** MiB — the v6e→v7 halving).
- **5 codebases** (deferred from Wave 4's lower-priority tier):
  - `codebases/graphcast.md` (commit `08cf736`) — DeepMind weather model; splash wrapper + `WeatherMeshMask`.
  - `codebases/simply.md` (`f40b81e`) — DeepMind serving framework; DMA-overhead-bytes autotune heuristic.
  - `codebases/jaxite.md` (`e4a3351`) — Google FHE library; only non-ML Pallas TPU ref.
  - `codebases/qwix.md` (`b966dc4`) — Google quantization framework; AQT successor.
  - `codebases/aqt.md` (`9d1667e`) — deprecated; superseded by qwix.
- **11 scaling-book source pages** (one per content chapter, per-chapter summaries from a parallel Explore agent; each page ~150 lines with key claims, key data points, techniques referenced, gaps & caveats, connections). Book dated 2025-02-04; **v6e mentioned briefly, v7 absent** — explicit gaps flagged on every chapter page.

**Submodules added (5)**: `raw/code/{graphcast, simply, jaxite, qwix, aqt}`.

**Pages updated**:
- `README.md` — 5 new "Ingested codebases" entries.
- `wiki/index.md` — Codebases 21→26 (new "Wave 4 follow-up" sub-section), Sources 34→45 (new "Scaling-book chapters (11)" sub-section), Concepts 81→96 (three new sub-sections: Pallas-authoring patterns, Autotune-harness patterns, Hardware facts). Header page count 146→178.

**Key result**: Wave 4 follow-up work complete — concept-pattern stubs filed, deferred repos ingested, scaling-book finally out of "Wave 3 pending" limbo. Lint pass found and fixed 12 broken markdown links (all pre-existing in the gemma4 experiments tree — renamed experiment files with `-accepted` / `-rejected` / `-potential` suffixes whose cross-references hadn't been updated). Zero broken links remain.

**Notes**:

### Lint findings and actions taken
- **12 broken `.md` links** — all fixed. Pattern: gemma4 experiment files were renamed with outcome suffixes (`-accepted` / `-rejected` / `-potential`) after initial cross-references were written. Bulk-fixed via `sed` across five referring files. Also fixed one `../concepts/` → `../../concepts/` relative-path error in `OBSERVATIONS.md`, and one reference to a non-existent `codebases/transformers.md` (replaced with explicit upstream reference to HF transformers repo).
- **10 orphan pages** detected — all are pages created in this turn; all now indexed, no longer orphans.
- **Zero submodule-commit drift** — every codebase page's pinned commit matches its submodule checkout.
- **Zero `[!warning]` contradictions** in the wiki.
- **Hypotheses-open-14+-days lint** skipped — the gemma4 program doesn't use separate hypothesis files (it tracks them inline in the program README).
- **Experiments-without-profile-artifacts lint** skipped for this turn (not directly related to the new work).

### Scaling-book ingestion discipline
- Delegated per-chapter reading to an Explore subagent using a structured-summary schema; agent returned a 7,000-word consolidated summary that seeded the 11 source pages without requiring each chapter's full body in the main-agent context.
- **v6e (Trillium) mentioned briefly** in Ch 2; **v7 (Ironwood) absent throughout** — every chapter page flags this as a "Gaps & caveats" note because the wiki's primary gemma4 program runs on v6e-4, meaning the book's constants are already one generation behind.
- Chapter 4 KV-cache formula (`2 × S × L × K × H` bytes/sequence) and Chapter 7 critical-batch thresholds (B > 120 int8, B > 240 bf16) are the most directly-cite-able data points from the book.
- Chapter 9 "matmul roofline matched to 0.4%" worked example is now documented as an upper-bound datapoint for any "is my profile-vs-roofline gap significant?" question.

### Five deferred codebases — quick notes
- **graphcast**: non-LLM splash example; `WeatherMeshMask` is the canonical "custom splash mask" reference cited on the concept stub.
- **simply**: the DMA-overhead-bytes autotune heuristic (~0.5 MiB virtual bytes per DMA) comes from here; already concept-paged.
- **jaxite**: only non-ML Pallas TPU reference in the wiki; bf16-byte-split → u32 reassembly pattern.
- **qwix**: Google's **current** quantization framework; `QArray`-aware `pallas_call` wrapper is the right layer for future quant hypotheses.
- **aqt**: predecessor; deprecated. Do not target for new work.

### Remaining deferred / out-of-scope
- `JetStream` (AI-Hypercomputer) — **archiving 2026-02-01**; deliberate skip.
- `pytorch/xla` — consumer of tpu-inference via torch_xla custom ops; would be a framework page, not a Pallas source.
- `labyrinth-ssr` / `Essential-AI/maxtext-external` — vendor-only snapshots with no novel content.
- `pytorch/pytorch` — Inductor Pallas codegen is a compilation target, not a kernel library; deliberate skip.
- `google-deepmind/gemma` — not enumerated by the directory analysis; remains a gap.
- NVIDIA Mosaic-GPU kernels outside `jax-ml/jax` / tokamax / axlearn — not exhaustively surveyed; gap noted.

## [2026-04-23] ingest-codebase (batch) | Wave 4 — 11 Pallas-kernel repos

**Op**: ingest-codebase (batch of 11, driven by the [2026-04-23 Pallas kernel directory analysis](analyses/2026-04-23-pallas-kernel-directory.md)'s "Recommended next ingestion wave").

**Submodules added (11)**: `raw/code/{axlearn, tpu-inference, maxtext, maxdiffusion, ringattention, alphafold3, recurrentgemma, ejkernel, EasyDeL, sglang-jax, marin}`. **alphafold3 pinned to tag `v3.0.1`** (commit `231efc9`) because the ingested `gated_linear_unit/` directory was removed from `main` after that tag.

**Pages created (11 codebases)**:
- `wiki/codebases/axlearn.md` — novel SSM Pallas (Mamba1 / Mamba2-SSD / RAttention), splash extensions.
- `wiki/codebases/tpu-inference.md` — broadest novel TPU Pallas surface; crown-jewel tuning tables.
- `wiki/codebases/ringattention.md` — canonical Liu-2023 paper companion; unidirectional, no zig-zag.
- `wiki/codebases/maxtext.md` — closest public analogue of this wiki's gemma4 program.
- `wiki/codebases/maxdiffusion.md` — only repo wiring ring-attention as first-class splash-integrated kernel.
- `wiki/codebases/alphafold3.md` — only public production-grade Pallas fused GLU (GPU via Triton-on-Pallas).
- `wiki/codebases/recurrentgemma.md` — canonical LRU scan; ancestor of axlearn Mamba.
- `wiki/codebases/ejkernel.md` — broadest community TPU Pallas surface (17 kernels).
- `wiki/codebases/EasyDeL.md` — ejkernel consumer; operations-registry wrapper.
- `wiki/codebases/sglang-jax.md` — novel EAGLE speculative-decoding tree kernels + largest tuning table.
- `wiki/codebases/marin.md` — **deployment-time autotune harness** — the pattern this wiki should emulate.

**Pages updated**:
- `wiki/sources/2025-ultrascale-playbook.md` — Gaps & caveats: **partial retractions** on hypothesis candidates #2, #3, #4 based on the directory findings (ring-attention has three public refs; zig-zag still absent; RMSNorm/LayerNorm correctly unimplemented per exp 33; fused GLU reference exists at alphafold3 v3.0.1).
- `wiki/codebases/tokamax.md` — ring-attention section updated to name the three public reference impls (maxdiffusion splash-integrated, haoliuhl canonical, ejkernel splash wrapper) and reiterate that zig-zag remains open.
- `README.md` — added 11 new "Ingested codebases" entries; removed duplicate jax entry.
- `wiki/index.md` — Codebases 10→21; split into "Wave 4" (the 11 new) and "Wave 1–3" (existing). Header count 135→146.

**Key result**: Wave 4 ingestion complete — scope discipline held. The 11 new pages are shorter than average (50–180 lines each) because they intentionally defer per-kernel detail to the [pallas-kernel directory subpages](analyses/pallas-kernel-directory/) rather than re-transcribing. Every new codebase page links forward to the authoritative subpage §1–§6 row. Two partial retractions and one reaffirmation landed on the ultrascale playbook source page.

**Notes**:
- **Novelty anchors** (kernels / patterns that exist only in the newly-ingested repos):
  - axlearn: Mamba1 (`mamba_kernels.py`), Mamba2/SSD (`ssd_kernels.py`), RAttention linear attention (`linear_attention_kernels.py`), splash-with-dropout + logit-sink.
  - tpu-inference: RPA v3 (+ hd64), MLA v1/v2, fused_moe v1, quantized_matmul blockwise, all_gather_matmul, GDN + triangle_solver, SparseCore gather/scatter, structured_sparse_matmul v1.
  - alphafold3 (v3.0.1): Pallas fused GLU (GPU, `mosaic_gpu` via Triton).
  - recurrentgemma: LRU Pallas scan with complex accumulators + multi-shard correction.
  - ringattention: canonical Pallas TPU ring-attention (paper companion).
  - sglang-jax: EAGLE tree-speculative-sampling kernels + ~2,000+ RPA tuning entries (largest table surveyed).
  - marin: deployment-time kernel-agnostic autotune harness with six distinguishing properties over tokamax's write-time tuner.
- **Crown-jewel autotune patterns** (from marin's harness; all concept-candidate behaviors): compile-time-aware candidate filtering (threshold = 0.20 s), VMEM-OOM-aware fallthrough, sharding-preserving benchmark lowering, off-thread compile for mesh-bound contexts, jaxpr-hashed cache keys, GCS-aware persistent cache (shares bucket with PJRT compile cache).
- **VMEM budgets baked into kernels** (concept-level facts from tpu-inference quantized_matmul): v6 = 96 MiB, v7 = 48 MiB. RPA default = 100 MB. update_kv_cache = 64 MB. Worth elevating to concept pages.
- **NVIDIA weight-tile limit** (marin fused CE): `_NVIDIA_WEIGHT_TILE_BYTES_LIMIT = 101_376` — same on GB10 and H100. Concept fact.
- **DMA-overhead-equivalent** (simply; not ingested but referenced): ~0.5 MiB virtual bytes, assumed constant across TPU generations.
- **alphafold3 v3.0.1 pin discipline**: kernels removed from `main` after that tag. Every wiki link must include `@v3.0.1` or use the local submodule tree.
- **Deferred** (per Wave 4 proposal lower-priority tier): jaxite (niche FHE non-ML), graphcast (wrapper), simply (wrapper), qwix/aqt (quant framework), JetStream (archiving 2026-02-01), pytorch/pytorch Inductor, labyrinth-ssr / Essential-AI (vendor-only), pytorch/xla (consumer of tpu-inference via torch_xla custom ops).
- **Concept-page stubs not yet filed**: 13 reusable patterns surfaced by the directory (online-softmax-with-logit-sink, in-kernel-dropout, two-level-chunk-recomputation, grouped-program-ids-for-L2, DMA-overhead-heuristic, multi-shard-sequence-parallel-correction, block-sparse-offset-masks, jaxpr-hash-cache-keys, compile-time-aware-filtering, VMEM-OOM-fallthrough, manual-MLIR-dialect-Pallas, Pallas-on-Triton-fused-GEMM-activation-GEMM, custom-splash-masks). Candidate for a subsequent pass; not blocking.
- **Ingestion-page verbosity note**: each new page deliberately short — the per-kernel detail is authoritative in the [pallas-kernel directory subpages](analyses/pallas-kernel-directory/) (written by six parallel research agents), so codebase pages link forward rather than re-transcribe. Following SCHEMA's "One entity/concept/model per page. Split when a page exceeds ~500 lines" rule + avoiding duplication.

## [2026-04-23] analyze | Pallas kernel directory — ~200 kernels across ~30 repos

**Op**: analyze (directory/catalog; six parallel web-research agents).

**Pages created**:
- `wiki/analyses/2026-04-23-pallas-kernel-directory.md` — main directory page: cross-cutting functional-category tables (attention/paged/ring/MoE/norm/GLU/matmul/collectives/SSM/CE-loss), stability distribution, autotune/tuning-table inventory (crown jewels), reusable patterns (13 concept-page candidates), confirmed gaps, direct impact on open hypothesis candidates, recommended Wave-4 ingestion order.
- `wiki/analyses/pallas-kernel-directory/01-upstream-jax-tokamax.md` — ~55 kernels (jax-ml/jax + openxla/tokamax).
- `wiki/analyses/pallas-kernel-directory/02-ai-hypercomputer.md` — ~17 kernels (maxtext + maxdiffusion + JetStream). JetStream archives 2026-02-01.
- `wiki/analyses/pallas-kernel-directory/03-inference-engines.md` — ~33 kernels (vllm-project/tpu-inference + sglang-jax + aphrodite).
- `wiki/analyses/pallas-kernel-directory/04-research-labs.md` — ~18 kernels (apple/axlearn + google-deepmind/{recurrentgemma, simply, graphcast, alphafold3@v3.0.1}).
- `wiki/analyses/pallas-kernel-directory/05-frameworks-quant.md` — ~18 kernels (tunix, qwix, aqt, jaxite, paxml/praxis, pytorch/xla, marin/levanter, pytorch/pytorch Inductor).
- `wiki/analyses/pallas-kernel-directory/06-community-research.md` — ~50 rows (ejkernel, EasyDeL, ringattention, flashback, gla-jax, sqtian, maxtext-external, tpu-research, recompute_dont_restore + small repos).

**Pages updated**:
- `wiki/index.md` — Analyses 2 → 3 + 6 subpages under a nested list; page count 128 → 135.

**Key result**: **~200 Pallas kernels catalogued across ~30 repos**, each with source-path URL, backend, stability, performance claims (verbatim when they exist), use case, and callers. `mosaic_tpu` backend dominates (~60%). Every production-grade TPU kernel is either published upstream in jax-ml/jax / openxla/tokamax or vendored from them. Novel community kernels concentrated in six repos: **vllm-project/tpu-inference** (RPA v2/v3, MLA v1/v2, fused_moe v1, quantized_matmul blockwise, all_gather_matmul, fused_gdn, sparse_core, batched_rpa), **apple/axlearn** (Mamba1, Mamba2/SSD, RAttention linear-attention — the only public TPU-Pallas SSM kernels publicly), **AI-Hypercomputer/maxdiffusion** (ring_attention integrated with splash), **google-deepmind/alphafold3@v3.0.1** (production Pallas fused-GLU), **google-deepmind/recurrentgemma** (canonical LRU scan), **erfanzar/ejkernel** (broadest community TPU surface — 17 kernels).

**Crown-jewel tuning artifacts**: sglang-jax's ~2,000+ entry RPA `tuned_block_sizes.py`; tpu-inference's 1,200+ RPA v2 + 600+ quantized_matmul + 47 megablox + 28 fused_moe v1 tables; marin/levanter's **kernel-agnostic deployment-time autotune harness** (jaxpr-hashed keys, compile-cost-aware candidate filtering at 0.20s threshold, VMEM-OOM fallthrough, GCS-persistent cache colocated with PJRT). VMEM budgets baked into kernels: quantized_matmul v6=96 MiB, v7=48 MiB; RPA default=100 MiB; update_kv_cache=64 MiB. DMA-overhead-equivalent from simply: ~0.5 MiB virtual bytes across TPU generations.

**Notes**:
- **Partial retractions and closures of open hypothesis candidates on [sources/2025-ultrascale-playbook.md](sources/2025-ultrascale-playbook.md)**:
  - Candidate #2 ("wire tokamax `ring_attention_kernel` through `dot_product_attention`") — three public reference impls confirmed: maxdiffusion splash-integrated, haoliuhl canonical standalone, ejkernel splash wrapper. Hypothesis reduced from "open research" to "port + adapt" with three patterns to compare.
  - Candidate #3 ("Zig-Zag Ring Attention on TPU — no implementation found") — **retraction stands.** Ring Attention Pallas exists publicly (§6.4), but Zig-Zag causal-balance variant is **confirmed absent from every repo surveyed**, including the canonical haoliuhl repo (straight unidirectional ring, `below_or_on_diag` check only — not load-balanced).
  - Candidate #4 ("TPU-native Pallas kernels for gated_linear_unit and layer_norm in tokamax") — **partially retracted.** RMSNorm/LayerNorm absence from maxtext + tpu-inference + axlearn + upstream is external evidence that XLA-fusion is sufficient — **consistent with Gemma 4 exp 33's −8.1% empirical result.** Don't build RMSNorm. Fused GLU: AlphaFold3 v3.0.1 provides a production Pallas fused-GLU reference (GPU); porting to Mosaic-TPU is feasible but needs HLO-level validation that XLA isn't already fusing.
- **External validation of Gemma 4 exp 33 lesson** ("Pallas loses when XLA already fuses"): maxtext + tpu-inference + axlearn hand-write megablox, ragged-paged-attention, gather-reduce-sc, fused_moe — but **do not** hand-write RMSNorm, SwiGLU, softmax, or elementwise activations. That absence is data.
- **AlphaFold3 Pallas kernels live only on tag `v3.0.1`** — removed from `main`. Pin the tag in every URL.
- **jondeaton/ring-attention-jax-pallas** — **404 confirmed**, repo does not exist. Dropped from the catalog. User has no matching public repo per `gh api users/jondeaton/repos`.
- **Backend mislabels in community repos**: flashback, gla-jax, mamba2-jax-pallas all advertise Pallas but are Triton/Mosaic-GPU only. Only haoliuhl, ejkernel, sqtian, recompute_dont_restore, labyrinth-ssr (vendored), Essential-AI (vendored), and rdyro/moe_in_jax are actually TPU Pallas. Catalog flags each.
- **New concept-page candidates** (13 surfaced): online-softmax-with-logit-sink, in-kernel-dropout, two-level-chunk-recomputation, grouped-program-ids-for-L2, DMA-overhead-heuristic, multi-shard-sequence-parallel-correction, block-sparse-offset-masks, jaxpr-hash-cache-keys, compile-time-aware-filtering, VMEM-OOM-fallthrough, manual-MLIR-dialect-Pallas, Pallas-on-Triton-fused-GEMM-activation-GEMM, custom-splash-masks. Not filed as stubs yet — listed on the main directory page under "Reusable Pallas-authoring patterns".
- **Recommended Wave 4 ingestion order** (listed on the directory page): apple/axlearn (narrow kernel subdirs), vllm-project/tpu-inference, AI-Hypercomputer/maxtext, AI-Hypercomputer/maxdiffusion (narrow: splash_attention/ only), haoliuhl/ringattention, google-deepmind/alphafold3 @ v3.0.1 (narrow: gated_linear_unit/ only), google-deepmind/recurrentgemma, erfanzar/ejkernel + EasyDeL (paired), sgl-project/sglang-jax (narrow: speculative-decoding kernels), marin-community/marin (narrow: levanter/kernels/pallas/ + autotune harness).
- **Methodology**: six general-purpose subagents in parallel, each with identical per-kernel row schema; verified top candidates via GitHub API + WebFetch. Durations ~4–6 minutes each; total wall clock ~6 minutes for research + ~10 minutes for consolidation.
- **Gaps flagged**: NVIDIA Mosaic-GPU kernel catalog for GPU-side parity (enumerated in jax-ml/jax but not exhaustively surveyed in downstream GPU-Pallas repos); `google-deepmind/gemma` not checked; internal Google Gemini trees private; Anthropic/xAI/Cohere/Character private. Ragged-Paged-Attention arXiv ID appeared as `2604.15464` in one search result (likely future-dated scrape glitch) — verify before citing.

## [2026-04-23] ingest-codebase | jax (jax-ml/jax)

**Op**: ingest-codebase.
**Pages created**:
- `wiki/codebases/jax.md` — codebase parent page for `jax-ml/jax` at commit `feb5ba0585` (HEAD on 2026-04-23; bleeding-edge pin). Scoped to perf-relevant surfaces only — the repo is far too large for exhaustive ingestion.

**Pages updated**:
- `.gitmodules` — added `raw/code/jax` submodule.
- `README.md` — added jax to "Ingested codebases" list at the top.
- `wiki/index.md` — Codebases 9→10 (jax inserted at top); header count 127→128.

**Key result**: The ground-truth JAX repo is now ingested. The codebase page focuses on the four perf-relevant surface buckets — transformations & compilation, parallelism & sharding, kernels & lowering, profiling & analysis — and indexes 12 concrete performance-relevant levers with file-path anchors. The **first-party reference TPU Pallas kernel tree** at `jax.experimental.pallas.ops.tpu.*` (splash_attention, paged_attention, ragged_paged_attention, megablox, flash_attention, matmul, all_gather, threefry) is now first-class in the wiki — previously referenced only transitively through tokamax's mirror and the 2026-04-23 pallas-kernel-source-survey analysis.

**Notes**:
- **Commit `feb5ba0585` is `HEAD` on 2026-04-23** — the most-recent possible pin when ingesting. `git submodule update --remote raw/code/jax` to bump.
- **`jax.experimental.roofline` surfaced as a first-party alternative to `pallas-forge.roofline_chart`**: works on any JAX function, not just Pallas kernels; `roofline`, `register_roofline`, `roofline_and_grad` are the public API. The [pallas-forge](codebases/pallas-forge.md) page's `roofline_chart` is duplicated functionality; the codebase page flags this.
- **`jax.experimental.compilation_cache`, `jax.experimental.layout`, `jax.lax.scan` are all named in the gemma4 ceiling analysis as remaining levers.** The jax page now gives them concrete file-path anchors. The compilation cache is "infrastructure only, no TPS" per the analysis; layout is the backing API for the SEQ_MINOR choice in exp 24; scan-over-layers Option B is one of three remaining viable paths.
- **Splash-attention upstream-vs-mirror reconciliation**: `jax/experimental/pallas/ops/tpu/splash_attention/` in this repo is the authoritative upstream for both the wiki's splash-attention concept page and the `tokamax` `_src/ops/experimental/tpu/splash_attention/` copy. Future splash-tuning hypotheses can target either entry point; the jax codebase page now says so.
- **MegaBlox (`ops/tpu/megablox/gmm.py`) is first-party for MoE grouped matmul.** The 2026-04-23 pallas-kernel-source-survey already surfaced this; the jax codebase page gives it the in-tree citation.
- **Scope discipline**: page deliberately does not enumerate every jax module — numpy, scipy, nn, random, lax primitives other than `scan`, the jax2tf / jax2onnx paths, export/AOT surfaces beyond `serialize_executable`, the CUDA/Triton GPU Pallas path beyond a pointer, and the docs/tests/benchmarks directories are all explicitly out of scope. The page's job is to index the perf-relevant knobs, not to teach JAX.
- **Two surfaces underused by this wiki so far, now documented**: `jax.experimental.source_mapper` (HLO↔Python back-reference — complements the xprof graph viewer) and `jax.experimental.custom_partitioning` (SPMD escape hatch — the mechanism production libraries use to route around partitioner gaps in ragged-paged-attention / MoE kernels per the pallas-kernel survey).
- **No hypotheses / concept stubs filed.** Concept pages for most of the JAX surfaces ([pallas-kernel](concepts/pallas-kernel.md), [scan-over-layers](concepts/scan-over-layers.md), [sharding](concepts/sharding.md), [rematerialization](concepts/rematerialization.md), [jax-trace](concepts/jax-trace.md), etc.) already exist and link both ways.

## [2026-04-23] analyze | Public Pallas kernel source survey

**Op**: analyze (web-research survey).
**Pages created**:
- `wiki/analyses/2026-04-23-pallas-kernel-source-survey.md` — categorized inventory of every public source of JAX Pallas kernel code (Tier 1 production libraries → Tier 7 marginal). Agent-delegated WebSearch + WebFetch + GitHub API; verified top candidates.

**Pages updated**:
- `wiki/index.md` — Analyses 1 → 2; header page count 126 → 127.

**Key result**: Five production-grade ingest candidates identified: **AI-Hypercomputer/maxtext** (direct trainer analogue), **vllm-project/tpu-inference** (broadest novel kernel surface: ragged-paged-attention v2/v3, MLA, gdn, sparse-core, structured-sparse-matmul), **AI-Hypercomputer/maxdiffusion** (ring-attention reference), **apple/axlearn** (Mamba/SSD + rattention), **sgl-project/sglang-jax** (simple-gla, spec-decoding).

**Notes**:
- **Partial retraction** on two open hypothesis candidates from [sources/2025-ultrascale-playbook.md](sources/2025-ultrascale-playbook.md):
  - Candidate #2 ("wire tokamax `ring_attention_kernel` through `dot_product_attention`") — public reference impl exists at `AI-Hypercomputer/maxdiffusion/src/maxdiffusion/kernels/splash_attention/ring_attention_kernel.py`. Reduces the hypothesis from "open research" to "port + adapt".
  - Candidate #3 ("Zig-Zag Ring Attention on TPU — no implementation found") — Ring Attention Pallas exists publicly in `haoliuhl/ringattention` (770⭐, canonical Liu et al. impl). Whether the specific Zig-Zag causal-balance variant is implemented remains unverified — needs code-level read.
- **Cross-reference with gemma4 exp 33 lesson** ("Pallas loses when XLA already fuses"): external evidence supports this. maxtext and tpu-inference hand-write megablox, ragged-paged-attention, gather-reduce-sc — but **do not** hand-write RMSNorm or SwiGLU. That absence is data.
- **AlphaFold3 Pallas GLU** exists at `google-deepmind/alphafold3` **tag v3.0.1** — removed from `main`. Pin the tag before referencing.
- **New category surfaced: kernel-optimization agents** — `ucb-bar/autocomp`, `primatrix/Glaucis` (evolutionary Pallas search), `aryatschand/JAXBench` (LLM-authored kernel benchmark). Direct analogues to this wiki's autoresearch mission; worth shallow reads for search-procedure priors.
- **Scope discipline**: did not file individual hypothesis pages or update the existing playbook hypothesis candidates; the analysis page cross-references them inline and leaves promotion to human adjudication.
- **Gaps flagged in the analysis** for a follow-up sweep: NVIDIA Mosaic-GPU ops inside `jax-ml/jax`, GoogleCloudPlatform sample repos, `google-deepmind/gemma`, private-org repos (Anthropic, xAI, Cohere, Character), Ragged-Paged-Attention arXiv ID verification, `jondeaton/ring-attention-jax-pallas` 404 on verification.

## [2026-04-23] ingest-codebase + ingest-source | pallas-forge + Karpathy LLM-wiki backfill

**Op**: ingest-codebase (pallas-forge) + ingest-source (Karpathy LLM-wiki).
**Pages created**:
- `wiki/codebases/pallas-forge.md` — codebase parent page for `linhkid/pallas-forge` at commit `090510b7`. Already used (and found wanting) in gemma4 exp 20; the page makes that context first-class and captures the methodological lessons (isolated-microbench vs in-graph, no-custom_vjp blocker, v5e-only numbers, canonical 3D-grid matmul template).
- `wiki/sources/2026-karpathy-llm-wiki.md` — source page for Karpathy's "LLM Wiki" idea file, the methodological ancestor of this wiki's `SCHEMA.md`. Already locally saved as `raw/sources/2026-karpathy-llm-wiki.md`; backfilling the wiki-side page makes the design lineage explicit.

**Pages updated**:
- `wiki/index.md` — Codebases 8→9 (added pallas-forge at top of list); Sources 33→34 (new "Methodology (1)" subsection); header count bumped to 126 pages.

**Key result**: Two ingestion gaps closed. The pallas-forge page consolidates what the gemma4 program has already learned about the library (exp 20 + exp 33) rather than letting it sit only in experiment pages; the Karpathy-wiki source page documents where this wiki's operating protocol comes from, which matters for future schema edits.

**Notes**:
- **pallas-forge status recap (as captured on the codebase page)**: 3 reference kernels, tuner, roofline, XProf-trace integration — *but* forward-only (no `jax.custom_vjp`), so any training hypothesis that swaps a pallas-forge kernel into a module's `forward` crashes at `jax.value_and_grad` ("Linearization failed to produce known values"). Gemma4 exp 20 is the direct confirmation. Alternative for kernel swaps in training: [tokamax](codebases/tokamax.md) (exposes autodiff-capable `rms_norm` / `layer_norm` / splash-attention paths).
- **Generalizable lesson already in the wiki, now cross-linked**: from the 2026-04-23 Gemma 4 optimization-ceiling analysis (exp 33) — *"Pallas kernels are a net win only when XLA wasn't already exploiting the pattern via fusion"*. pallas-forge's v5e README numbers (3.44× RMSNorm isolated-microbench) do not predict in-graph wins on v6e-4; exp 33 showed an 8.1% regression. The codebase page flags this explicitly under "Performance-relevant surfaces §3 Isolated-microbench ≠ in-graph gain".
- **pallas-forge `TPU_SPECS` gap**: the hardware-preset table for `roofline_chart` covers v4/v5e/v5p only; no v6e entry. Anyone running `roofline_chart` on v6e must supply peak-TFLOPS and peak-HBM-bandwidth directly.
- **Karpathy-wiki source is methodology, not TPU-perf content**: ingested to make the schema's lineage traceable; no hypotheses generated. Its "Connections" section explicitly maps the source's pattern to this wiki's `SCHEMA.md` / `CLAUDE.md` / `index.md` / `log.md` implementations.
- **Ingestion-audit findings recorded for completeness**: (1) the `notes.md` TODO `using git submodule add https://github.com/linhkid/pallas-forge under raw/code, update readme` was already satisfied (`.gitmodules` entry exists, `raw/code/pallas-forge/` is checked out at commit `090510b`); the only missing piece was the wiki page, now filed. (2) `raw/sources/2026-karpathy-llm-wiki.md` was on disk but had no wiki counterpart, now filed. (3) No other unprocessed raw sources or un-documented submodules found.
- **Out of scope for this ingestion** (per `notes.md` lines 28–37, the gemma-4-E4B import + torchax trainer work): that is implementation, not ingestion; the Gemma 4 program already has its own `experiments/gemma4_autoresearch_optimization/` tree with 33 experiments completed and a ceiling analysis filed. Not touched.

## [2026-04-23] analyze | Gemma 4 E4B on v6e-4 — optimization ceiling reached at exp 25

**Op**: analyze (session ceiling synthesis).

**Pages created**:
- [`wiki/analyses/2026-04-23-gemma4-v6e4-optimization-ceiling.md`](analyses/2026-04-23-gemma4-v6e4-optimization-ceiling.md) — synthesis of the 33-experiment loop. Documents the trajectory (baseline 30,570 → exp 25 33,372, +9.2%), what worked vs what didn't, the step-time decomposition at exp 25, the Pallas-fuses-into-matmul lesson from exp 33, and what would actually move the needle next (hardware scale-up, scan-over-layers Option B, or accepting ceiling).

**Pages updated**:
- `wiki/index.md` — Analyses section populated.

**Key result**: Optimization loop has reached diminishing returns on this hardware/model combo. Trunk stays at exp 25 (33,372 TPS, seq=1024 batch=3 fsdp=4 bf16, splash_pallas + SEQ_MINOR + block=1024 + fused_bwd + bf16 CE + selective remat). Eight experiments since exp 25 (exp 26–33) produced zero further wins: scan-over-layers blocked (5 Gemma-specific issues), tokamax DPA blocked (mosaic_tpu no sliding-window), 2D mesh 2.4× slower at this chip count, Pallas RMSNorm −8.1% due to XLA already fusing it with neighbor matmuls, long-seq and XLA-flag-isolation experiments neutral or dominated.

**Notes**:
- **Generalizable lesson (exp 33)**: Pallas kernels are a net win only when XLA wasn't already exploiting the pattern via fusion. Splash won (exp 8) because XLA can't express online-softmax. RMSNorm loses because XLA already fuses it. Likely Pallas SwiGLU would too — don't build it.
- **Remaining viable paths** (none attempted here): (a) 300-500 LOC scan-over-layers Option B for compile-time compression, (b) hardware scale-up to v6e-8 or v5p-4 where 2D mesh and collective overlap become economic, (c) persistent compile cache (infrastructure only, no TPS).
- **Exp 28 kept but not merged**: +0.9% at seq=2048 b=1 strictly dominated by exp 25's higher-batch throughput. Preserved for long-seq reference.

---

## [2026-04-23] protocol + experiments | Gemma 4 E4B program — program.md formalization + exp2 crash

**Op**: manual (protocol formalization) + run-experiment (exp2, crashed).
**Pages created**:
- `wiki/experiments/gemma4_autoresearch_optimization/program.md` — the agent-facing protocol for this program, adapted from the sibling-wiki TorchTitan autoresearch template but specialized to Gemma 4 E4B / torchax / v6e-4.
- `wiki/experiments/gemma4_autoresearch_optimization/RESULTS.tsv` — tab-separated ledger (gitignored via `wiki/experiments/*/RESULTS.tsv`).
- `wiki/experiments/gemma4_autoresearch_optimization/OBSERVATIONS.md` — skim-and-reason aggregation log; backfilled with baseline + exp1 + exp2 blocks.
- `wiki/experiments/gemma4_autoresearch_optimization/2026-04-23-exp2-pin-out-shardings-rejected.md` — Exp 2 experiment page (crashed/invalid, reverted).

**Pages updated**:
- Program `README.md` — stripped the in-README "Optimization loop procedure" section; replaced with a short "How to start the optimization loop" pointer to `program.md`. History extended for exp2 + the protocol formalization.

**Key result**: **Exp 2 CRASHED** pre-trace with `ValueError: Received incompatible devices` on `weights['lm_head.weight']`. Attempt to pin `out_shardings=(loss_ns, weights_ns, opt_state_ns)` on `jax.jit` to fix the ~150 s step-1 recompile hit a tied-weight-dedup plumbing issue in torchax's `JittableModule` — the tied `lm_head.weight` ↔ `embed_tokens.weight` emits an `out_shardings` leaf whose device list collapses to `[0]` even though the input sharding is full-mesh `[0, 1, 3, 2]`. Reverted the `train.py` change; step-1 recompile remains open. Hypothesis marked `verdict: invalid` (pre-trace crash, nothing measured) — still viable under a different implementation (`dedup_parameters=False` + manual tying, or `with_sharding_constraint` inside the train step).

**Notes**:
- **Protocol formalization**: Karpathy-autoresearch-style discipline now lives in a dedicated `program.md` that defines the fixed bindings (hardware, conda env, trainer path, baseline command, libtpu version, profile/HLO conventions, xprof_mcp server), the architectural invariants (can/cannot-change tables for Gemma 4), the measurement protocol (TPS primary, MFU secondary, median steps 6–15), the ledger + observations schema, the full experiment loop, and accumulated heuristics. Future experiments start from reading `program.md` and tail-ing `OBSERVATIONS.md` / `RESULTS.tsv`.
- **Retroactive backfill**: baseline + exp1 + exp2 are transcribed into `RESULTS.tsv` and `OBSERVATIONS.md` for continuity — these experiments predate the formal protocol but provide the foundation the protocol was written against.
- **libtpu is at 0.0.40, the latest** (verified across PyPI, `libtpu-lts-releases`, `libtpu-nightly-releases`). The exp1 "unknown flag" failure was a name error (real name is `--xla_tpu_overlap_compute_collective_tc`, not `_comms`), not a version issue. Documented the symbol-dump-as-source-of-truth trick in `program.md` heuristics.
- **Discovered gotcha**: tied `lm_head` ↔ `embed_tokens` + torchax `JittableModule.dedup_parameters=True` + HF `load_state_dict(assign=True)` interact in a way that confuses `out_shardings` construction. Logged as a `program.md` heuristic; a future experiment can route around it via `dedup_parameters=False`.
- Step-1 recompile mitigation deferred; next experiment focuses on a structural memory-saving change (per the memory-ceiling rule — baseline hit 95% HBM).

## [2026-04-23] experiment | Gemma 4 E4B — Exp 1: async-collective XLA flags (REFUTED)

**Op**: run-experiment (first optimization-loop cycle on the Gemma 4 program).
**Pages created**: `wiki/experiments/gemma4_autoresearch_optimization/2026-04-23-exp1-async-collective-flags-rejected.md`.
**Pages updated**: program `README.md` history (baseline-seq-1024 + Exp 1).
**Raw artifacts**: `raw/profiles/2026-04-23-gemma4-loss-confirm/` (20-step seq=1024 baseline, clean loss), `raw/profiles/2026-04-23-gemma4-exp1-async-collectives/` (exp1 trace). Both symlinked into the live xprof instance as `gemma4_baseline_seq1024_20260423` and `gemma4_exp1_async_collectives_20260423`.

**Key result**: **Hypothesis REFUTED.** Enabling `--xla_tpu_enable_latency_hiding_scheduler` + `--xla_tpu_enable_async_collective_fusion` (+ `fuse_all_gather`, `multiple_steps`) via `LIBTPU_INIT_ARGS` regressed steady-state step time **134.4 → 168.3 ms (+25 %)** at seq=1024, batch=1, FSDP=4, v6e-4. Collectives got faster (all-gather −5 ms, all-reduce-scatter −9 ms) but compute-fusion memory traffic blew up (convolution fusion +2.5× bytes, loop fusion +1.9×). Scheduler gained freedom to reorder; it fused collectives but broke compute-order locality. Loss trajectory preserved (bf16-reorder noise only).

**Notes**:
- 20-step loss-descent confirmation (pre-experiment): loss 3.93 → 1.97 over 20 steps at seq=1024; step time stable at 134.4 ± 0.5 ms. Confirms the baseline scaffold is training correctly.
- Ran full autoresearch loop: baseline profile → hypothesis pick (#9) → experiment → profile diff → verdict → file page → pick next hypothesis. First complete cycle on this program.
- **Flag-placement gotcha**: `--xla_tpu_*` flags go in `LIBTPU_INIT_ARGS`, **not** `XLA_FLAGS`. First attempt bombed with `parse_flags_from_env.cc:234 Unknown flags in XLA_FLAGS`. Logged for future experiments — hypothesis-writers should not trust the "XLA flag" name.
- **Libtpu 0.0.40 does not know `--xla_tpu_overlap_compute_collective_comms`.** Dropped before the successful run. Good instance of a flag catalog drifting vs. the installed runtime.
- **xprof-mcp tools used in the loop**: `list_runs`, `get_overview`, `get_top_hlo_ops`, `get_op_profile`, `get_memory_profile`, `get_device_information`. The HLO-op-level before/after diff was the decisive evidence for "refuted" — wall-clock alone would have been ambiguous, but the bytes-accessed explosion made the mechanism clear.
- Hypothesis parked, not retired — at larger effective batch (via remat or NaN-fix at seq=2048) the collective-overlap win may swamp the compute-locality loss. Revisit after hypothesis #6.
- Next natural hypothesis: **fix the step-1 recompile** (explicit `in_shardings` / `out_shardings` on `jax.jit`) — saves ~150 s per experiment run, pure iteration-speed win. Then **#6 selective remat**.

## [2026-04-22] experiment | Gemma 4 E4B baseline (v6e-4, FSDP)

**Op**: run-experiment (baseline infrastructure check)
**Pages created**: `wiki/experiments/gemma4_autoresearch_optimization/2026-04-22-baseline.md`.
**Pages updated**: program `README.md` (History section updated with first baseline numbers); `wiki/experiments/gemma4_autoresearch_optimization/torchax/{train.py,model/sharding.py,requirements.txt}` (8 targeted fixes to get the scaffold to actually run — enumerated on the baseline page).
**Raw artifacts**: `raw/profiles/2026-04-22-gemma4-baseline/` — xprof trace, steps 5–7 at seq=2048.

**Key result**: First working run. Steady-state **249 ms/step at seq=2048, batch=1, FSDP=4 → ~33k tokens/sec, ~26% MFU** (corrected from an initial `6PT` overestimate of 44% — `P=8B` double-counts Gemma 4's Per-Layer-Embedding lookup tables, which don't participate in matmul FLOPs). Seq-length sweep: **13% MFU @ seq=512, 23% @ seq=1024, 26% @ seq=2048**. Verdict: **supported** on the infrastructure-check hypothesis; **NOT a quality-valid baseline at seq=2048** until the NaN loss is fixed.

**Notes**:
- Hardware was **v6e-4**, not the v6e-8 the scaffold README assumed. FSDP default auto-picked fsdp=4 from `jax.device_count()` — no config change needed.
- **Default sharding strategy flipped TP=8 → FSDP** per user directive. Both paths now available; FSDP is the default. FSDP sharding rule: every ≥2D param shards on its largest dim divisible by `fsdp_size` over a 1D `'fsdp'` axis.
- **Python 3.13.13** env (`gemma4_py313`) works with jax 0.10.0, torch 2.11.0+cpu, torchax 0.0.12 (editable install from the wiki submodule), transformers 5.7.0.dev0 (Gemma 4 only on main), datasets, optax, accelerate.
- **8 scaffold fixes** applied to get from written-but-untested to running; all enumerated in the baseline page's "Scaffold changes applied" table. Most consequential:
  1. `interop._jax_view` → `interop.jax_view` (pytree-map variant; single-value left torch tensors unconverted).
  2. Load `Gemma4ForConditionalGeneration` (not `Gemma4ForCausalLM`) — HF checkpoint is multimodal-only and `ForCausalLM` silently re-inits every weight (name-prefix mismatch against `model.language_model.*`).
  3. Monkey-patch `model.forward` with a text-only path (bypass the multimodal orchestrator's `input_ids[mask] = pad` which is not JIT-traceable).
  4. Apply `final_logit_softcapping=30.0` in the text-only forward (did not fix the seq=2048 NaN, but is semantically required).
  5. Requirements.txt pointed torchax at `pytorch/xla` subdirectory; actual repo is `google/torchax`. Switched to an editable install from the wiki's own submodule (commit `8f957d1`).
- **Correctness issues found by the baseline** (flagged for the next experiment):
  - **NaN loss at seq≥2048** — loss is clean at seq∈{512, 1024}. Likely bf16 attention overflow or a Gemma 4 hybrid-attention mask edge case. Prerequisite for any seq=2048 perf work.
  - **OOM at batch=4, seq=2048** — attention N×N materialized (no flash/splash). Directly motivates hypothesis #1 (Splash Attention).
  - **Step 1 recompiles** — both step 0 and step 1 take ~155 s, step 2+ hits the cache. Likely a sharding-spec / donation mismatch on step-1 inputs. Low-effort follow-up: pass explicit `in_shardings` to `jax.jit`.
- **Arithmetic error noted publicly**: initial MFU claim of 44% was wrong — used `6PT` with `P = headline 8B` instead of `P ≈ non-embedding-matmul params`. Gemma 4's "E4B" headline includes PLE lookups which add params but no matmul FLOPs. User caught this; corrected to ~26% MFU via detailed per-matmul FLOP counting. Keep this in mind for future model-size-related FLOP estimates.
- **Compile time dominates** for short runs: 2 × ~155 s compile vs 2.5 s of useful work at seq=2048 × 8 steady-state steps. Hypothesis #7 (scan-over-layers) should collapse this.

## [2026-04-22] scaffold | Gemma 4 E4B torchax trainer (untested)

**Op**: manual (scaffold code for first execution path of an optimization program)
**Pages created**: 9 files under `wiki/experiments/gemma4_autoresearch_optimization/torchax/` (1,215 lines total):
- `train.py` (439 lines) — fine-tune trainer with profile-step capture.
- `model/sharding.py` (245 lines) — 2D `(dp, tp)` mesh + NeMo-Megatron sharding adapted for Gemma 4 GQA.
- `model/README.md` (118 lines) — config + sharding assumptions, upstream source of truth.
- `data.py` (118 lines) — wikitext loader + fixed-length packer.
- `README.md` (134 lines, includes augmented "Running the trainer" section) — runbook.
- `model/__init__.py` (46 lines) — re-exports `Gemma4Config`, `Gemma4ForCausalLM`, `Gemma4ForConditionalGeneration`, `Gemma4Model` from `transformers`.
- `run.sh` (50 lines) — wrapper setting `XLA_FLAGS` + `LIBTPU_INIT_ARGS`, forwards to `train.py`.
- `config.yaml` (22 lines) — default args.
- `requirements.txt` (43 lines) — deps pinned against torchax commit `8f957d1`.

**Pages updated**: none (wiki markdown was already in place; only trainer code added).

**Key result**: First execution path of the Gemma 4 program is scaffolded. Status marked **UNTESTED** in multiple places — scaffold written from ingested source pages (jax-huggingface part 2 sharding recipe, torchax codebase architecture, xprof capture docs) without running a single step.

**Research findings worth capturing as wiki content later (not yet pages)**:
- **Gemma 4 E4B is public Apache-2.0**, not gated (login still required for HF hub). `config.json` readable at `https://huggingface.co/google/gemma-4-E4B/raw/main/config.json`.
- **Architecture specifics**: 42 layers, hidden=2560, `num_attention_heads=8`, `num_key_value_heads=2`, `head_dim=256`, `intermediate_size=10240`, vocab=262144, sliding_window=512, max_position=131072. "E4B" = effective-4B via Per-Layer Embeddings; **~8B with embeddings**.
- **Novelties vs Gemma 3**: hybrid attention (local SW 512 + global), `num_kv_shared_layers=18` (cross-layer KV sharing), `rope_type=proportional` with `partial_rotary_factor=0.25` on full-attention layers, `final_logit_softcapping=30.0`, `gelu_pytorch_tanh` MLP, `tie_word_embeddings=true`.
- **Multimodal**: E4B ships vision + audio branches. Trainer targets text-only (`Gemma4ForCausalLM` with fallback to `Gemma4ForConditionalGeneration`).
- **Sharding corner case**: `num_kv_heads=2` does NOT divide `tp=8`. Default partitioning therefore **replicates K/V projections** rather than silently dropping parallelism — flagged as a future hypothesis.
- **Canonical class names** (per HF `transformers` main, `transformers_version: 5.5.0.dev0`): `Gemma4Config`, `Gemma4ForCausalLM`, `Gemma4ForConditionalGeneration`, `Gemma4Model`. Transformers ships Gemma 4 in `src/transformers/models/gemma4/`.
- **DeepMind gemma repo** exposes `gm.nn.Gemma4_E4B()` + `gm.ckpts.CheckpointPath.GEMMA4_E4B_IT` — native-JAX reference for the `../jax/` folder when that path is activated.

**Assumptions flagged for baseline-run verification**:
1. `Gemma4ForCausalLM` import works; falls back to `ForConditionalGeneration` if not.
2. HF state-dict key naming matches Gemma-family convention (`q_proj`, `k_proj`, …, `embed_tokens`) — regex-based sharder in `sharding.py` is fragile and should be verified with `print(list(model.state_dict())[:20])`.
3. `num_kv_shared_layers=18` may surface as extra/renamed params not covered by the regex — they default to replicated (conservative).
4. torchax API at commit `8f957d1` matches the calls in `train.py` (`JittableModule`, `interop.jax_view/torch_view`, `enable_performance_mode`, `apply_jax_`, `save_checkpoint`).
5. `wikitext-2-raw-v1` chosen as default (small, fast smoke-test). `wikitext-103-raw-v1` available via flag.
6. HF pytree registration targets `CausalLMOutputWithPast` + `DynamicCache`; `StaticCache` registration deferred to a future decode hypothesis.
7. `final_logit_softcapping=30.0` assumed to be implemented inside HF's forward (Gemma 2/3 precedent).

**Known gaps in the scaffold**:
- `--grad_accum` is parsed but not threaded through the training loop.
- No `with_sharding_constraint` activation annotations inside the forward (relies on GSPMD propagation from weight shardings).
- No checkpoint *load* path — `--checkpoint_dir` only saves.
- Splash-attention swap (program hypothesis #1) not wired yet.
- Optimizer states inherit gradient dtype (bf16 if forward is bf16); no explicit fp32 promotion — a baseline concern.

**Next steps for the human** (runbook):
1. `pip install -r wiki/experiments/gemma4_autoresearch_optimization/torchax/requirements.txt` on a v6e-8 host.
2. `huggingface-cli login` with a token that's accepted the Gemma license.
3. `bash wiki/experiments/gemma4_autoresearch_optimization/torchax/run.sh --steps 5 --profile_steps 3`.
4. File the baseline numbers into `wiki/experiments/gemma4_autoresearch_optimization/<YYYY-MM-DD>-baseline.md` (the first dated experiment page).

## [2026-04-22] file-program | Gemma 4 E4B — TPU autoresearch optimization

**Op**: manual (file a new optimization program)
**Pages created**: `wiki/experiments/gemma4_autoresearch_optimization/README.md` — program page for `google/gemma-4-E4B` on TPU v6e via torchax.
**Pages updated**: `wiki/index.md` — Models section (0 → 1).

**Key result**: First `model/` analogue filed. 16 open hypotheses consolidated from Wave 1/2 findings, the xprof-mcp TPU_OPTIMIZATION guide, and the Ultra-Scale Playbook — now have a place to attach. Baseline not yet captured; hypothesis #0 in the ranked list is "capture baseline profile."

**Notes**:
- **Intentional schema deviation**: SCHEMA.md specifies `wiki/models/<slug>.md` for model-under-optimization pages and `wiki/experiments/<YYYY-MM-DD>-<slug>.md` (flat) for experiments. This program uses a **nested folder** `wiki/experiments/gemma4_autoresearch_optimization/` that co-locates the program README (functions as the model page), the dated experiment files (schema-conformant names inside the folder), and optionally local scripts/code. Rationale: a long-running optimization program generates many related files and benefits from being namespaced together; the flat experiments/ directory would make it hard to find "everything about Gemma 4" vs. "everything about the next model."
- This deviation is the **second** intentional one in the wiki (first was the `autoresearch` codebase page's reframed "Structural surfaces we borrow" H2). If it works, the next SCHEMA.md edit should codify `wiki/experiments/<program-slug>/` as a permitted layout for multi-experiment programs, with the README.md inside doubling as the `model` page.
- **Code location decision (2026-04-22)**: option (b) — inside the program folder — selected by human. Further refined: **split into two sibling subfolders by execution path** rather than one `code/` folder:
  - `wiki/experiments/gemma4_autoresearch_optimization/torchax/` — primary, Gemma 4 via torchax.
  - `wiki/experiments/gemma4_autoresearch_optimization/jax/` — secondary, native-JAX port (port-equivalence discipline required: must reproduce torchax-baseline outputs within bf16 tolerance before perf numbers count).
  Each subfolder has its own README documenting conventions (dated copies for divergent scripts, relative-path references from experiment pages, binaries go to `raw/profiles/`, not these folders). The program's top-level README links both. Next SCHEMA.md update should codify `wiki/experiments/<program-slug>/` folders with execution-path-named subfolders for code as a permitted layout.
- Hypotheses are listed on the program page but **not filed as `wiki/hypotheses/*.md` individually** — the program page serves as the consolidated ranked list. Once individual hypotheses become in-flight experiments, each will be promoted to `wiki/hypotheses/<slug>.md` per schema.
- **Gemma 4 E4B**: user confirmed the identifier is correct (per https://huggingface.co/google/gemma-4-E4B). Claude's training cutoff predates this model's release; treating it as a black-box target with Gemma-family architecture (GQA, SwiGLU, RMSNorm) until the baseline ingest confirms exact config.

## [2026-04-22] lint | link hygiene + cross-link + 4 missing stubs

**Op**: lint (automated pass, 2 parallel subagents + main-thread checks)
**Pages created**: `wiki/concepts/{hbm-bandwidth,ridge-point,reduce-scatter,trace-me}.md` (4 stubs).
**Pages updated**:
- `wiki/codebases/tokamax.md` — 2 broken links fixed (`../scaling-book.md` / `../stablehlo.md` → `./scaling-book.md` / `./stablehlo.md`).
- `wiki/codebases/xprof.md` — 5 broken links fixed (missing `2026-` year prefix on xprof source slugs; also dropped stale `(stub — fill once source pages land)` annotations now that the targets exist).
- `wiki/concepts/{mark-step-sync,tensor-parallelism,sharding,ici,collective-communication,all-reduce,kv-cache,static-cache,jax-trace,decode-profile-signature,dcn,attention-block-sizes}.md` — 12 stubs had jax-huggingface and/or ultrascale-playbook sources added to their `## Sources` section.
- `wiki/index.md` — totals (77→81 concepts, 118→122 pages); Performance metrics & roofline (11→13), Parallelism & collectives (11→12), Profiling (10→11).

**Key result**: All broken markdown links in wiki/ eliminated. 0 orphan pages. Wave-2 ↔ human-ingest cross-link asymmetry closed for the known-relevant concept stubs.

**Notes**:
- **Broken-link residue**: 10 intentional placeholder links in `SCHEMA.md` itself (e.g., `../sources/2022-flash-attention.md`, `relative/path.md`, index-template `<slug>` placeholders). Left as-is — they are documentation examples.
- **Orphans**: 0 under `wiki/sources/` and `wiki/concepts/`. Codebase pages are expected to be lightly-linked (mostly from `index.md` and a few sources) and were not scanned as orphan candidates.
- **Unlinked-mention candidates reported but NOT auto-fixed** (judgment required): top 20 opportunities to wrap concept names in prose as markdown links. The main offenders (by raw mention count in prose without a link) are `megascale` (24× in megascale-viewer source), `hbm` (19× in TPU_OPTIMIZATION), `ici` (16× in TPU_OPTIMIZATION), `splash-attention` (12× in its own source page — expected), and `hlo` (12× in hlo-op-stats). Most of these are "first-mention" linking candidates rather than blanket-wrap-every-mention — deferred to per-page editing rather than bulk automation.
- **Cross-link held-backs** (content-unjustified, flagged for future reconsideration):
  - `jax-huggingface-part-1` → `xla-flags` or `hlo-dumping-and-diffing`: source does not discuss either.
  - `jax-huggingface-part-4` → `custom-trace-annotations`: hardware is A100 GPU and the source only mentions profiling as a gap.
  - `ultrascale-playbook` → `int8-quantization`: playbook covers FP8 (DeepSeek tile-scaled), not int8/AQT. Concept mismatch.
- **Submodule commit freshness**: all 8 codebase pages' `commit:` frontmatter matches current `git submodule status`. No drift.
- **Frontmatter `sources:` field**: all 81 concept stubs consistently carry `sources: N`. Schema doesn't require it, but the vault convention is now established and consistent — no reconciliation needed. If desired, a future edit to `SCHEMA.md` could codify the convention.

## [2026-04-22] ingest-source | Ultra-Scale Playbook (Tazi et al., HF, 2025-02-19)

**Op**: ingest-source
**Pages created**:
- `wiki/sources/2025-ultrascale-playbook.md` (primary source page — **first non-2026 slug**: playbook is dated Feb 2025).
- `wiki/concepts/{ring-attention,context-parallelism,sequence-parallelism,pipeline-parallelism,expert-parallelism}.md` — 5 stubs for parallelism concepts not present after Wave 2.

**Pages updated**:
- `wiki/concepts/{rematerialization,flash-attention,splash-attention,fsdp,tensor-parallelism,sharding,async-collectives,dtype-strategy}.md` — appended the new source to `## Sources` with a one-line GPU↔TPU claim.
- `wiki/codebases/{tokamax,torchax,scaling-book,autoresearch}.md` — appended the source under `## Sources` with a per-codebase connection note.
- `wiki/index.md` — added the source under Sources and the 5 new stubs under Concepts.

**Raw artifacts**:
- `raw/sources/2025-ultrascale-playbook.html` (788 KB full HTML capture of the static asset URL).
- `raw/assets/ultrascale-playbook/` — **90 figures, 5.2 MB**. Every `<img>` referenced by the playbook, downloaded from `nanotron-ultrascale-playbook.static.hf.space`. Referenced inline in the source page; the full inventory is tabulated at the bottom of that page.

**Key result**: —

**Notes**:
- Emphasis directed by human: **GPU/PyTorch ↔ TPU/JAX differences in scaling/optimization mechanics**. The source page's centrepiece is a 20-row translation matrix ("Axis | GPU/PyTorch (playbook) | TPU / JAX / XLA | Tuning surface that actually matters on TPU"). Every playbook claim in the Key-claims table is annotated with "Transfers to TPU?" and, where not, the TPU delta.
- GPU-specific sections explicitly flagged as **not transferring**: Section 10 (memory coalescing, tiling, thread coarsening, control divergence, `torch.compile`+Triton) — different programming model; Pallas Mosaic-TPU is our analogue with its own tuning vocabulary.
- The HF Space URL is dynamically rendered; `WebFetch` on the public URL returned a loading shell. The static-asset URL (`nanotron-ultrascale-playbook.static.hf.space/index.html`) served the complete document and all figures.
- **6 hypothesis candidates** surfaced but **not filed** as `hypotheses/*.md` — schema requires a `model:` slug; no model page exists yet. Listed on the source page under `## Gaps & caveats`, to be promoted when the first model page is filed:
  1. Selective activation recomputation via `jax.checkpoint_policies` (Korthikanti 70% / 2.7% claim).
  2. Wire tokamax `ring_attention_kernel` through `dot_product_attention` dispatch (kernel exists, API gap only — Wave 1 finding + this source confirms it).
  3. Zig-Zag Ring Attention on TPU — no implementation found in any ingested codebase; open algorithmic port from Brandon et al. 2023.
  4. TPU-native Pallas kernels for `gated_linear_unit` and `layer_norm` in tokamax — Wave 1 finding that these fall back to XLA reference; this source quantifies the upside category (fused kernels pay off on memory-bound ops).
  5. DeepSeek-V3 tile-scaled FP8 (1×128 activations, 128×128 weights+scales) on v6e MXU.
  6. Expose tokamax splash-attention **backward** block sizes to the autotuner — Wave 1 hidden-tuning-surface finding cross-referenced.
- Concept-page convention in the vault includes a `sources: N` frontmatter integer alongside the `## Sources` H2 section. SCHEMA.md prescribes only the section. My 5 new stubs follow the vault convention (both). Noting the coexistence; not reconciling here.
- Index entry for this source is added alongside the existing Wave 2 rebuild; Concepts section in the index now lists my 5 new stubs explicitly.

## [2026-04-22] ingest-source + stub | Wave 2 — profiling & optimization docs

**Op**: ingest-source (batch) + stub (concept stubs)
**Pages created**: 28 source pages + 72 concept stubs.
- Sources: `wiki/sources/2026-xprof-mcp-tpu-optimization.md`; xprof docs `2026-xprof-{overview-page,trace-viewer,memory-profile,memory-viewer,graph-viewer,utilization-viewer,terminology,hlo-op-stats,hlo-op-profile,framework-op-stats,perf-counters,custom-call-profiling,capturing-profiles,jax-profiling,pytorch-xla-profiling,tensorflow-profiling,docker-deployment,kubernetes-deployment,roofline-model,megascale-stats,megascale-viewer,megascale-viewer-sql}.md`; tokamax docs `2026-tokamax-{supported-ops,basic-usage,splash-attention,autotuning,benchmarking}.md`.
- Concepts: 72 stubs under `wiki/concepts/` grouped as Architecture & hardware (12), Performance metrics & roofline (11), Compiler & HLO (12), Kernels (8), Optimization techniques (7), Parallelism & collectives (7), Inference (5), Profiling (10).
**Pages updated**: `wiki/index.md` (Sources/Codebases/Concepts sections rebuilt; merged with the concurrent `jax-huggingface` ingest).
**Key result**: ~3,475 lines of source content + 72 concept stubs. Wiki now has a working vocabulary — hypothesis candidates can be stated in terms of existing concepts with citations.
**Notes**:
- Six subagents ran in parallel for source ingestion (one per content group); a seventh consolidated concept stubs from their deduplicated recommendations.
- Subagent reports surfaced additional hypothesis candidates beyond Wave 1 findings (not yet filed):
  - **xprof-mcp TPU_OPTIMIZATION**: per-matmul fp32 cast ~17% overhead; int8 shifts v5e critical batch 240→~120→~240; Llama-2-7B decode 8.8× from static-cache; flash attention saves ~32 MB/layer/request at N=4096; selective AC ~+2.7% compute for ~70% activation memory. Most claims are v5e-anchored — generalization to v6e is not pinned down in the source.
  - **tokamax-supported-ops**: `docs/supported_ops.md` lists `dot_product_attention` as GPU-only, but the code ships a TPU Pallas/Mosaic backend — doc is stale. Flagged on the source page.
  - **tokamax-splash-attention / autotuning**: raw docs are 2–3 line stubs; source pages were written from code + basic-usage doc. The autotune backward-pass block-size coverage gap (Wave 1) is now cross-linked from `autotuning.md` and `attention-block-sizes.md`.
  - **xprof-roofline**: the "bytes accessed" definition in arithmetic intensity isn't scoped to a memory tier in the doc; peak FLOPs / bandwidth per TPU generation are pulled from Device Information at runtime and not listed.
  - **xprof-megascale**: `megascale_viewer_sql` has a minor inconsistency between `pt.name` vs `ppt.name` as `device` column across the two example queries — could mislabel rows.
  - **xprof-perf-counters**: doc lists filters/columns but not individual counter semantics — deeper docs/source needed before counter-level hypotheses.
  - **xprof-hlo-op-profile** "wasted time" sort is computed against FLOPs utilization only; will underweight memory-bound ops.
- Concept stubs: 72 created; subagent flagged 4 more worth adding in a follow-up (`hbm-bandwidth`, `ridge-point`, `reduce-scatter`, `trace-me`/TraceMe). Deferred — will add if Wave 3 scaling-book references them in depth.
- No broken markdown links introduced: grep confirmed `reduce-scatter` only appears as prose in `fsdp.md` / `collective-communication.md`, not as a link.
- No schema deviations this wave.
- Concurrency note: the `jax-huggingface` codebase + 4 source pages were ingested by the human (next log entry) while Wave 2 subagents were running; index.md was merged to reflect both. Wave 2 subagents did not see or cross-link to the jax-huggingface pages; Wave 3 (scaling-book) or a dedicated LINT pass should add cross-links from `jax-huggingface-part-{2,3}.md` into the new concept stubs (sharding, tensor-parallelism, static-cache, kv-cache, splash-attention, etc.).

## [2026-04-22] ingest-codebase + ingest-source | jax-huggingface (learning_machine subfolder)

**Op**: ingest-codebase + ingest-source (combined, user-directed option B)
**Pages created**: wiki/codebases/jax-huggingface.md; wiki/sources/2026-jax-huggingface-part-{1,2,3,4}.md
**Pages updated**: wiki/index.md; .gitmodules (added `raw/code/learning-machine` submodule, commit `93328b2`)
**Key result**: 5 pages written. Ingestion scoped to `jax-huggingface/` subfolder of `qihqi/learning_machine`; sibling subprojects (llama_ref, spmd_sharding, torch_pallas, flash_attn_speed, jax_perf, etc.) deferred. First `source/` pages in the wiki — exercised the source template alongside the codebase template.
**Notes**:
- Per-post data points captured in source-page tables: Part 1 v6e 1-chip Llama-2-7B forward 4.37s→13ms; Part 2 8-chip TP 13ms→3.4ms (3.8× cached, blog rounds to 4.3×); Part 3 50-tok decode 130.9s DynamicCache eager → 88.4s StaticCache eager → 14.77s StaticCache+jit (8.9×); Part 4 **A100 GPU, not TPU**: 5.9s→1.07s/image after VAE `methods_to_compile=['decode']` fix.
- **HF API drift flagged:** Part 3 post text's `StaticCache` pytree flattener (`cache.key_cache`/`cache.value_cache`) does NOT match the companion `jax_hg_03.py` (`cache.layers[i].keys`/`.values`). Script is current-HF-API version. Noted on codebase page and Part 3 source page. Candidate observation once a `model/` page exists.
- **Hardware ambiguity in Part 3:** post does not state device for the 130/88/14.77s numbers. Flagged in source-page "Gaps & caveats". Resolving this is a prerequisite before using those numbers as a baseline.
- **Part 4 hardware is A100 GPU.** Kept ingested because the `torchax.compile` / `CompileOptions` / `methods_to_compile` / scheduler-move patterns are TPU-portable, but reported numbers are not. Explicitly flagged on both the codebase page and Part 4 source page.
- Performance-relevant surfaces on the codebase page enumerate 10 concrete anchors (sharding recipe, KV-cache post-prefill sharding, functional_call escape from captured-constants, methods_to_compile override, static_argnames routing, scheduler tensor-move, pytree cookbook, static-arg strategies, default_matmul_precision, profile-capture idiom) — each grounded in a specific file:line.
- No hypotheses / concept stubs filed — no `model/` page yet to attach them to. Per-page "Future hypothesis anchors" sections carry the candidates forward.

## [2026-04-22] ingest-codebase | Wave 1 — seven repos

**Op**: ingest-codebase (batch)
**Pages created**: wiki/codebases/{xprof,xprof-mcp,torchax,tokamax,stablehlo,scaling-book,autoresearch}.md
**Pages updated**: wiki/index.md
**Key result**: 7 codebase parent pages written in parallel. Total 935 lines. Commit SHAs recorded in each page's frontmatter.
**Notes**:
- Per-repo "discuss before writing" step was batched into a single up-front categorization with the human (A=direct ingest, B=methodology, C=book-as-sources) rather than seven separate discussions.
- Scope discipline held: codebase pages map structure; the docs under each repo's `docs/` were deferred to Wave 2 (profiling/optimization) and Wave 3 (scaling-book chapters as sources). No source/concept/hypothesis pages created.
- Noteworthy findings surfaced during ingestion, flagged for follow-up when hypotheses are formulated:
  - **tokamax**: TPU `gated_linear_unit` and `layer_norm` have NO TPU-specific kernel — they silently fall back to the XLA reference. Candidate hypothesis source.
  - **tokamax**: splash-attention backward-pass block sizes (`block_q_dkv`, `block_kv_dkv`, `block_kv_dkv_compute`, `block_q_dq`, `block_kv_dq`) are exposed via `SplashConfig` but NOT surfaced to the autotuner (hidden tuning surface). Note: a Splash Attention kernel is also available upstream in JAX at `jax.experimental.pallas.ops.tpu.splash_attention` — the tokamax copy under `_src/ops/experimental/tpu/splash_attention/` mirrors that implementation, so hypothesis-writers can target either entry point.
  - **tokamax**: `ring_attention_kernel` exists but isn't reachable from `tokamax.dot_product_attention`.
  - **torchax**: `torchax.compile()` modes `dynamo` and `export` raise; only `jax` mode is functional.
  - **scaling-book**: book dated 2025-02-04 → Wave 3 source slugs will use `2025-` prefix, not `2026-`.
- **autoresearch** page uses the reframed H2 title **"Structural surfaces we borrow"** in place of "Performance-relevant surfaces" (the repo is methodology, not TPU-perf content). This is an intentional schema deviation for this single page.

## [2026-04-22] manual | wiki bootstrap

**Op**: manual
**Pages created**: SCHEMA.md, CLAUDE.md, wiki/index.md, wiki/log.md
**Pages updated**: —
**Key result**: —
**Notes**: Bootstrapped autoresearch-oriented schema from scratch. Independent of sibling `tpu_wiki` by design. Loop: sources + codebases + profiles → concepts + models → ranked hypotheses → experiments → observations → revised priors. Next: ingest first codebase and/or file the first model page.
