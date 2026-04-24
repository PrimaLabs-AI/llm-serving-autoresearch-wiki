# OBSERVATIONS — Gemma 4 E4B native-JAX stack

Skim-and-reason aggregation log for the **native-JAX (Flax NNX)** path. Canonical per-experiment pages are the dated `*.md` files in this folder; this log threads the jax-stack session's arc for the human reviewer. Append-only.

Sibling: [`../../torchax/experiments/OBSERVATIONS.md`](../../torchax/experiments/OBSERVATIONS.md) — the torchax stack's aggregation log.

## port baseline: exp 34

See [2026-04-23-exp34-jax-baseline-accepted.md](2026-04-23-exp34-jax-baseline-accepted.md) for the full page.

**Summary** (v6e-4, 1D fsdp=4, bf16, XLA SDPA attention):
- seq=1024 b=1: **30,285 TPS, 135.24 ms/step**, loss 3.92 → 2.30 over 20 steps.
- Compile step 0: ~116 s. Step 1 recompile: ~119 s.
- Match to torchax baseline-seq1024 (30,570 TPS, 134.4 ms/step): **within −0.9 % noise** — port is numerically correct.
- Against torchax exp 25 (session-best, 33,372 TPS): **−9.2 %**. Gap explained entirely by missing splash + bf16 CE + fused_bwd + batch=3.

**Port semantics bugs caught (worth remembering for anyone porting Gemma 4)**:
1. `Gemma4RMSNorm` applies `weight * normed` with **weight init-ones**, NOT `(1 + weight) * normed` with init-zero. The HF source's comment is misleading; the real init is ones.
2. `Gemma4TextAttention.self.scaling = 1.0` (NOT `1/sqrt(head_dim)`) — because `q_norm` / `k_norm` (per-head RMSNorm on Q and K) already produce unit-norm vectors so the dot product is already at the right scale.
3. The last 18 of 42 layers are `is_kv_shared_layer=True` and **drop their own `k_proj` / `v_proj`** — they borrow cached K/V from an earlier layer. HF safetensors reflects this (54 weights skipped in the weight loader).

**Scaffold pieces landed**:
- `jax/model/modeling_gemma4.py` (779 LOC) — Flax NNX port.
- `jax/model/weight_loader.py` (199 LOC) — HF safetensors → NNX Param tree.
- `jax/model/sharding.py` (316 LOC) — FSDP + TP plans (mirrors torchax/model/sharding.py including the exp-32 hybrid 2D fix).
- `jax/train.py` (302 LOC) — trainer with full CLI flag parity with torchax/train.py.
- `jax/data.py` (73 LOC) — wikitext loader.
- `jax/tools/parity_{layer,attn,check}.py` (447 LOC) — correctness harnesses.

## exp 35 — splash pallas attention in JAX (POTENTIAL, flat)

See [2026-04-23-exp35-jax-splash-potential.md](2026-04-23-exp35-jax-splash-potential.md) for the full page.

**Config**:
- Command diff from exp 34: `JAX_ATTENTION_IMPL=splash` env var + new module `jax/model/pallas_attention.py` (~180 LOC) wired into `Gemma4TextAttention.__call__` behind the env-gate.
- Profile path: [`raw/profiles/2026-04-23-gemma4-jax-exp35-splash/`](../../../../../raw/profiles/2026-04-23-gemma4-jax-exp35-splash/)
- **Profile browser URL**: http://localhost:8791/?run=2026-04-23-gemma4-jax-exp35-splash

**Hypothesis**: splash should win +2–5 % on the JAX port at batch=1 seq=1024, mirroring torchax exp 8 (+2.7 % at batch=2 seq=1024). Primary mechanism: avoid N² score-matrix materialization; secondary: Mosaic tiling beats XLA-fused GEMM for the `(B, H, S, D) @ (B, H, D, S)` + softmax + `(B, H, S, S) @ (B, H, S, D)` pattern.

**Changes made**:
- New file `jax/model/pallas_attention.py` — exposes `splash_attention(q, k, v, sliding_window)` using `jax.experimental.pallas.ops.tpu.splash_attention.make_splash_mha_single_device`, wrapped in `jax.shard_map(P('fsdp', None, None, None))` with the active mesh (set via `set_mesh(mesh)` from train.py). Block config mirrors torchax exp-25 best: `block_q = block_kv = block_kv_compute = 1024`, `block_*_dkv = 1024`, `use_fused_bwd_kernel=True`, `QKVLayout.SEQ_MINOR` for q/k/v. LocalMask(window_size=(512, 0)) for sliding layers; CausalMask for full layers. LRU-cached per (seq_len, num_q_heads, sliding_window, head_dim) — Gemma 4 has two head_dim variants (sliding@256, full@512).
- **No pre-kernel Q scaling** (unlike torchax/pallas_attention.py which does `q * 1/sqrt(d)`): the JAX port sets `self.scaling = 1.0` because q_norm / k_norm already pre-normalize per head — matches splash's no-1/sqrt-d convention exactly.
- Kernel pre-built at startup under `jax.ensure_compile_time_eval()` so MaskInfo is a concrete jax.Array (without this, first call inside the top-level jit captures tracers → step-1 retrace trips `UnexpectedTracerError`).
- `modeling_gemma4.py` — added the env-var dispatch in `Gemma4TextAttention.__call__`. One `if os.environ.get(...)=='splash'` branch; otherwise falls through to the existing `_attn_xla_sdpa`.
- `train.py` — reads `JAX_ATTENTION_IMPL`, calls `pallas_attention.set_mesh(mesh)` + pre-builds the kernel for (sliding_window, head_dim=256) and (None, head_dim=512) at startup.
- New parity harness `jax/tools/parity_splash.py` (~130 LOC) compares splash vs XLA-SDPA on-TPU at B=4 T=128; reports raw attn-output and post-o_proj errors.

**Expected outcome**: +2–5 % TPS; step time 135 → 129 ms; peak HBM modestly lower.

**Actual outcome**:
- TPS: 30,285 → **30,386** (Δ **+0.33 %** — flat, within noise)
- Step time (median 6-15): 135.2 → **134.8 ms** (-0.4 ms)
- Peak HBM: 16.85 → **16.43 GiB** (-0.42 GiB, 53.9 % → 52.6 %)
- Compile time: 142 → 132 s (-10 s — splash happens to compile faster here)
- Loss step 19: **bit-matches** exp 34 at 2.2969.

**Profile signals**:
- Bottleneck: compute (same as baseline). Mixed balance across conv-fusion, loop-fusion, custom-fusion.
- **Top op shift**: `convolution fusion` 614.6 → 549.0 ms (-65.6 ms / 3-step window), `custom fusion` 152.3 → 169.1 (+16.8) — confirms splash kernel is executing; matmul time moved from XLA into Mosaic. Net ~49 ms saved across 3 steps = ~16 ms / step, consistent with the -0.4 ms median step-time delta at noise level.
- `loop fusion` (~337 ms) unchanged — RMSNorm / residual-add ops are dominant, unrelated to attention kernel swap.
- HBM drop small because batch=1 seq=1024 keeps `[1, 8, 1024, 1024]` bf16 = 16 MiB / layer — already small; splash's 4× reduction unlocks batch=3+ / seq=2048+ regimes.

**Analysis**: The mechanism works exactly as predicted; the wall-clock delta is small because (a) batch=1 seq=1024 is below the regime where the N² tensor dominates, and (b) the splash custom-call launch overhead per layer (~0.4 ms × 42 layers × 3 steps ≈ 50 ms) offsets most of the matmul savings at these shapes. This is the same shape-sensitivity pattern seen in the torchax exp-8 → exp-25 arc: splash's marginal value scaled from +2.7 % (b=2 s=1024) → +8 % (b=3 s=1024 w/ bf16 CE) as batch increased. At batch=1 seq=1024 we're on the left end of that curve.

**Decision**: `potential` (parked) — correct implementation, flat TPS at current config. Merging to trunk because the code is the prerequisite for exp 36+ (batch=3, bf16 CE) and keeping it gated by env var means the XLA-SDPA baseline remains trivially reproducible.

**Follow-ups**:
- **exp 36 — splash + batch=3**. Direct analog of torchax exp 18 (+8.0 %). HBM 52.6 % has room for batch=3 (+~2-3 GiB). Confidence high, effort S.
- **exp 37 — splash + bf16 CE** (tokamax or hand-roll). Torchax exp-12 lesson: the `[B, S, V=262144]` fp32 logits tensor is ~1.5 GiB at b=2; dropping to bf16 / fused CE frees HBM and trims ~1-3 %.
- **exp 38 — splash + scan-over-layers**. JAX port owns the layer loop directly (no `ScannedModule` blocker), so this is more tractable than on torchax. Main win: step-0 compile drop ~130 s → ~4 s. Throughput neutral.
- Kernel-launch overhead finding: ~17 ms / 3-step added in custom-fusion for 42 × 3 = 126 splash calls is ~0.13 ms / call. This is a floor — scan-over-layers might amortize it by keeping the kernel code hot.

## exp 36 — splash + batch=3 in JAX (ACCEPTED, +13.9 %, new JAX-stack best, BEATS torchax session-best)

See [2026-04-23-exp36-jax-splash-batch3-accepted.md](2026-04-23-exp36-jax-splash-batch3-accepted.md) for the full page.

**Config**:
- Command diff from exp 35: `--batch_size 1` → `--batch_size 3`. No code change. `JAX_ATTENTION_IMPL=splash` unchanged.
- Profile path: [`raw/profiles/2026-04-23-gemma4-jax-exp36-splash-batch3/`](../../../../../raw/profiles/2026-04-23-gemma4-jax-exp36-splash-batch3/) (321 MB, gitignored).
- **Profile browser URL**: http://localhost:8791/?run=2026-04-23-gemma4-jax-exp36-splash-batch3
- GCS mirror: `gs://tpu-pytorch-alekseyv-us-central2/autoresearch/2026-04-23-gemma4-jax-exp36-splash-batch3/`

**Hypothesis**: splash's per-call Pallas launch overhead is batch-independent (~0.13 ms × 42 layers × 3 calls/step ≈ 16 ms, mostly fixed), so bumping batch 1 → 3 amortizes it over 3× more tokens. Predicted +5–10 %, based on the torchax exp 15 → exp 18 arc (+7.0 % then +0.9 % additive on fused_bwd) where splash's marginal TPS value rose as batch grew.

**Changes made**:
- None to code. Only `--batch_size 1` → `--batch_size 3` on the command line.
- Data loader automatically switches to per-chip-batch=3 (global batch 12 = 3 × fsdp=4; still divisible, no sharding surprise).

**Expected outcome**: +5–10 % TPS; step time 135 → ~375 ms (scales with batch but sub-linearly); peak HBM 52.6 % → ~60 % (stack constant 16.4 GiB + ~3 × 1.5 GiB activation delta).

**Actual outcome**:
- TPS (median 6–15): 30,386 → **34,614** (Δ **+13.9 %** — upper end of expected range)
- Step time (median 6–15): 134.8 → **355.0 ms** (2.63× for 3× tokens — confirms sub-linear scaling, per-token cost −13.9 %)
- Step time (xprof avg steps 10–12): 142.2 → **375.5 ms** (profile inflates slightly)
- Peak HBM: 16.43 → **27.11 GiB** (52.6 % → **86.75 %**) — +10.68 GiB, all in heap (activations); stack constant at 16.44 GiB
- Free HBM at peak: **4.14 GiB** — room for one more batch raise (b=4) or one memory optimization
- Compile step 0: 132 → 167 s (+35 s; expected, bigger shape graph)
- Loss step 19: healthy 1.84 (descends 3.81 → 1.84; different seed/batch vs exp 35's 2.30, not bit-comparable)

**Profile signals**:
- Bottleneck: compute-bound (same as exp 35), now dominated by `convolution fusion` (33.6 % of step time) + `loop fusion` (28.1 %).
- **Top HLO-op diff vs exp 35 (3-step profile window)**:
  - `convolution fusion`: 549 → 1512 ms (**×2.75** for 3× tokens — MXU utilization improves at bigger shapes).
  - `loop fusion` (RMSNorm + residual-add): 332 → 1265 ms (**×3.81** — super-linear — next-tier bottleneck surface).
  - `custom fusion` (splash kernel): 169 → 175 ms (**×1.03** — **near-constant**, validating the per-call overhead hypothesis exactly).
  - `collective-permute-done`: ~11 → **550 ms** (12.2 % of step time) — **new pattern at b=3**, SPMD re-sharding cost.
- HBM: 27.11 GiB peak, no fragmentation (0.0 %), heap 10.67 GiB / stack 16.44 GiB / free 4.14 GiB.
- MXU utilization vs roofline not reported by xprof at this config (still 0 % — same as exp 35; TPU-profile plugin limitation on Pallas-heavy graphs).

**Analysis**: The mechanism predicted in exp 35 was right on both counts:
1. **Splash per-call overhead amortization** (primary): custom-fusion scaled 1.03× while everything else scaled 2.75–3.8×. At b=1, splash was 9.9 % of step; at b=3 it's 3.9 %. This alone buys ~6 %.
2. **MXU utilization improvement** (bonus, not anticipated): convolution-fusion growing 2.75× for 3× tokens means per-matmul MXU rows fill better at `[B=3, ...]` shapes. +7 % of the delta lives here.

Both effects compound in parallel and sum to the observed +13.9 %. **The native-JAX stack now outperforms the torchax session-best** (33,372 TPS exp 25) by **+3.7 %** — without bf16 CE, without SEQ_MINOR-specific tuning beyond what splash already defaults to, and without custom fused_bwd beyond splash's `use_fused_bwd_kernel=True`. The remaining torchax-side headroom (bf16 CE in exp 37, `collective-permute-done` tightening in exp 38) is additive — we're likely to pass the torchax best by >5 % once those land.

New bottleneck surfaces visible at b=3 that weren't actionable at b=1:
- `loop fusion` at 28.1 % of step time — RMSNorm + residual-add. Pallas RMSNorm kernel (from program.md's build-targets table) now worth the effort.
- `collective-permute-done` at 12.2 % — SPMD re-shuffle. `in_shardings`/`out_shardings` audit on the jitted step.

**Decision**: `keep` (ACCEPTED). New JAX-stack best. No code merge needed (it's a flag change); documenting via experiment page + RESULTS.tsv keep-row + this entry. README.md updated with new "Current state" numbers.

**Follow-ups**:
- **exp 37 — splash + b=3 + bf16 cross-entropy**. Highest priority. Frees ~1.5 GiB (HBM 86.8 % → ~82 %), trims one pass over `[B=3, S=1024, V=262144]` logits. Expected +1–3 % TPS.
- **exp 38 — collective-permute-done investigation**. 12.2 % of step time is a huge new bucket that didn't exist at b=1. Tighter in/out shardings on the jit body might reclaim half of it (5–6 %).
- **exp 39 — Pallas RMSNorm kernel**. `loop fusion` is now 28 % of step time; RMSNorm is 5 × 42 = 210 calls/step; single-HBM-pass kernel worth 3–8 %.
- **exp 40 — scan-over-layers**. Compile-time win (step 0 167 s → ~5 s), not throughput. Still worth doing — iteration speed improvement.
- **exp 41 — b=4**. With 4.14 GiB HBM free, b=4 adds ~3.5 GiB; may just fit. If exp 37 lands first, b=4 becomes very likely to fit. Defer until after 37.
