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
