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

## queued next experiments

(matches the "Queued experiments" table in [the stack README](README.md))

- **exp 35 — splash Pallas attention in JAX.** Port `torchax/model/pallas_attention.py` to call JAX-native `jax.experimental.pallas.ops.tpu.splash_attention` directly (no `torchax.interop.call_jax` layer). Biggest known gap closer.
- **exp 36 — scan-over-layers.** torchax exp 26 parked this due to 5 stack-specific blockers (`ScannedModule` asserting `not kwargs`, heterogeneous state_dict due to kv-shared layers, etc.). The JAX port has none of those constraints — we control the forward directly — so Option B is more tractable here. Expected: ~40× compile-step-0 drop, maybe +2–5 % step time.
- **exp 37 — tokamax memory-efficient CE.** Fuses `lm_head + log_softmax + NLL` into one Pallas kernel, freeing ~1.5 GiB of fp32 logits memory. Potential unlocker for batch=4 on this stack.
- **exp 38 — step-1 recompile root-cause.** 119 s (~same as torchax) — needs `out_shardings` pinning + donation annotations. Pure iteration-speed win.
