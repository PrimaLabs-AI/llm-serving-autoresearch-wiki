"""Scan-over-layers wiring for Gemma 4 under torchax.

Intended to collapse the 42-layer Python `for layer in self.layers: ...`
in `Gemma4TextModel.forward` into a single `jax.lax.scan`, so the compiled
HLO holds O(1) copies of the layer block instead of O(42). Expected wins:
~150s -> ~5-15s compile; ~2-8% step time via shared activation buffers.

Pattern mirrors `model.pallas_attention.register_splash_attention(mesh)` and
`model.sharding.*`: `train.py` imports `register_scan_over_layers(mesh)` and
calls it once at startup; the register function installs (or refuses to
install) a monkey-patched `Gemma4TextModel.forward`. Errors never propagate
out of the register call -- on any problem we log
`[scan_layers] fallback to unscanned: <reason>` and leave HF's original
forward in place so the trainer still runs.


=============================================================================
STATUS, 2026-04-22 investigation:

OPTION A (wrap `model.model.language_model.layers` with
`torchax.train.ScannedModule`) DOES NOT APPLY to Gemma 4. It is a non-starter
for *five independent* reasons, each of which is load-bearing for correct
Gemma 4 semantics:

  1. `ScannedModule.forward` asserts `not kwargs` (train.py:110). Gemma 4's
     `Gemma4TextDecoderLayer.forward` is invoked *exclusively* via kwargs from
     `Gemma4TextModel.forward` (modeling_gemma4.py:1675-1684):
         decoder_layer(hidden_states, per_layer_input,
                       shared_kv_states=shared_kv_states,
                       position_embeddings=position_embeddings[layer_type],
                       attention_mask=causal_mask_mapping[layer_type],
                       position_ids=position_ids,
                       past_key_values=past_key_values, **kwargs)
     so handing the layer list to ScannedModule directly would blow the
     assertion the moment the wrapped forward is called.

  2. `ScannedModule._stack_layer_weights` does `torch.stack([m.state_dict()[k]
     for m in modules])` for every key. Gemma 4's layer stack is NOT
     homogeneous: layers with index >= `num_hidden_layers -
     num_kv_shared_layers` (= last 18 of 42) are `is_kv_shared_layer=True` and
     omit `k_proj`, `v_proj`, `k_norm`, `v_norm` entirely
     (modeling_gemma4.py:1179-1191 conditional `nn.Linear` creation;
     :1597-1600 adds those keys to `_keys_to_ignore_on_load_unexpected`).
     Stacking would KeyError or produce ragged tensors. Even if we padded
     the shared layers with zeros, `nn.functional_call` on the stacked module
     would still invoke K/V projections for layers that should NOT run them.

  3. Layers carry a *per-layer-index* attribute that the stacked-weights scan
     cannot thread. `Gemma4TextAttention.__init__` captures `layer_idx` as
     `self.layer_idx` and, for shared layers, `self.kv_shared_layer_index`
     (modeling_gemma4.py:1159-1171). These are plain Python ints baked into
     `self` at construction time; scan body runs `functional_call(one_mod, ...)`
     against a single prototype module whose `layer_idx` is fixed at 0. The
     downstream semantics (`shared_kv_states[self.kv_shared_layer_index]`
     lookup at :1219, `shared_kv_states[self.layer_idx] = ...` store at
     :1237) would all collapse to layer-0 keys.

  4. `shared_kv_states` is a Python `dict` mutated by side-effect inside the
     decoder layer (:1237 `shared_kv_states[self.layer_idx] = key_states,
     value_states`). `jax.lax.scan` requires the body to be a pure function
     producing a new carry with a static pytree structure. A dict whose *key
     set grows* across iterations is not a valid scan carry: the pytree's
     treedef changes step-to-step. ScannedModule's carry is just
     `(hidden_states, *non_hidden_args)`; it has no story for a mutating dict.

  5. Non-hidden args are passed through scan unchanged (not indexed by step).
     Gemma 4 needs per-layer indexing for at least three inputs:
         per_layer_input = per_layer_inputs[:, :, i, :]   (PLE, :1673)
         position_embeddings[layer_types[i]]              (per-layer cos/sin,
                                                           :1679)
         causal_mask_mapping[layer_types[i]]              (sliding vs full,
                                                           :1680)
     ScannedModule has no mechanism to provide a `[n_layers, ...]`-shaped
     array to index into. All three would need to be stacked and sliced by
     scan step, which is exactly the body-rewrite that ScannedModule was
     designed to *avoid* by treating all non-hidden args as constants.

None of the five are workaroundable by passing different arguments to
ScannedModule -- the class signature and body are fixed (no kwargs; stack
by state_dict key; no step-indexed inputs; carry == tuple of tensors). The
only way to use ScannedModule for Gemma 4 would be to write a wrapper
`Gemma4ScannableLayer` that (a) absorbs all the kwargs into positional args,
(b) takes layer_idx/layer_type/per_layer_input as scanned inputs, (c)
replaces the dict-based `shared_kv_states` with explicit `[2, B, KV_H, S, D]`
carry tensors, and (d) handles is_kv_shared_layer heterogeneity via a
conditional on a scanned boolean flag -- which is exactly Option B below.

Therefore this module currently implements only the diagnostic-and-fallback
path. Option B is sketched in the header of `_raise_option_b_needed`.


=============================================================================
WHAT OPTION B WOULD REQUIRE (concrete sub-problems for the next agent pass):

(B1) Homogenize the layer state_dict. For every K/V-shared layer, populate
     the missing `k_proj.weight`, `v_proj.weight`, `k_norm.weight`,
     `v_norm.weight` entries with zero-shaped stand-ins so
     `torch.stack([m.state_dict()[k] ...])` works. Must not change
     semantics; the scan body's attention must branch on a boolean
     `is_kv_shared_layer[i]` and skip the K/V projections when True,
     reading from a per-step `shared_kv_states_carry` tensor instead.

(B2) Replace dict-based `shared_kv_states` with a fixed-shape carry.
     Non-shared layers with `store_full_length_kv=True` write into a
     `[num_layer_types, B, n_kv, S, head_dim]` buffer indexed by the
     layer's type. Shared layers read from that buffer using
     `kv_shared_layer_index` pre-computed per layer as a static int array
     of length 42. This turns the mutating Python dict into a pytree-stable
     carry: `(hidden, kv_carry_k, kv_carry_v)`.

(B3) Stack per-layer inputs into `[42, ...]` tensors: `per_layer_inputs`
     already has shape `[B, S, 42, D_ple]`; transpose to
     `[42, B, S, D_ple]` as a scanned input. Build a `[42, B, H_q, S, D_q]`
     cos/sin stack by selecting the right layer_type slot per layer
     (there are only two: sliding/full). Same for attention_mask ->
     `[42, B, 1, S, S]` (or whatever the splash-attention mask shape is).
     These all become `scan`'s `xs` (step-indexed input), not carry.

(B4) Stack per-layer weights. The existing `ScannedModule` template already
     does `torch.stack` per state_dict key. Reuse the pattern but post-B1
     homogenization: after padding the kv-shared layers with zero stand-ins,
     all 42 state_dicts share the same key set, so the stack is well-typed.
     Scanned weight tensors gain a leading [42, ...] axis.

(B5) Stack `layer_scalar`. Each layer has a learnable `register_buffer
     ('layer_scalar', torch.ones(1))` (modeling_gemma4.py:1348) that the
     layer's forward applies as `hidden_states *= self.layer_scalar` (:1421).
     Stack into `[42, 1]` and index by scan step; include in the scanned
     weight pytree so autodiff propagates.

(B6) Scan body as a pure function. Define `scan_body((hidden, kv_k, kv_v),
     (weights_i, per_layer_i, cos_i, sin_i, mask_i, layer_idx_i,
      kv_shared_idx_i, is_shared_i, layer_scalar_i))`:
        - do input_layernorm
        - compute Q always; compute K/V only if not is_shared_i (use
          jax.lax.cond or a masked-select + masked-write)
        - write K/V into kv_k[layer_idx_i], kv_v[layer_idx_i] via
          dynamic_update_slice iff is_shared_i == False and
          store_full_length_kv (precompute)
        - read K/V from kv_k[kv_shared_idx_i], kv_v[kv_shared_idx_i] if
          is_shared_i == True
        - attention via splash_attention_fn (same code already in
          pallas_attention.py; it only needs Q, K, V, mask)
        - o_proj, rest of layer, *= layer_scalar_i, return (new_hidden,
          kv_k, kv_v), None
     Wrap in `jax.checkpoint(..., policy=ckpt_policy)` for remat.

(B7) Backward-pass correctness. `jax.lax.scan` supports reverse-mode autodiff
     iff the body is a pure function (B6 already requires that). The
     `jax.lax.cond` for is_shared_i must not have a different pytree
     structure on true/false branches -- use `jnp.where` on outputs or
     ensure both branches return a K,V of the same shape.

None of B1-B7 is individually hard; together they are ~300-500 lines of
careful wiring against HF source that changes between transformers releases.
Given the contract-invariance requirement (same output distribution,
loss-sanity-check mandatory), this belongs in a dedicated design pass with
at least one loss-vs-baseline run after every sub-problem.
=============================================================================
"""

from __future__ import annotations

from typing import Any


class _ScanLayersUnsupported(RuntimeError):
  """Raised internally when scan-over-layers cannot wrap Gemma 4 as-is.

  Caught by `register_scan_over_layers` and converted to a warning +
  fall-through to the stock Gemma 4 forward.
  """


def _blocker_list() -> list[str]:
  """Reasons Option A (torchax.train.ScannedModule) doesn't fit Gemma 4.

  Each entry is (short-name, explanation). Printed for the human when
  fallback fires, so we don't lose the diagnosis.
  """
  return [
    "kwargs-only forward: ScannedModule.forward asserts `not kwargs`, but "
    "Gemma4TextModel.forward calls decoder_layer with every arg as kwarg.",
    "state_dict heterogeneity: last 18 layers omit k_proj/v_proj/k_norm/"
    "v_norm (is_kv_shared_layer=True); torch.stack across layers KeyErrors.",
    "per-layer constants baked into self: layer_idx, kv_shared_layer_index, "
    "store_full_length_kv are plain Python ints captured at __init__; scan "
    "body runs against a single prototype layer whose idx is fixed at 0.",
    "mutating dict carry: shared_kv_states is a Python dict whose keys grow "
    "during the layer loop; jax.lax.scan requires a pytree-stable carry.",
    "per-step inputs not supported: Gemma 4 needs per-layer per_layer_input, "
    "position_embeddings[layer_type], attention_mask[layer_type]; "
    "ScannedModule treats all non-hidden args as scan constants.",
  ]


def _raise_option_b_needed() -> None:
  """Placeholder for a future Option B implementation.

  If/when Option B is undertaken, this function becomes the patched forward
  and the monkey-patch site below starts returning True on success. For now
  it always raises so the register call falls back cleanly.
  """
  raise _ScanLayersUnsupported(
    "Gemma 4's 42-layer stack is heterogeneous (kv-shared layers) and "
    "uses dict-carry semantics; torchax.train.ScannedModule (Option A) "
    "cannot wrap it. Option B (custom scan body with stacked weights + "
    "explicit kv carry) is not yet implemented -- see scan_layers.py "
    "header for the sub-problems B1-B7."
  )


def register_scan_over_layers(mesh: "Any") -> bool:
  """Attempt to install a scan-over-layers forward for Gemma 4.

  Always returns (never raises). On success, monkey-patches
  `Gemma4TextModel.forward` and returns True. On *any* failure, logs
  `[scan_layers] fallback to unscanned: <reason>` and returns False so the
  trainer proceeds with HF's stock 42-layer Python loop.

  Args:
    mesh: active `jax.sharding.Mesh`. Accepted for API parity with
      `register_splash_attention(mesh)` / `register_pallas_rmsnorm(mesh)`;
      Option B will need it for `shard_map` around the scan body, but the
      current diagnostic path ignores it.

  Returns:
    True iff a scan-based forward was successfully installed. False means
    HF's default forward is still active.
  """
  del mesh  # unused until Option B

  # Option A probe: can we even import ScannedModule and check the blockers?
  try:
    from torchax.train import ScannedModule  # noqa: F401
  except Exception as exc:
    print(f"[scan_layers] fallback to unscanned: torchax.train.ScannedModule "
          f"import failed ({type(exc).__name__}: {exc})")
    return False

  # Summarize why Option A doesn't apply and then attempt Option B. Option B
  # is currently a stub that always raises; the except below catches it and
  # keeps the trainer running. When Option B is implemented, remove the
  # `_raise_option_b_needed()` call and do the real monkey-patch.
  blockers = _blocker_list()
  print("[scan_layers] Option A (torchax.train.ScannedModule) does not apply "
        "to Gemma 4. Reasons:")
  for i, b in enumerate(blockers, 1):
    print(f"[scan_layers]   ({i}) {b}")

  try:
    _raise_option_b_needed()
  except _ScanLayersUnsupported as exc:
    print(f"[scan_layers] fallback to unscanned: {exc}")
    return False
  except Exception as exc:  # pragma: no cover - defensive
    print(f"[scan_layers] fallback to unscanned: unexpected "
          f"{type(exc).__name__}: {exc}")
    return False

  # Unreachable until Option B lands.
  return True  # pragma: no cover
