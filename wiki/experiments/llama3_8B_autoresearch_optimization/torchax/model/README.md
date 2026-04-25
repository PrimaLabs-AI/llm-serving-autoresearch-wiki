# `model/` — Llama 3 8B model source

## Where the model code lives

The authoritative PyTorch implementation of Llama 3 8B is in HuggingFace
`transformers` at `src/transformers/models/llama/` — confirmed present on
`main` as of **2026-04-22**:

```
transformers/src/transformers/models/llama/
  configuration_llama.py     # Llama4Config, Llama4TextConfig, Llama4VisionConfig, ...
  modeling_llama.py          # Llama4Model, Llama4ForCausalLM, Llama4ForConditionalGeneration
  modular_llama.py           # shared submodules (attention, MLP, decoder layer)
  processing_llama.py        # multimodal Llama4Processor
  feature_extraction_llama.py
  image_processing_llama.py  # + image_processing_pil_llama.py
  video_processing_llama.py
  convert_llama_weights.py
  __init__.py
```

`__init__.py` in this folder **re-exports** the HF classes. It does not copy
their implementation. This keeps us in sync with upstream bugfixes and avoids
a 2000-line modeling file drifting in-tree. When a hypothesis needs to modify
the forward (e.g., swap in Splash attention, replace RMSNorm, stack scan
over layers), the modification should go in a separate module in this
folder and **subclass** / monkey-patch the upstream class — experiment pages
must document the patch explicitly.

The **DeepMind JAX reference** (`meta-llama/llama3` GitHub) also ships a
Llama 3 8B implementation, including an explicit `gm.nn.Llama4_E4B()` class and
`gm.ckpts.CheckpointPath.GEMMA4_E4B_IT` checkpoint loader. We keep this as
the comparison baseline for the native-JAX port under `../../jax/` — not used
here.

## Llama 3 8B E4B config (from `config.json`, 2026-04-22)

Text-only path (the path this program optimizes):

| Field | Value |
|---|---|
| `model_type` | `llama` |
| `architectures` | `["Llama4ForConditionalGeneration"]` |
| `num_hidden_layers` | **42** |
| `hidden_size` | **2560** |
| `num_attention_heads` | **8** |
| `num_key_value_heads` | **2** |
| `head_dim` | 256 |
| `global_head_dim` | 512 |
| `intermediate_size` | **10240** |
| `vocab_size` | **262144** |
| `max_position_embeddings` | 131072 |
| `sliding_window` | 512 |
| `num_kv_shared_layers` | 18 |
| `rms_norm_eps` | 1e-6 |
| `final_logit_softcapping` | 30.0 |
| `hidden_activation` | `gelu_pytorch_tanh` |
| `tie_word_embeddings` | true |
| `dtype` | `bfloat16` |

RoPE: `rope_theta=500000.0`, `rope_type=default`.

## ASSUMED — verify against real config

- **`Llama4ForCausalLM` is the right text-only head.** The config lists
  `Llama4ForConditionalGeneration` as the canonical architecture
  (multimodal). We load the causal-LM head directly. If that class does not
  exist in the installed transformers version, fall back to
  `Llama4ForConditionalGeneration` and pass text-only inputs; HF will skip
  the vision/audio towers. `train.py` documents this branch.
- **GQA grouping for TP sharding.** We assume Q projections are laid out as
  a single `(hidden, num_heads * head_dim) = (2560, 2048)` linear. Llama 3
  did this; Llama 3 8B config is consistent. **Verify by inspecting a real
  state_dict key.** The regex in `sharding.py` matches substrings
  `q_proj / k_proj / v_proj / o_proj / gate_proj / up_proj / down_proj` —
  HF Llama-family convention. If Llama 3 8B's modular file fused any of these
  (e.g., a single `qkv_proj`), the regex must be updated.
- **LM head weight tying.** `tie_word_embeddings=true` means `lm_head.weight`
  and `embed_tokens.weight` alias the same tensor. JittableModule's
  parameter deduplication handles this (see torchax `interop.py:88-98`) —
  but a sharding rule placed on only one of the two names can be silently
  dropped. We assign the row-sharded spec `P(None, 'tp')` to **both**
  substrings and rely on the dedup to pick the survivor.
- **`final_logit_softcapping=30.0` is implemented as a `tanh` + scale** in
  HF (per Llama 2/3 precedent). This is a pointwise op and does not affect
  sharding, but it **does** show up as a non-trivial HLO node near the end
  of the forward — note for baseline profile analysis.

## See also

- Program page: [`../../../README.md`](../../README.md)
- torchax codebase page: [`../../../../codebases/torchax.md`](../../../../codebases/torchax.md)
- jax-huggingface part 2 (TP recipe):
  [`../../../../sources/2026-jax-huggingface-part-2.md`](../../../../sources/2026-jax-huggingface-part-2.md)
- DeepMind Llama JAX reference: <https://github.com/meta-llama/llama3>
