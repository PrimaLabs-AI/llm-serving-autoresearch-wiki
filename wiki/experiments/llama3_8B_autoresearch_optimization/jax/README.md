# Native-JAX (Flax NNX) Llama 3 8B trainer

Companion folder to [`../torchax/`](../torchax/README.md) — a **pure-JAX** port of the same Llama 3 8B training pipeline. No torchax, no PyTorch at run time. Uses Flax NNX for module construction, `optax.adamw` for the optimizer, and the same splash / tokamax / scan-over-layers kernel stack the torchax sibling validated.

The trainer mirrors the torchax sibling **flag-for-flag** so the two stacks can be A/B'd directly. All design decisions about kernel choice, remat policy, batch/seq sweet-spot, and CE impl are inherited from the torchax experiment series — see [`../program.md`](../program.md) and the [torchax experiments index](../torchax/experiments/README.md).

## Layout

```
jax/
  README.md                this file
  train.py                 native-JAX trainer (fire.Fire)
  data.py                  wikitext-2-raw-v1 packer + fake_dataloader
  splash_attn.py           tokamax-splash dispatch (verbatim from ../torchax/)
  Dockerfile               jax-ai-image base + transformers + flax + tokamax
  model/
    __init__.py            re-exports
    modeling_llama3.py     Flax NNX port of LlamaForCausalLM(+Scan)
    sharding.py            FSDP / TP plans + mesh helpers; SCAN_SHARDING_PLAN
    weight_loader.py       HF safetensors -> NNX param tree (sharded device_put)
  experiments/
    README.md              experiment index for this stack
```

## Architectural decisions

- **Param naming matches HF dot-for-dot.** `model.embed_tokens.weight`, `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`, etc. so the weight loader is a 1:1 mapping with no transpose layer.
- **Linear weight shape is `(out, in)`** (matches `torch.nn.Linear`). Forward computes `x @ weight.T`.
- **RoPE precomputed once** at `LlamaRotaryEmbedding.__init__` (single layer-type — much simpler than Gemma 4's hybrid sliding/full).
- **Attention dispatch via env var** `JAX_ATTENTION_IMPL=splash|xla`. The trainer sets this based on `--use_splash` and registers the mesh with `set_splash_mesh(mesh)` so the kernel's `shard_map` has a concrete `Mesh` (Mosaic custom calls cannot be auto-partitioned).
- **Scan-over-layers** is implemented by stacking each per-layer Linear/RMSNorm weight along a new leading dim (size `num_hidden_layers`). The forward calls `jax.lax.scan` over a functional decoder body that indexes the stacked params layer-by-layer. Sharding plan in `model/sharding.py` (`SCAN_SHARDING_PLAN`) prepends a `None` to every per-layer PartitionSpec so the stack dim stays unsharded.
- **Tokamax CE goes through `shard_map`** with `psum` across `fsdp` (matches the torchax wrapper exactly). `chunked_xla` requires fp32 inputs (validated by torchax exp 62b vs invalid exp 66 — bf16 destroys precision in the lse accumulator).
- **MFU formula** is the MaxText train-step TFLOPs (×3 for fwd + 2× bwd). Identical lines to torchax `train.py` lines 596-632.

## Running

The trainer expects to be invoked with `python -m train` from this folder so `model/` and `data.py` resolve as siblings on `sys.path`.

```bash
WIKI_ROOT="/mnt/disks/persist/torch-tpu/tpu_performance_autoresearch_wiki"
cd "$WIKI_ROOT/wiki/experiments/llama3_8B_autoresearch_optimization/jax"

# canonical: bs=3 seq=8192 fp32-master + bf16-compute, scan + tokamax-CE
python -m train \
    --use_real_data=True \
    --use_splash=True \
    --use_scan=True \
    --use_tokamax_ce=True \
    --tokamax_ce_impl=chunked_xla \
    --tokamax_ce_autotune=True \
    --weights_dtype=fp32 \
    --compute_dtype=bf16 \
    --master_dtype=fp32 \
    --batch_size=3 \
    --seqlen=8192 \
    --train_steps=15
```

Add `--profile_dir=$PROFILE_DIR --profile_step=5` to capture an xprof trace at step 5.

## Flags

Same set as the torchax sibling. Key flags only:

| Flag | Default | Notes |
|---|---|---|
| `--model_id` | `meta-llama/Meta-Llama-3-8B` | HF repo. |
| `--batch_size` | `4` | Per-chip. Global = `batch_size × fsdp`. Sweet-spot at this shape: `bs=3 seq=8192` (torchax exp set). |
| `--seqlen` | `1024` | Smoke-test default; production runs use 8192. |
| `--train_steps` | `15` | Step 0 compiles; steps 0-1 dropped from MFU. |
| `--tp_parallelism` | `1` | 1 = pure FSDP across all chips; >1 = 2D mesh. |
| `--weights_dtype` | `bf16` | `bf16` or `fp32` (master). |
| `--compute_dtype` | `match` | `match` inherits `weights_dtype`; `bf16` with `weights_dtype=fp32` → AMP. |
| `--master_dtype` | `match` | `fp32` forces optimizer mu/nu to fp32 even when weights are bf16. |
| `--use_splash` | `False` | True = splash kernel via env-gated dispatch. Trainer registers the mesh. |
| `--use_scan` | `False` | True = `LlamaForCausalLMScan` (32 layers stacked into one scan body). |
| `--scan_remat_policy` | `nothing_saveable` | `jax.checkpoint_policies` attribute name. |
| `--use_tokamax_ce` | `False` | True = `tokamax.linear_softmax_cross_entropy_loss` (skip lm_head materialization). |
| `--tokamax_ce_impl` | `mosaic_tpu` | `chunked_xla` is the validated faster path (exp 62b). |
| `--tokamax_ce_autotune` | `False` | True = autotune CE block sizes. |
| `--profile_dir` / `--profile_step` | unset | xprof capture at the given step. |

## See also

- [`../program.md`](../program.md) — shared experiment protocol.
- [`../torchax/`](../torchax/) — sibling stack (primary, validated).
- [`experiments/`](experiments/) — JAX-stack experiments index.
