---
title: "Qwen2.5-7B-Instruct"
type: model
tags: [model, qwen, dense, h100]
target_hardware: [h100, h200, b200, mi300x]
created: 2026-05-04
updated: 2026-05-04
---

`Qwen/Qwen2.5-7B-Instruct` — Alibaba's 7.6B-parameter instruction-tuned dense transformer. Public, ungated. Used here as the **first non-gpt-oss target** in the autoresearch program — validates that the loop generalizes beyond a single model family.

## Target metrics

Per-replica on a single H100 80 GB SXM, vLLM 0.19.0, BF16. To be filled by experiments.

| Workload | Conc | Output tok/s | TTFT p50 | TPOT p50 | E2E p99 | KV % | Source |
|---|---:|---:|---:|---:|---:|---:|---|
| decode (256 in / 4096 out) | 256 | TBD | TBD | TBD | TBD | TBD | (round N: BASE — pending) |
| prefill (4096 in / 512 out) | 1024 | TBD | TBD | TBD | TBD | TBD | (round N: BASE — pending) |
| sharegpt (~175 in / ~2K out) | 512 | TBD | TBD | TBD | TBD | TBD | (round N: BASE — pending) |

## Hardware

| Slug | Status | Comment |
|---|---|---|
| [h100](../hardware/h100.md) | active | 80 GB SXM. Model weights ~14 GB BF16 → ~62 GB free for KV cache. Headroom is generous; concurrencies can be ~2× the gpt-oss-20B numbers. |
| [h200](../hardware/h200.md) | future | 141 GB. Even more KV headroom; spec decoding cells become viable at higher concurrencies. |
| [b200](../hardware/b200.md) | future | FP4 support; relevant if Qwen 2.5 7B has an FP4 quantized variant. |
| [mi300x](../hardware/mi300x.md) | future | ROCm path; same model fits trivially. |

## How to run

Single H100 80 GB, BASE config (stock vLLM 0.19.0). The model is downloaded to `/srv/models/Qwen/Qwen2.5-7B-Instruct/` on the box.

```bash
sudo docker run -d --name qwen7b-bench --gpus all --ipc=host --shm-size 16g \
    -p 8000:8000 -v /srv/models:/models:ro \
    --entrypoint python3 \
    vllm/vllm-openai:v0.19.0 \
    -m vllm.entrypoints.openai.api_server \
    --model /models/Qwen/Qwen2.5-7B-Instruct \
    --served-model-name qwen \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9
```

Health: `curl -fsS http://localhost:8000/health`. Tear down with `sudo docker rm -f qwen7b-bench`.

The bench's `bench/scripts/experiments/gptoss20b_config_search/run_matrix.sh` is **not** usable here — it hardcodes the gpt-oss-20B model and the Eagle3 draft path. Use direct `docker run` + `bench/scripts/benchmark/sweep_api_providers_evalscope.py` for now. (Future cleanup: parameterize `run_matrix.sh` over model + draft.)

## Properties

- **Architecture**: Qwen2 dense transformer (no MoE), GQA, RoPE, SwiGLU
- **Parameters**: 7.6B
- **Hidden size**: 3,584
- **Layers**: 28
- **Attention heads**: 28 query, 4 KV (GQA ratio 7:1)
- **Vocab**: 152,064 (tiktoken-style BPE)
- **Native context**: 32K (128K via YaRN — not used here)
- **Weight size**: ~14 GB BF16 (4 safetensors shards), ~7.6 GB FP8 (not yet released by Alibaba), ~3.8 GB GPTQ-4bit (community)

## Baseline

(Empty — round N BASE pending.)

## Current best

(Empty — round N BASE pending.)

## Known bottlenecks

(Empty until first experiment lands. Hypotheses below predict where the bottleneck will sit on H100 80 GB.)

## Open hypotheses

| Hypothesis | Status | Predicted |
|---|---|---|
| [BASE on H100](../hypotheses/qwen2.5-7b-base-on-h100.md) | open | establish floor on three workloads |

## Retired hypotheses

(none yet)

## History

| Date | Event |
|---|---|
| 2026-05-04 | Model page created. Weights downloaded to `/srv/models/Qwen/Qwen2.5-7B-Instruct/` on `h100-1`. First hypothesis (BASE) filed. |

## See also

- [Qwen2.5-7B Instruct on Hugging Face](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) (model card, downloads)
- [vLLM engine page](../engines/vllm.md) (the only engine targeting Qwen on this box today)
- [H100 hardware page](../hardware/h100.md) (specs, optimization gotchas)
- [vllm-tune codebase](../codebases/vllm-tune.md) (bench infrastructure surface)

## Sources

- [`https://huggingface.co/Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) — official model card and config
- [Qwen2.5 technical report](https://arxiv.org/abs/2412.15115) — architecture details
