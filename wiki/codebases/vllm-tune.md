---
title: "vllm-tune"
type: codebase
tags: [codebase, serving, gpt-oss, qwen, vllm, sglang, h100, mi300x, h200, docker, evalscope]
commit: 764c5c1
created: 2026-05-04
updated: 2026-05-04
---

`vllm-tune` is PrimaLabs's production-grade serving-optimization research suite. It is the engine that produces every benchmark, finding, and shipping artifact this wiki points at. The repo lives at `~/vllm-tune` on every PrimaLabs GPU box (currently `h100-1`); it is the canonical source of truth for serving runtime configuration, container deployment, and load-generation tooling. The wiki is the research layer over `vllm-tune`: PICK turns choose what to test next, RUN turns invoke `vllm-tune`'s scripts, results flow back as wiki experiments.

## Overview

A single repository covering three levels of work:

1. **Active research suites** for specific (model, hardware) targets — `gpt-oss-120B` on 2×/4× MI300X and 4× H100; `gpt-oss-20B` config search on H100 80 GB / 94 GB / 141 GB; Qwen2.5-32B Bayesian search on H200 + MI300X (designed, not yet executed).
2. **Production deployment artifacts** — three Docker SKUs already pushed (`primalabs/gptoss-vllm-{baseline,optimized,decode}:v0.19.0`), each a 4× TP=1 + nginx LB topology.
3. **Reusable infrastructure** — Docker compose (NVIDIA + ROCm), evalscope-driven load generators, lossless-gate validation harness (lm-eval on GSM8K / HellaSwag / MMLU @ ≤0.1% deviation), DeepHyper Bayesian optimizer integration, monitoring stack (Prometheus + Grafana).

## Architecture

Three layers, each with clear inputs and outputs:

| Layer | Role | Key paths |
|---|---|---|
| **Server side** | Container per GPU (TP=1), behind an nginx LB on a single host port | `deploy/<sku>/Dockerfile`, `deploy/<sku>/entrypoint.sh`, `deploy/<sku>/nginx.conf` |
| **Orchestrator** | Drive a config matrix: per cell, tear down the previous container, launch with new flags, wait for `/health`, run evalscope sweep, snapshot `/metrics`, tear down | `scripts/experiments/<study>/run_matrix.sh`, `scripts/launch/launch_*.sh` |
| **Load generator** | Concurrency × workload sweep against any OpenAI-compatible endpoint (`self`, `fireworks`, `together`, etc.) | `scripts/benchmark/sweep_api_providers_evalscope.py`, `scripts/benchmark/bench_api_providers.py` |

Server and load-gen can run on the same box (single-GPU studies like the gpt-oss-20B matrix) or split (4× MI300X server + separate Mac/laptop client).

## Key abstractions

| Abstraction | What it is | File |
|---|---|---|
| **Config matrix** | Bash function `set_config_flags` that maps an ID (e.g. `OPT`, `LEAN`) to a vLLM flag list. Each ID is a falsifiable hypothesis | `scripts/experiments/gptoss20b_config_search/run_matrix.sh` |
| **Per-replica peak concurrency** | Map workload → concurrency representing the per-GPU equivalent of the 4-GPU peak it landed at in the baseline-vs-optimized findings | same file, `WORKLOAD_CONC` array |
| **evalscope sweep** | One row per `(provider, workload, concurrency)` cell; column schema documented in [`sources/2026-04-gptoss-20b-v019-findings.md`](../sources/2026-04-gptoss-20b-v019-findings.md) | `scripts/benchmark/sweep_api_providers_evalscope.py` |
| **Lossless gate** | lm-eval-harness on GSM8K, HellaSwag, MMLU (200 samples each), require ≤0.1% deviation vs reference | `scripts/optimize/validate_config.py` |
| **DeepHyper search** | Bayesian optimization over `config/search_space.yaml` to suggest next config to try | `scripts/optimize/deephyper_search.py` |
| **Production SKU** | A complete Dockerfile + entrypoint + nginx config that bakes in a specific vLLM flag set, ready to push | `deploy/<sku>/` |

## Entry points

| Use case | Command |
|---|---|
| Run a single config from a search matrix | `DOCKER="sudo docker" bash scripts/experiments/gptoss20b_config_search/run_matrix.sh --vllm-version v0.19.0 --gpu-mem 0.85 --configs <ID>` |
| Run all configs in a matrix | drop `--configs` filter |
| Resume a partial sweep | rerun with same `OUT_DIR` env (matrix re-uses existing per-config CSVs) |
| Run a one-off concurrency sweep against any endpoint | `python scripts/benchmark/sweep_api_providers_evalscope.py --self-urls $URL --providers self --workloads decode prefill sharegpt --concurrency 1 8 64 256 1024 --tokenizer-path <path> --out <csv>` |
| Build a SKU image | `cd deploy/<sku> && docker build -t primalabs/<sku>:<tag> .` |
| Launch a SKU locally | `bash scripts/launch/launch_vllm.sh optimized off` |
| Bayesian search for next config to try | `python scripts/optimize/deephyper_search.py --runtime vllm --max-evals 100` |
| Validate a config preserves accuracy | `python scripts/optimize/validate_config.py --runtime vllm --config <name>` |

## Dependencies

- **Docker + NVIDIA Container Toolkit** (or ROCm Container Toolkit on AMD)
- **vLLM** v0.18.0 (gpt-oss-120B path, MXFP4 regression in 0.19+) or v0.19.0 (gpt-oss-20B path; the regression is 120B-specific and v0.19's KV-budgeting fix unlocks default `gpu_memory_utilization=0.9` on 80 GB SXM)
- **SGLang** v0.4.2+ (NVIDIA: `lmsysorg/sglang:latest`; ROCm: `lmsysorg/sglang:v0.4.5-rocm630`)
- **TensorRT-LLM** for the H200 / B200 paths (NVIDIA-only — `vllm-tune`'s `vllm-decode` SKU uses Amazon P-EAGLE which still requires NVIDIA tooling)
- **evalscope[perf]** ≥1.6.x — load generator
- **lm-evaluation-harness** 0.4.4 — lossless gate
- **DeepHyper** + `aiohttp` + `numpy` — for the Bayesian optimizer
- **Prometheus + Grafana** containers (`primalabs/benchmark-prometheus`, `primalabs/benchmark-grafana`) — monitoring stack

## Performance-relevant surfaces

The flags this repo maps to falsifiable claims. These are the surfaces the wiki's hypotheses operate on:

| Knob | Where set | Effect studied |
|---|---|---|
| `--gpu-memory-utilization` | `run_matrix.sh:GPU_MEM`, `deploy/*/entrypoint.sh:GPU_MEM` | KV cache budget. 0.85 needed on H100 80 GB SXM under vLLM 0.18; 0.9-0.95 OK on v0.19 with budgeting fix |
| `--block-size` | `run_matrix.sh` BLK64 vs OPT (128) | block_size=128 was tuned for 120B's wider GQA; on 20B the KV per-block overhead may dominate |
| `--max-num-batched-tokens` | `run_matrix.sh` BATCH8K (8192) vs OPT (16384) | 16k matmul flop-bound on 120B; finishes too fast on 20B → BATCH8K may be net better on 20B |
| `--kv-cache-dtype fp8` | `run_matrix.sh` NOFP8KV vs OPT | KV not the bottleneck on 20B at 80 GB; quant cost may exceed capacity gain |
| `--dtype bfloat16` | `run_matrix.sh` NOBF16 vs OPT | Drop bf16 promotion; let MXFP4 stay native |
| `--speculative-config method=eagle3 num_speculative_tokens=K` | `run_matrix.sh` `SPEC_K3`, `SPEC_K2` | Eagle3 acceptance decays through positions; OPT round 7 measured pos0:pos1:pos2 = 100:77:62; trade k=3 → k=2 saves ~9% draft compute for ≤3% acceptance loss |
| `--tensor-parallel-size` | per launch_*.sh | Per-replica TP=1 on H100/MI300X is the production pattern (one container per GPU + nginx LB) |
| `--enable-chunked-prefill`, `--enable-prefix-caching` | (referenced in Qwen-32B Part A search space; not in gpt-oss matrix) | Workload-dependent; planned for Qwen2.5-32B Part A study |

Container-level surfaces:

| Knob | Where set | Effect studied |
|---|---|---|
| vLLM version pin | `run_matrix.sh:VLLM_VERSION`, `deploy/*/Dockerfile:FROM` | 120B needs v0.18.0 (v0.19 has 7× MXFP4 MoE regression); 20B uses v0.19.0 (regression is 120B-specific) |
| `block_size` × `kv-cache-dtype fp8` interaction | tuning by hand | fp8 KV at blk128 vs blk64 pivots on 20B vs 120B |
| `served-model-name` | per entrypoint | Affects metric labels and sweep wrapper expectations |
| `HIP_FORCE_DEV_KERNARG=1`, `GPU_MAX_HW_QUEUES=2`, `TORCH_BLAS_PREFER_HIPBLASLT=1` | ROCm entrypoints | Required for the MI300X 25,501 tok/s ShareGPT peak in [`sources/2026-04-gptoss-120b-2xmi300x-throughput.md`](../sources/2026-04-gptoss-120b-2xmi300x-throughput.md) |

## Notable files

- `experimental-plan.md` — the umbrella Qwen2.5-32B Part A/B/C study spec (designed, not executed)
- `docs/gptoss-20b-v019-config-search-findings.md` — winning config (LEAN); ingested as [`sources/2026-04-gptoss-20b-v019-findings.md`](../sources/2026-04-gptoss-20b-v019-findings.md)
- `docs/gptoss-20b-config-search-plan.md` — the 9-config matrix plan; ingested as [`sources/2026-04-gptoss-20b-config-search-plan.md`](../sources/2026-04-gptoss-20b-config-search-plan.md)
- `docs/gptoss-20b-80gb-h100-oom-finding.md` — vLLM 0.18 sampler-warmup OOM bug; ingested as [`sources/2026-04-gptoss-20b-h100-oom.md`](../sources/2026-04-gptoss-20b-h100-oom.md)
- `docs/gptoss-120b-max-throughput-2xmi300x.md` — 28,484 tok/s peak; ingested as [`sources/2026-04-gptoss-120b-2xmi300x-throughput.md`](../sources/2026-04-gptoss-120b-2xmi300x-throughput.md)
- `CLIENT_HANDOFF*.md` — split-host (server-vs-client) sweep runbooks
- `config/search_space.yaml`, `config/vllm_params.yaml`, `config/sglang_params.yaml` — DeepHyper search space + per-runtime parameter sets

Other 13 docs in `~/vllm-tune/docs/` (gpt-oss-120B clock-frequency, memory-bottleneck, dual-MI300X, API-provider-benchmark, sharegpt-specdec, speculative-decoding, stacked-optimization, cost-comparison, mi300x-findings, mi300x-api-comparison-runbook, 4×H100-vs-fireworks, 4×MI300X-vs-4×H100, 70B-llama3-mi300x-findings) — not yet ingested as wiki sources; pending as priors-mature.

## Connections

- **Engines:** [vLLM](../engines/vllm.md), [SGLang](../engines/sglang.md), [TensorRT-LLM](../engines/tensorrt-llm.md). vllm-tune drives all three (TRT-LLM via the H200 path that's not yet executed).
- **Hardware:** [H100](../hardware/h100.md), [B200](../hardware/b200.md), [MI300X](../hardware/mi300x.md). All three are first-class targets in `vllm-tune`'s study set.
- **Hypotheses:** the gpt-oss-20B matrix lives in the wiki as 8 hypotheses (`hypotheses/gptoss-20b-{base,opt,k2,nospec,blk64,batch8k,nofp8kv,nobf16,lean}-on-h100.md`). Round-7 [OPT experiment](../experiments/2026-05-04-gptoss20b-h100-opt.md) was the first of these executed via the autoresearch loop.
- **Sources:** the four ingested docs above; thirteen more pending.

## Sources

- On-box checkout: `~/vllm-tune/` at commit `764c5c1` (the SHA recorded in this page's frontmatter), tip date 2026-04-29
- Top-level docs: `~/vllm-tune/{README,QUICKSTART,CLIENT_HANDOFF,CLIENT_HANDOFF_H100,CLIENT_HANDOFF_MI300X_QUAD}.md`
- Repository origin: `https://github.com/pbalapra/vllm-tune.git` (private)

*Updated when `vllm-tune` advances materially. Recapture commit SHA in frontmatter on each refresh.*
