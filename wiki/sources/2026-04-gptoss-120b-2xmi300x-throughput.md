---
title: "GPT-OSS-120B max throughput on 2× MI300X (Hotaisle) — 28,484 tok/s peak with Eagle3 k=3"
type: source
tags: [docs, internal, gpt-oss, mi300x, rocm, vllm-0.18, eagle3, throughput, cost-analysis]
created: 2026-05-04
updated: 2026-05-04
---

The 15-experiment max-throughput study on 2× AMD Instinct MI300X VF (Hotaisle SR-IOV) that establishes the Eagle3 k=3 prior used across the gpt-oss family. **Peak: 28,484 tok/s** on random data with Eagle3 k=3 + `max_num_batched_tokens=16384`; 25,501 tok/s on real ShareGPT data; 4,210 tok/s on prefill-heavy. Pinned to **vLLM v0.18.0 ROCm** because v0.19.0 has a 7× MoE regression (Triton MXFP4 backend replacing the faster v0.18.x kernels). v0.18.1 also works but requires ~33 min AITER JIT compile on first launch.

## Overview

Bench was 2× MI300X 192 GB HBM3 VFs running independent TP=1 servers (2× TP=1 strategy), with Eagle3 RedHatAI draft, KV cache in fp8, GPU_MEM=0.95, block_size=128. Concurrency saturated at 1024/GPU; going higher gave no benefit. CPU + system memory were never the bottleneck. The doc also benchmarks per-million-token cost at $1.99/GPU/hr Hotaisle pricing — beats together.ai/fireworks.ai (15× cheaper) and deepinfra (5× cheaper) for output tokens on chat workloads.

## Key claims

1. **Peak throughput is 28,484 tok/s** on random data with Eagle3 k=3 + batch16k, decode-heavy (256 in / 4096 out). Real ShareGPT lands at 25,501 tok/s with the same flags.
2. **Concurrency saturates at 1024 per GPU** (~21,500 tok/s combined no-spec). 2048 and 3072 give zero benefit. Bottleneck is GPU compute, not scheduling.
3. **Eagle3 k=3 works on v0.18.0 MI300X** — +29% on random, +18.6% on real ShareGPT — contradicting an earlier dual-GPU study on v0.18.1 that found spec decode hurt. v0.18.0 is the version that wins.
4. **vLLM v0.19.0 has a 7× MoE regression on this model** (Triton MXFP4 MoE backend replaces the v0.18.x kernels). 21,497 tok/s → 2,648 tok/s combined at 1024 conc.
5. **ngram k=5 is within 1.5% of Eagle3 k=3 on random data** (28,186 vs 27,778) but loses on real ShareGPT (per prior study, ngram hurts real data −1 to −7%). Ship Eagle3 for production.
6. **Batch-token tuning doesn't help decode-heavy.** 16k/32k/65k showed no improvement vs default 8192. Lever only helps prefill-heavy (+7% per prior study).
7. **Cost competitiveness:** $0.043/M output tokens at 80% util on chat — beats together.ai/fireworks.ai by 14×, deepinfra by 4.4×. Break-even is 1,843 tok/s vs together.ai; we deliver 13.8× headroom.
8. **Parallel CUDA graph capture deadlocks on SR-IOV VFs** — sequential launch (GPU0 ready → GPU1) is required. Adds ~10 min to cold start.

## Key data points

### Top throughput results (combined 2× TP=1, decode-heavy 256 in / 4096 out except sharegpt)

| Experiment | Config | Combined tok/s | vs Baseline |
|---|---|---:|---:|
| P3_eagle3k3_batch16k | Eagle3 k=3 + batch16k | **28,484** | **+32.5%** |
| P3_ngram5 | ngram k=5 | 28,186 | +31.1% |
| P3_eagle3k3 | Eagle3 k=3 (no batch tuning) | 27,778 | +29.2% |
| P4_sharegpt_eagle3_batch16k | Eagle3 k=3 + batch16k, real ShareGPT | **25,501** | +18.6% |
| P1_conc1024 | no spec, 1024 conc/GPU | 21,497 | baseline |
| P5_sustained_4k | no spec, 4096 requests sustained | 21,493 | -0.0% |
| P6_prefill_batch65k | 4096 in / 512 out, batch65k, no spec | 4,210 | -80.4% |

### vLLM version impact

| vLLM Version | Throughput (2× TP=1, 1024 conc) | Notes |
|---|---|---|
| v0.18.0 | **21,497 tok/s** | Used for all results |
| v0.18.1 | ~21,000 tok/s (est.) | Same kernels, 33-min AITER JIT on first start |
| v0.19.0 | **2,648 tok/s** | **7× regression** — Triton MXFP4 MoE backend |

### Recommended decode-heavy config (verbatim)

```bash
vllm/vllm-openai-rocm:v0.18.0
  HIP_FORCE_DEV_KERNARG=1 GPU_MAX_HW_QUEUES=2 TORCH_BLAS_PREFER_HIPBLASLT=1
  ROCR_VISIBLE_DEVICES=<n>
  --model gpt-oss-120b --served-model-name gptoss
  --dtype bfloat16 --trust-remote-code
  --kv-cache-dtype fp8
  --gpu-memory-utilization 0.95
  --block-size 128
  --speculative-config '{"method":"eagle3","model":"<eagle3>","num_speculative_tokens":3}'
  --max-num-batched-tokens 16384
```

Expected: ~25,500 tok/s combined on real chat, $0.043/M output tokens.

### System utilization

GPU busy 100%, peak CPU 37.8%, peak system memory 7.7%, ~750 W per GPU under load. CPU and memory never bound.

## Techniques referenced

- 2× TP=1 strategy (independent replicas, shared LB) on dual-GPU host
- Eagle3 speculative decoding (RedHatAI draft, k=3)
- ngram speculative decoding (k=5)
- KV cache fp8 quantization
- max_num_batched_tokens tuning
- AITER (AMD Composable Kernel) JIT compilation + caching
- Sequential CUDA graph capture (workaround for SR-IOV VF deadlock)
- NUMA balancing disabled (`echo 0 > /proc/sys/kernel/numa_balancing`)
- HIP env tuning (`HIP_FORCE_DEV_KERNARG`, `GPU_MAX_HW_QUEUES`)
- Cost-per-million-token analysis vs hosted API providers

## Gaps & caveats

1. **vLLM v0.19.0 is unusable for 120B until MXFP4 MoE kernels are fixed upstream.** The 7× regression makes self-hosting uncompetitive on this version.
2. **Eagle3 spec result contradicts prior v0.18.1 study.** Treat the +29% / +18.6% as v0.18.0-specific until reproduced on a newer version.
3. **AITER JIT for v0.18.1 is a 33-min cold-start cost** — viable only with persistent volume mount at `/usr/local/lib/python3.12/dist-packages/aiter/jit`.
4. **Parallel server launch deadlocks on Hotaisle SR-IOV VFs.** Sequential launch is required; this is host-class-specific (bare-metal MI300X may not have this).
5. **Cost numbers assume $1.99/GPU/hr Hotaisle pricing** as of April 2026. Other AMD providers will differ.
6. **120B finding does not transfer wholesale to 20B.** The companion [v019 findings](2026-04-gptoss-20b-v019-findings.md) shows that on 20B the v0.19 MoE regression *does not bite*, freeing 20B to run on v0.19. The "OPT" config tuned here for 120B was the wrong default for 20B — that's the entire raison d'être of the 9-config search.

## Connections

- Hypotheses (this doc is the load-bearing source for Eagle3 k=3 priors that the 9-config matrix tests on 20B):
  - [`gptoss-20b-opt-baseline`](../hypotheses/gptoss-20b-opt-baseline.md)
  - [`gptoss-20b-k2-on-h100`](../hypotheses/gptoss-20b-k2-on-h100.md) (challenges k=3 default on 20B)
  - [`gptoss-20b-nospec-on-h100`](../hypotheses/gptoss-20b-nospec-on-h100.md)
  - [`gptoss-20b-batch8k-on-h100`](../hypotheses/gptoss-20b-batch8k-on-h100.md) (challenges batch16k default on 20B)
  - [`gptoss-20b-blk64-on-h100`](../hypotheses/gptoss-20b-blk64-on-h100.md) (challenges block_size=128 default tuned for 120B)
  - [`gptoss-20b-lean-on-h100`](../hypotheses/gptoss-20b-lean-on-h100.md) (the 20B winner that diverges from the 120B OPT)
- Related source: [`gptoss-20b-v019-findings`](2026-04-gptoss-20b-v019-findings.md) — the 20B counterpart that explains why v0.19 is fine on 20B even though it's broken on 120B
- Related source: [`gptoss-20b-config-search-plan`](2026-04-gptoss-20b-config-search-plan.md) (matrix designed against the 120B-tuned defaults from this study)
- Codebase (yet to ingest): `wiki/codebases/vllm-tune.md`
- Hardware: `wiki/hardware/mi300x.md`

## Sources

- `~/vllm-tune/docs/gptoss-120b-max-throughput-2xmi300x.md` (canonical, on-box only — not committed to wiki repo)
