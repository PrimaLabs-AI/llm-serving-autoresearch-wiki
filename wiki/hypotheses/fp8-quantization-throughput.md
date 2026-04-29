---
title: "FP8 quantization increases max concurrency by 50-80% with <1% quality degradation"
type: hypothesis
tags: [serving, quantization, fp8, concurrency]
model: ""
engine: vllm
workload: multi-turn-agentic
status: open
expected_gain: "50-80% higher max concurrency before OOM"
confidence: high
effort: S
origin: human
created: 2026-04-29
updated: 2026-04-29
---

## Statement

Using FP8 quantization (via `--quantization fp8 --dtype float16`) will increase the max achievable concurrency by 50-80% for multi-turn agentic workloads on Hopper GPUs, with less than 1% quality degradation, because the reduced model weight and KV cache memory footprint allows more concurrent sequences before hitting GPU memory limits.

## Rationale

The primary concurrency bottleneck in LLM serving is KV cache memory. Each concurrent request allocates KV cache proportional to sequence length. FP8 quantization halves the memory footprint of model weights (from bf16) and can also compress KV cache to FP8, roughly doubling the available KV cache memory. For Hopper GPUs (H100, H200), FP8 matmul is hardware-accelerated so there is minimal throughput penalty.

For multi-turn agentic workloads where context grows across turns, KV cache per request is large, making memory the binding constraint.

## Proposed experiment

1. Run vLLM with bf16 — find max concurrency before OOM
2. Run vLLM with `--quantization fp8 --dtype float16` — find max concurrency before OOM
3. At each level, measure throughput and TTFT
4. Spot-check output quality on 20-50 samples vs. bf16 baseline

Expected delta: 50-80% higher max concurrency, throughput at equivalent concurrency within 5% of bf16, quality degradation < 1%.

## Risks

- FP8 quality degradation may exceed 1% for some model families
- Not all GPU architectures support FP8 (requires Hopper or newer)
- vLLM's FP8 implementation may not cover all model architectures
- KV cache quantization to FP8 may have higher quality impact than weight-only quantization

## Dependencies

- Hopper-class GPU (H100, H200)
- Model with FP8 checkpoint or calibration support
