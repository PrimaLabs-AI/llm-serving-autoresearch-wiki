---
title: "Chunked prefill improves high-concurrency TTFT by 25-40% without throughput loss"
type: hypothesis
tags: [serving, chunked-prefill, concurrency]
model: ""
engine: vllm
workload: parallel-tool-use
status: open
expected_gain: "25-40% TTFT reduction at concurrency>64"
confidence: medium
effort: S
origin: human
created: 2026-04-29
updated: 2026-04-29
hardware: any
---

## Statement

Enabling chunked prefill (`--enable-chunked-prefill`) in vLLM will reduce TTFT p99 by 25-40% at concurrency levels above 64, with no throughput regression, because it allows the scheduler to interleave prefill and decode steps rather than blocking decode until the full prefill completes.

## Rationale

Without chunked prefill, a long prefill request occupies the GPU exclusively, blocking all in-progress decode requests. Under high concurrency, multiple long prefills can serialize, causing decode stalls and high tail latency. Chunked prefill breaks the prefill into smaller chunks that can be interleaved with decode steps, reducing head-of-line blocking.

The [parallel tool use workload](../workloads/parallel-tool-use.md) has 4-32 parallel requests hitting simultaneously with 8K token prompts — this is exactly the burst pattern that causes prefill-induced decode stalls.

## Proposed experiment

1. Launch vLLM without chunked prefill — baseline
2. Launch vLLM with `--enable-chunked-prefill --max-num-batched-tokens 4096`
3. Run the parallel tool use workload at concurrency levels [32, 64, 128, 256]
4. Measure: TTFT p50/p99, throughput (tokens/s), TPOT p99

Expected delta: 25-40% TTFT p99 reduction at concurrency > 64, throughput within 5% of baseline.

## Risks

- Chunked prefill adds scheduling overhead that may reduce throughput at low concurrency
- The `max_num_batched_tokens` tuning interacts with chunk size — wrong value could hurt both metrics
- Some attention backends may not support chunked prefill efficiently

## Dependencies

- GPU server with vLLM installed
- Model specified
