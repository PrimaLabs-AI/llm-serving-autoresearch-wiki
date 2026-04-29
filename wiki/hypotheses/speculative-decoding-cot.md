---
title: "Speculative decoding improves decode throughput by 1.5-2x for chain-of-thought workloads"
type: hypothesis
tags: [serving, speculative-decoding, decode-throughput]
model: ""
engine: vllm
workload: chain-of-thought
status: open
expected_gain: "1.5-2x output tokens/s with speculative decoding"
confidence: medium
effort: M
origin: human
created: 2026-04-29
updated: 2026-04-29
---

## Statement

Using speculative decoding with a small draft model (e.g., JackFram/llama-68m as draft for llama-3-8b target) will improve decode throughput by 1.5-2x for chain-of-thought workloads, because the long output sequences give the draft model many opportunities for token-level speculation, and the high acceptance rate for coherent reasoning text means most speculative tokens are accepted.

## Rationale

Chain-of-thought workloads produce long outputs (1K-16K tokens) where decode-time autoregression dominates. Each decode step processes only 1 token, leaving GPU compute underutilized. Speculative decoding uses a small draft model to propose K tokens in parallel, then the target model verifies them in a single forward pass. When the acceptance rate is high (which it tends to be for coherent reasoning), this effectively decodes K tokens per step instead of 1.

The [chain-of-thought workload](../workloads/chain-of-thought.md) has output lengths of 1K-16K tokens — the decode-heavy profile that benefits most from speculation.

## Proposed experiment

1. Run vLLM without speculative decoding — baseline decode throughput
2. Run vLLM with `--speculative-model JackFram/llama-68m --num-speculative-tokens 5`
3. Run the chain-of-thought workload at concurrency levels [16, 32, 64]
4. Measure: output tokens/s, TPOT, acceptance rate, total throughput

Expected delta: 1.5-2x output tokens/s, TPOT reduced proportionally.

## Risks

- Draft model adds memory overhead, reducing KV cache capacity and potentially lowering max concurrency
- Acceptance rate may be lower than expected for diverse reasoning tasks
- Draft model latency may offset gains if the draft is too slow
- vLLM's speculative decoding implementation may not support all model architectures

## Dependencies

- Both target and draft models available
- GPU with enough memory for both models + KV cache
