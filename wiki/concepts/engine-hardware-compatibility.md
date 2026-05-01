---
title: "Engine × Hardware Compatibility"
type: concept
tags: [generated, compatibility]
created: 2026-05-01
updated: 2026-05-01
---

> **Generated** by `scripts/regenerate-compat-table.py` from `wiki/engines/*.md`
> `supported_hardware:` frontmatter. **Do not edit by hand** — edits will be
> overwritten on the next `LINT`.

The orchestration loop's scheduler reads each engine's `supported_hardware`
field directly. This page is a human-readable readout of that data.

| Engine | h100 | b200 | mi300x |
|---|---|---|---|
| SGLang | ✓ | ✓ | ✓ |
| TensorRT-LLM | ✓ | ✓ | ✗ |
| vLLM | ✓ | ✓ | ✓ |
