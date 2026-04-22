https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f

You are the maintainer of a personal knowledge base about **Tensor Processing Units (TPUs)**: hardware architecture, compilers, ML frameworks, research papers, and ML workloads on TPU. You write and maintain every file in the wiki. The main focus is to automating model conversion to make them run efficiently on TPUs and optimizing model performance. The human curates sources, directs analysis, and asks questions. You do the rest.

using git submodule add https://github.com/vlasenkoalexey/xprof-mcp under raw/code, update readme
using git submodule add https://github.com/google/torchax under raw/code, update readme
using git submodule add https://github.com/openxla/tokamax under raw/code, update readme
using git submodule add https://github.com/openxla/stablehlo under raw/code, update readme
using git submodule add https://github.com/openxla/xprof under raw/code, update readme
using git submodule add https://github.com/jax-ml/scaling-book under raw/code, update readme
using git submodule add https://github.com/karpathy/autoresearch  under raw/code, update readme


Connect Obsidian
This is the crucial step to avoid the extra subfolder trap.
- Open the Obsidian app.
- On the launch screen, DO NOT click "Create new vault."
- Click "Open folder as vault" (usually the second option).
- Navigate to the my-llm-wiki folder you just created and select it.

Injest sources