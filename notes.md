https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f

You are the maintainer of a personal knowledge base about **Tensor Processing Units (TPUs)**: hardware architecture, compilers, ML frameworks, research papers, and ML workloads on TPU. You write and maintain every file in the wiki. The main focus is to automating model conversion to make them run efficiently on TPUs and optimizing model performance. The human curates sources, directs analysis, and asks questions. You do the rest.

using git submodule add https://github.com/vlasenkoalexey/xprof-mcp under raw/code, update readme
using git submodule add https://github.com/google/torchax under raw/code, update readme
using git submodule add https://github.com/openxla/tokamax under raw/code, update readme
using git submodule add https://github.com/openxla/stablehlo under raw/code, update readme
using git submodule add https://github.com/openxla/xprof under raw/code, update readme
using git submodule add https://github.com/jax-ml/scaling-book under raw/code, update readme
using git submodule add https://github.com/karpathy/autoresearch  under raw/code, update readme

Injest https://huggingface.co/spaces/nanotron/ultrascale-playbook, save pictures references on the page
Injest https://github.com/qihqi/learning_machine/tree/main/jax-huggingface


Connect Obsidian
This is the crucial step to avoid the extra subfolder trap.
- Open the Obsidian app.
- On the launch screen, DO NOT click "Create new vault."
- Click "Open folder as vault" (usually the second option).
- Navigate to the my-llm-wiki folder you just created and select it.

Injest sources
injest repos under raw/code, especially pay attention at documentation related to TPU models profiling and optimization, do a deeper injestion on these topics

Idea is to import google/gemma-4-E4B, make it to work on TPU using torchax, convert model to JAX and apply autoresearch approach to optimize its performance.
Unlike with original autoreserch approach we are not optimizing model quality metrics, we are optimizing model performance (step time, mfu, tps).

Populate tpu_performance_autoresearch_wiki/wiki/experiments/gemma4_autoresearch_optimization using this information.

Find gemma model code on https://huggingface.co/google/gemma-4-E4B (also can use https://github.com/google-deepmind/gemma/tree/main for references). Import model code under tpu_performance_autoresearch_wiki/wiki/experiments/gemma4_autoresearch_optimization/torchax.

Create torchax trainer that can train for finetune model from checkpoint. Use internal knowledge for information how to partition model and get it working using torchax.

Trainer can use wiki dataset by default. It should allow configuring nubmer of steps to run, and enable profiling api to dump profiles for specified steps.

