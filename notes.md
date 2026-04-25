https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f

You are the maintainer of a personal knowledge base about **Tensor Processing Units (TPUs)**: hardware architecture, compilers, ML frameworks, research papers, and ML workloads on TPU. You write and maintain every file in the wiki. The main focus is to automating model conversion to make them run efficiently on TPUs and optimizing model performance. The human curates sources, directs analysis, and asks questions. You do the rest.

using git submodule add https://github.com/vlasenkoalexey/xprof-mcp under raw/code, update readme
using git submodule add https://github.com/google/torchax under raw/code, update readme
using git submodule add https://github.com/openxla/tokamax under raw/code, update readme
using git submodule add https://github.com/openxla/stablehlo under raw/code, update readme
using git submodule add https://github.com/openxla/xprof under raw/code, update readme
using git submodule add https://github.com/jax-ml/scaling-book under raw/code, update readme
using git submodule add https://github.com/karpathy/autoresearch  under raw/code, update readme
using git submodule add https://github.com/linhkid/pallas-forge under raw/code, update readme
using git submodule add https://github.com/jax-ml/jax under raw/code, update readme. 
using git submodule add https://github.com/AI-Hypercomputer/tpu-recipes/tree/main raw/code, update readme. 


Injest https://huggingface.co/spaces/nanotron/ultrascale-playbook, save pictures references on the page
Injest https://github.com/qihqi/learning_machine/tree/main/jax-huggingface

search internet for all sources where pallas kernels code can be found. Do an extensive search in research
> returns list, saved under wiki/analyses/2026-04-23-pallas-kernel-source-survey.md

now based on this information reaserch every repo you listed and create a directory of available pallas kernels with references to source code, stability statetements, performance statements, application usecases, references where kernel is used



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

Now convert model to jax trying to follow model definition as close as possible where it make sense. Add new converted model to /mnt/disks/persist/torch-tpu/tpu_performance_autoresearch_wiki/wiki/experiments/gemma4_autoresearch_optimization/jax folder. Also create a trainer that should have same functinality as a torchax trainer that was created previously and have similar flags.

ok, now as you run some experiments and have a lot of context about this repo, I'd like you to update main README.md file for this repo. I'd like it to be like marketing page trying to advertise and sell idea of automatic model optimization and while also maintaining access to the codebase.

Idea is following, we want to automate TPU model performance optimization. For that we can follow Karpathy's autoresearch paper. In order for LLM to be able efficiently automate TPU performance optimization it needs to have an ability to profile model. For that we created xprof_mcp server. xprof_mcp server in fact goes beyond just profiling, it can also analyze HLO dumps and combine information from profiles with hlo dumps to backtrack how ops were generated and optimized by xla. 
But even that is not enough, LLM by default do not have sufficient information out of the box on TPUs and TPU model optimization. To solve that we leverage another Karpathy idea - wiki generation. In Wiki we can injest relevant information about how to use xprof and xprof mcp server. Relevant information about TPU model optimization tricks. Put overview of pallas kernels that other repo is using. And most imporantly we can directly add and injest source code for the codebase where model is hosted, source code for the framework that is running model (Jax or torchax). The would give LLM a comprehensive view of the entire project, access to the model, access to the framework. This would allow LLM to be able backtrace profile and HLO information to the model and underlying framework. This expands LLM ability to not just debug and improve model code, but also debug issues in the underlying framework.

We also would need to generate few images for the repo, see if there are skills that we can install to do that.

Ok, give it a try. This is very important, do your best to make this project/idea to look exciting and billiant. You can highlight that with this project model optimizations become so easy that it feels like cheating. 
You can also add that this immensly speeds up ramping up with model optimizaton process, and it is possible to ask LLM questions about profile model behavior and ask it to explore ideas on its own. 