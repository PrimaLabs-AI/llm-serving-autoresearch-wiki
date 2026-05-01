# Host Onboarding

How to add a fresh GPU box to the autoresearch loop.

## Prerequisites

You've rented a GPU instance from a cloud provider (Lambda Cloud, RunPod, Crusoe, TensorWave, etc.) and have:
- A public IP or DNS name
- An SSH user (usually `ubuntu` or `root`)
- A private key file on your Mac
- The image is Ubuntu 22.04+ with NVIDIA drivers (or ROCm for AMD)

## Add the box to your registry

If this is your first box:

```bash
cp .hosts.example.toml .hosts.toml
```

Edit `.hosts.toml` and add a block per host:

```toml
[hosts.h100-1]
ip        = "203.0.113.42"
user      = "ubuntu"
ssh_key   = "~/.ssh/lambda_key"
vendor    = "nvidia"             # or "amd"
hardware  = "h100"               # slug from wiki/hardware/
gpu_count = 8
```

The `hardware` slug **must** match a page under `wiki/hardware/<slug>.md`. The currently supported slugs are `h100`, `b200`, `mi300x`. To add a new one, also create a wiki page for it.

## Verify the box is reachable

```bash
python3 scripts/host_registry.py reachable h100-1
echo $?    # 0 = reachable, 1 = unreachable
```

## Run setup remotely

The first time the loop dispatches a round to a host, it will SSH in and run setup automatically. To trigger setup eagerly without running a round:

```bash
./scripts/remote-setup.sh h100-1
```

This SSH bootstrap clones the repo, detects the vendor (`nvidia-smi` vs `rocm-smi`), and runs the appropriate setup script (`setup-cuda.sh` or `setup-rocm.sh`). The bootstrap is mechanical — no Claude session involved.

Setup state is tracked in `.host-state.toml` (gitignored). Possible states:
- `pending` — bootstrap not yet run
- `running` — bootstrap in progress
- `ready` — bootstrap done; host can take rounds
- `failed` — bootstrap errored; check `last_error` and rerun
- `unreachable` — SSH failed reachability check; check IP / key / firewall

## What the box ends up with

After successful setup:
- Repo cloned at `~/llm-serving-autoresearch-wiki` (the bootstrap uses `git clone`)
- Engines installed: `vllm`, `sglang`, plus `tensorrt-llm` on NVIDIA only
- HF cache warmed for the model in `.env` (if `HF_TOKEN` is set)
- The box never reads or writes the wiki — it only emits `raw/benchmarks/<run>/` artifacts which the Mac rsync's back

## Replacing or destroying a box

When you tear down a box, just remove its block from `.hosts.toml` (and optionally `rm` the corresponding entry from `.host-state.toml`). The next loop run will simply not dispatch to it.

If a box's IP changes (cloud reboot), update `.hosts.toml`. The state stays valid — only the routing changed.

## Wiki write conflicts (FYI)

If you edit `wiki/index.md` (or any other wiki file) while a loop is running, the round's `git commit` step will fail and the loop will halt cleanly with a conflict message. Resolve the conflict, then rerun the same `./run_loop.sh …` command — already-committed rounds aren't repeated.
