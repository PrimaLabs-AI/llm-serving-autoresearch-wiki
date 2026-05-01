#!/usr/bin/env bash
# Runs on the remote GPU box, executed via:
#   ssh "$user@$ip" 'SKIP_GIT=1 bash -s' < scripts/remote-bootstrap.sh
#
# Output convention: prints DONE on success, FAIL=<reason> on failure
# (caller greps for these on stdout).
#
# Idempotent: rerunnable. Picks up where it left off.
#
# Two modes (controlled by SKIP_GIT env):
#   SKIP_GIT=1 (default for Mac-driver): the repo is pre-staged at $REPO_DIR
#              by an rsync from the Mac. We just verify it's there and proceed.
#   SKIP_GIT=0:                          we git clone / fetch / pull from
#              GitHub (works only if the repo is public or the box has
#              SSH access via agent forwarding or a deploy key).

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/PrimaLabs-AI/llm-serving-autoresearch-wiki}"
REPO_DIR="${REPO_DIR:-$HOME/llm-serving-autoresearch-wiki}"
BRANCH="${BRANCH:-mac-driver-multi-vendor}"
SKIP_GIT="${SKIP_GIT:-0}"

step() { echo ">> $*"; }
fail() { echo "FAIL=$1" >&2; exit 1; }

if [ "$SKIP_GIT" = "1" ]; then
    step "skip-git mode: verify repo pre-staged at $REPO_DIR"
    [ -d "$REPO_DIR" ] || fail "repo_not_pre_staged"
    [ -f "$REPO_DIR/setup.sh" ] || fail "repo_incomplete"
    cd "$REPO_DIR"
else
    step "ensure git is installed"
    if ! command -v git >/dev/null 2>&1; then
        sudo apt-get update -y >/dev/null 2>&1 || true
        sudo apt-get install -y git || fail "git_install"
    fi

    step "ensure repo is checked out at $REPO_DIR"
    if [ ! -d "$REPO_DIR/.git" ]; then
        git clone "$REPO_URL" "$REPO_DIR" || fail "git_clone"
    fi
    cd "$REPO_DIR"
    git fetch --all --prune || fail "git_fetch"
    git checkout "$BRANCH" || fail "git_checkout"
    git pull --ff-only origin "$BRANCH" || fail "git_pull"
fi

step "verify .env exists"
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "  (copied .env.example to .env — you may need to add HF_TOKEN)"
    else
        fail "no_env_file"
    fi
fi
# Source .env so later steps see HF_TOKEN, MODEL
set -a; source .env; set +a

step "run setup.sh (skip claude/repo/tensorrt)"
# --skip-repo:    we already rsync'd the working tree
# --skip-claude:  Claude lives on the Mac, not the box, in mac-driver mode
# --skip-tensorrt: TensorRT-LLM install is heavy and not needed for round 1
#                  (vLLM is the engine for the top-ranked hypothesis).
#                  Re-enable later via SETUP_FLAGS env override.
./setup.sh ${SETUP_FLAGS:-"--skip-claude --skip-repo --skip-tensorrt"} || fail "setup"

step "warm HF cache for ${MODEL:-<unset>}"
if [ -n "${MODEL:-}" ]; then
    if [ -n "${HF_TOKEN:-}" ]; then
        export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    fi
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')" || fail "hf_warm_${MODEL//\//_}"
fi

echo "DONE"
