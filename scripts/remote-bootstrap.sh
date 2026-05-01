#!/usr/bin/env bash
# Runs on the remote GPU box, executed via:
#   ssh "$user@$ip" 'bash -s' < scripts/remote-bootstrap.sh
#
# Output convention: prints DONE on success, FAIL=<reason> on failure
# (caller greps for these on stdout).
#
# Idempotent: rerunnable. Picks up where it left off.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/PrimaLabs-AI/llm-serving-autoresearch-wiki}"
REPO_DIR="${REPO_DIR:-$HOME/llm-serving-autoresearch-wiki}"
BRANCH="${BRANCH:-mac-driver-multi-vendor}"

step() { echo ">> $*"; }
fail() { echo "FAIL=$1" >&2; exit 1; }

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

step "run setup.sh"
./setup.sh || fail "setup"

step "warm HF cache for ${MODEL:-<unset>}"
if [ -n "${MODEL:-}" ]; then
    if [ -n "${HF_TOKEN:-}" ]; then
        export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    fi
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')" || fail "hf_warm_${MODEL//\//_}"
fi

echo "DONE"
