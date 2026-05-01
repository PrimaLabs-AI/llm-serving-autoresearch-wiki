#!/usr/bin/env bash
# Mac-side wrapper that ships scripts/remote-bootstrap.sh to a host
# and runs it. Updates .host-state.toml around the dispatch.
#
# Usage: ./scripts/remote-setup.sh <host-name>

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <host-name>" >&2
    exit 2
fi

HOST="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Resolve host details
ssh_target="$(python3 "$SCRIPT_DIR/host_registry.py" --root "$REPO_DIR" get "$HOST" ssh_target)"
ssh_key="$(python3 "$SCRIPT_DIR/host_registry.py" --root "$REPO_DIR" get "$HOST" ssh_key)"
ssh_key_expanded="${ssh_key/#\~/$HOME}"

echo "[remote-setup] $HOST → $ssh_target"
python3 "$SCRIPT_DIR/host_registry.py" --root "$REPO_DIR" state "$HOST" --set running

ssh_opts=(
    -i "$ssh_key_expanded"
    -o StrictHostKeyChecking=accept-new
    -o ControlMaster=auto
    -o "ControlPath=$HOME/.ssh/cm-%r@%h:%p"
    -o ControlPersist=10m
)

# Stage 1: rsync the Mac's repo to the box. The box is a stateless worker
# (per the design); the Mac is sole authoritative state. We exclude .git
# (large, unneeded), raw/ (multi-GB benchmark/profile artifacts), .venv,
# and the local registry files (.hosts.toml, .host-state.toml).
echo "[remote-setup] $HOST: rsync repo → $ssh_target:llm-serving-autoresearch-wiki/"
rsync -az --delete \
    --exclude=.git \
    --exclude=.venv \
    --exclude=__pycache__ \
    --exclude=.pytest_cache \
    --exclude=.hosts.toml \
    --exclude=.host-state.toml \
    --exclude=raw/profiles \
    --exclude=raw/benchmarks \
    --exclude=raw/loops \
    --exclude=raw/code \
    --exclude=raw/sources \
    -e "ssh ${ssh_opts[*]}" \
    "$REPO_DIR/" "$ssh_target:llm-serving-autoresearch-wiki/"

# Stage 2: stream the bootstrap to the box; capture stdout and stderr separately
out_file="$(mktemp)"
err_file="$(mktemp)"
trap 'rm -f "$out_file" "$err_file"' EXIT

set +e
ssh "${ssh_opts[@]}" \
    "$ssh_target" 'SKIP_GIT=1 bash -s' < "$SCRIPT_DIR/remote-bootstrap.sh" \
    > "$out_file" 2> "$err_file"
rc=$?
set -e

cat "$out_file"
cat "$err_file" >&2

if [ $rc -eq 0 ] && grep -q "^DONE$" "$out_file"; then
    python3 "$SCRIPT_DIR/host_registry.py" --root "$REPO_DIR" state "$HOST" --set ready
    echo "[remote-setup] $HOST ready"
    exit 0
fi

reason="$(grep -m1 '^FAIL=' "$err_file" "$out_file" 2>/dev/null | sed 's/.*FAIL=//' | head -1)"
python3 "$SCRIPT_DIR/host_registry.py" --root "$REPO_DIR" state "$HOST" --set failed
python3 "$SCRIPT_DIR/host_registry.py" --root "$REPO_DIR" state "$HOST" --set-error "${reason:-unknown}"
echo "[remote-setup] $HOST failed: ${reason:-unknown}" >&2
exit 1
