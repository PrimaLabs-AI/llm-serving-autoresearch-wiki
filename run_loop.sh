#!/usr/bin/env bash
# Mac-side autoresearch loop driver.
#
# Per round:
#   1. reachability sweep (ssh ping all in-scope hosts)
#   2. setup pass (run remote-bootstrap.sh on any pending/failed hosts)
#   3. PICK turn (claude --print prints HYPOTHESIS=<slug>)
#   4. schedule (host_registry.py schedule …)
#   5. RUN turn (claude --print does ssh+benchmark+rsync+writeup)
#   6. lint experiment page
#   7. git commit wiki + raw/benchmarks/<run>
#
# Usage:
#   ./run_loop.sh --rounds 5 [--hosts h100-1,b200-1] [--model <id>] [--tag <name>]
#
# After a halt (return code 2 from a wiki-conflict commit failure),
# resolve the conflict in your working tree and rerun the same command —
# completed rounds are already committed and won't be repeated.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATE="$(date +%Y-%m-%d)"

# Defaults
ROUNDS=5
HOSTS_FILTER=""
MODEL=""
TAG="loop"

while [[ $# -gt 0 ]]; do
    case $1 in
        --rounds) shift; ROUNDS="$1" ;;
        --hosts)  shift; HOSTS_FILTER="$1" ;;
        --model)  shift; MODEL="$1" ;;
        --tag)    shift; TAG="$1" ;;
        *)        echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

LOG_DIR="$SCRIPT_DIR/raw/loops"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${DATE}-${TAG}.log"

# Convenience: log to file and stderr
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG_FILE" >&2; }

# Resolve in-scope hosts (bash 3.2-compatible — no mapfile)
all_hosts=()
while IFS= read -r line; do
    [ -n "$line" ] && all_hosts+=("$line")
done < <(python3 "$SCRIPT_DIR/scripts/host_registry.py" list)
if [ -z "$HOSTS_FILTER" ]; then
    in_scope=("${all_hosts[@]+${all_hosts[@]}}")
else
    IFS=',' read -ra in_scope <<< "$HOSTS_FILTER"
fi

if [ ${#in_scope[@]} -eq 0 ]; then
    log "ERROR: no hosts in scope. Edit .hosts.toml (see docs/host-onboarding.md)."
    exit 2
fi

# Source MODEL from .env if not set on command line
if [ -z "$MODEL" ] && [ -f "$SCRIPT_DIR/.env" ]; then
    set -a; source "$SCRIPT_DIR/.env"; set +a
fi
if [ -z "${MODEL:-}" ]; then
    log "ERROR: --model is required (or set MODEL in .env)"
    exit 2
fi

reachability_sweep() {
    log "reachability sweep: ${in_scope[*]}"
    for h in "${in_scope[@]}"; do
        if python3 "$SCRIPT_DIR/scripts/host_registry.py" reachable "$h" >/dev/null 2>&1; then
            log "  $h: reachable"
        else
            log "  $h: UNREACHABLE"
            python3 "$SCRIPT_DIR/scripts/host_registry.py" state "$h" --set unreachable
        fi
    done
}

setup_pass() {
    log "setup pass"
    for h in "${in_scope[@]}"; do
        local state
        state="$(python3 "$SCRIPT_DIR/scripts/host_registry.py" list --summary | awk -v n="$h" '$1==n {print $4}')"
        case "$state" in
            ready) log "  $h: ready" ;;
            pending|failed)
                log "  $h: $state → bootstrapping"
                if "$SCRIPT_DIR/scripts/remote-setup.sh" "$h" >>"$LOG_FILE" 2>&1; then
                    log "  $h: bootstrap OK"
                else
                    log "  $h: bootstrap FAILED (see log)"
                fi
                ;;
            running) log "  $h: $state (concurrent run? skipping)" ;;
            unreachable) log "  $h: unreachable; skipping setup" ;;
        esac
    done
}

# Returns the hypothesis slug picked, or "none"
pick_turn() {
    local round="$1"
    local excluded="$2"
    local registry_summary
    registry_summary="$(python3 "$SCRIPT_DIR/scripts/host_registry.py" list --summary)"
    local user_msg
    user_msg=$(cat <<EOF
round=$round
model=$MODEL
excluded=${excluded:-}
registry_summary:
$registry_summary
EOF
)
    local out
    out="$(claude --print \
        --append-system-prompt "$(cat "$SCRIPT_DIR/prompts/pick.md")" \
        "$user_msg" 2>>"$LOG_FILE")"
    echo "$out" | grep -E '^HYPOTHESIS=' | head -1 | sed 's/^HYPOTHESIS=//'
}

# Returns "EXPERIMENT=<path>|VERDICT=<v>" on success, empty on failure
run_turn() {
    local hyp="$1"
    local host="$2"
    local run_slug="${DATE}-${hyp}"
    local user_msg
    user_msg=$(cat <<EOF
hypothesis=$hyp
host=$host
run_slug=$run_slug
model=$MODEL
EOF
)
    local out
    out="$(claude --print \
        --append-system-prompt "$(cat "$SCRIPT_DIR/prompts/run.md")" \
        "$user_msg" 2>>"$LOG_FILE")"
    local exp_line verdict_line
    exp_line="$(echo "$out" | grep -E '^EXPERIMENT=' | head -1)"
    verdict_line="$(echo "$out" | grep -E '^VERDICT=' | head -1)"
    if [ -n "$exp_line" ] && [ -n "$verdict_line" ]; then
        echo "${exp_line}|${verdict_line}"
    fi
}

run_round() {
    local round="$1"
    log "════ round $round / $ROUNDS ════"

    reachability_sweep
    setup_pass

    local excluded=""
    local hyp host scheduled
    local pick_attempts=0
    while [ "$pick_attempts" -lt 3 ]; do
        pick_attempts=$((pick_attempts + 1))
        hyp="$(pick_turn "$round" "$excluded")"
        if [ -z "$hyp" ] || [ "$hyp" = "none" ]; then
            log "  PICK: none (attempt $pick_attempts)"
            return 1
        fi

        # Read engine + hardware from hypothesis frontmatter
        local hyp_path="wiki/hypotheses/$hyp.md"
        if [ ! -f "$hyp_path" ]; then
            log "  PICK returned non-existent slug: $hyp"
            excluded="${excluded}${excluded:+,}$hyp"
            continue
        fi
        local hyp_hardware engine engine_supported
        hyp_hardware="$(awk '/^---$/{c++} c==1 && /^hardware:/{print $2; exit}' "$hyp_path")"
        engine="$(awk '/^---$/{c++} c==1 && /^engine:/{print $2; exit}' "$hyp_path")"
        if [ -z "$hyp_hardware" ] || [ -z "$engine" ]; then
            log "  hypothesis missing hardware: or engine: frontmatter"
            excluded="${excluded}${excluded:+,}$hyp"
            continue
        fi
        engine_supported="$(awk '/^---$/{c++} c==1 && /^supported_hardware:/{
            sub(/^supported_hardware:[[:space:]]*\[/, ""); sub(/\][[:space:]]*$/, ""); gsub(/[[:space:]]/, ""); print; exit
        }' "wiki/engines/$engine.md")"

        scheduled="$(python3 "$SCRIPT_DIR/scripts/host_registry.py" schedule \
            --hypothesis-hardware "$hyp_hardware" \
            --engine-supported "$engine_supported")"
        if [ "$scheduled" = "none" ]; then
            log "  schedule: $hyp unschedulable on current registry; excluding & retrying"
            excluded="${excluded}${excluded:+,}$hyp"
            continue
        fi

        host="$scheduled"
        log "  PICK: $hyp → host $host (attempt $pick_attempts)"
        break
    done

    if [ -z "${host:-}" ]; then
        log "  no schedulable hypothesis after 3 attempts; ending round"
        return 1
    fi

    log "  RUN: $hyp on $host"
    local result
    result="$(run_turn "$hyp" "$host")"
    if [ -z "$result" ]; then
        log "  RUN: malformed output; retrying once with stricter prompt"
        result="$(run_turn "$hyp" "$host")"
    fi
    if [ -z "$result" ]; then
        log "  RUN: failed twice; aborting round"
        return 1
    fi

    local exp_path verdict
    exp_path="$(echo "$result" | sed 's/|.*//; s/^EXPERIMENT=//')"
    verdict="$(echo "$result"  | sed 's/.*|//;  s/^VERDICT=//')"
    log "  result: $exp_path verdict=$verdict"

    if ! "$SCRIPT_DIR/scripts/lint-experiment-page.sh" "$exp_path" >>"$LOG_FILE" 2>&1; then
        log "  lint FAILED for $exp_path; aborting round (page left in place)"
        return 1
    fi

    git add wiki/ raw/benchmarks/ >>"$LOG_FILE" 2>&1
    if ! git commit -m "round $round: $hyp on $host ($verdict)" >>"$LOG_FILE" 2>&1; then
        log "  git commit FAILED (probable wiki edit conflict); halting"
        return 2
    fi
    log "  committed"
    return 0
}

log "════════════════ loop start ════════════════"
log "rounds=$ROUNDS hosts=${in_scope[*]} model=$MODEL tag=$TAG"

for r in $(seq 1 "$ROUNDS"); do
    set +e
    run_round "$r"
    rc=$?
    set -e
    case $rc in
        0) ;;  # ok
        1) log "round $r ended without commit; continuing"; ;;
        2) log "halting loop due to wiki conflict; resolve and rerun with --resume"; exit 2; ;;
        *) log "round $r exited with $rc; continuing"; ;;
    esac
done

log "════════════════ loop end ════════════════"
