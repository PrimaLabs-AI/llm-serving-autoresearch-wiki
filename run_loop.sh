#!/usr/bin/env bash
# Fully automated autoresearch loop for LLM serving optimization.
#
# Claude drives the entire loop:
#   1. Read wiki state (hypotheses, prior experiments, engine pages)
#   2. Pick the top-ranked hypothesis
#   3. Generate the engine config + workload + benchmark command
#   4. Execute the benchmark
#   5. Ingest results → write experiment page → update hypotheses
#   6. Propose next experiment
#   7. Repeat from step 2
#
# Usage:
#   ./run_loop.sh --model meta-llama/Meta-Llama-3-8B-Instruct --rounds 5
#   ./run_loop.sh --model meta-llama/Meta-Llama-3-8B-Instruct --rounds infinity
#
# Prerequisites:
#   ./run_eval.sh --setup    (or ./setup.sh --docker)
#   claude                    (Claude Code CLI installed and authenticated)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATE=$(date +%Y-%m-%d)

# Defaults
MODEL=""
ROUNDS=5
LOOP_LOG="${SCRIPT_DIR}/raw/benchmarks/${DATE}-loop-log.md"

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) shift; MODEL="$1" ;;
        --rounds) shift; ROUNDS="$1" ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

if [[ -z "${MODEL}" ]]; then
    echo "ERROR: --model is required"
    echo "Usage: $0 --model <model> [--rounds <N>]"
    exit 1
fi

# ---------------------------------------------------------------------------
# The system prompt that drives Claude's behavior in the loop
# ---------------------------------------------------------------------------

read -r -d '' SYSTEM_PROMPT << 'SYSTEM_PROMPT_EOF' || true
You are the autonomous optimization agent for this LLM serving autoresearch wiki.

Your job is to drive the optimization loop WITHOUT human intervention:
1. READ wiki/index.md to understand current state (hypotheses, experiments, observations)
2. READ the top-ranked open hypothesis
3. PLAN the next benchmark: which engine, which config change, which workload, which concurrency levels
4. EXECUTE the benchmark by running benchmark_harness.py or docker compose commands
5. INGEST results: write an experiment page per SCHEMA.md, update hypothesis status, extract observations
6. PROPose new hypotheses based on what you observed
7. UPDATE wiki/index.md and wiki/log.md
8. Output a JSON block with your next action so the loop script can continue

Rules:
- Always read wiki/index.md first on every turn
- Follow SCHEMA.md for all page formats
- Every experiment gets a page under wiki/experiments/
- Update hypothesis status: open → in_progress → supported/refuted/inconclusive
- Write observations for reusable findings
- Rank new hypotheses by expected_gain × confidence / effort
- If a benchmark fails, record it as inconclusive and move to the next hypothesis
- Never change model semantics — if output quality degrades, mark invalid

Output format — end your response with a JSON block:
```json
{
  "action": "benchmark" | "done" | "error",
  "engine": "vllm" | "sglang" | "trt-llm",
  "service": "vllm" | "sglang" | "trt-llm",
  "engine_config": {"key": "value"},
  "workload": "workload-slug",
  "concurrency_levels": [16, 32, 64, 128],
  "output_dir": "raw/benchmarks/YYYY-MM-DD-slug",
  "hypothesis": "hypothesis-slug",
  "summary": "one line summary of what this round found"
}
```

If action is "done", the loop will stop.
If action is "error", the loop will log and continue to the next hypothesis.
SYSTEM_PROMPT_EOF

# ---------------------------------------------------------------------------
# Loop iteration
# ---------------------------------------------------------------------------

round=0
consecutive_errors=0
max_errors=3

start_loop_log() {
    mkdir -p "$(dirname "${LOOP_LOG}")"
    {
        echo "# Autoresearch Loop Log — ${DATE}"
        echo ""
        echo "Model: ${MODEL}"
        echo "Started: $(date -Iseconds)"
        echo ""
    } > "${LOOP_LOG}"
}

log_round() {
    local round_num=$1
    local status=$2
    local summary=$3
    {
        echo "## Round ${round_num} — $(date +%H:%M:%S)"
        echo ""
        echo "**Status**: ${status}"
        echo "**Summary**: ${summary}"
        echo ""
    } >> "${LOOP_LOG}"
}

ensure_engine_running() {
    local engine=$1
    local service="${engine}"
    local health_url

    case $engine in
        vllm) health_url="http://localhost:8000/health" ;;
        sglang) health_url="http://localhost:30000/health" ;;
        trt-llm) health_url="http://localhost:8001/health" ;;
    esac

    # Check if already running
    if curl -sf "${health_url}" &>/dev/null; then
        echo "already running"
        return 0
    fi

    echo "starting ${service}..."
    cd "${SCRIPT_DIR}"

    if [[ "${engine}" == "trt-llm" ]]; then
        MODEL="${MODEL}" docker compose --profile trt-llm up -d "${service}"
    else
        MODEL="${MODEL}" docker compose up -d "${service}"
    fi

    # Wait for healthy
    local elapsed=0
    while [[ $elapsed -lt 300 ]]; do
        if curl -sf "${health_url}" &>/dev/null; then
            echo "healthy (${elapsed}s)"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done

    echo "FAILED to start"
    return 1
}

run_claude_turn() {
    local turn_prompt=$1

    cd "${SCRIPT_DIR}"

    # Use Claude Code in non-interactive print mode
    claude -p "${turn_prompt}" \
        --system-prompt "${SYSTEM_PROMPT}" \
        --allowedTools "Read,Write,Edit,Bash,Glob,Grep" \
        --output-format text \
        2>/dev/null
}

run_benchmark_from_json() {
    local json=$1
    local output_dir

    output_dir=$(echo "${json}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['output_dir'])" 2>/dev/null || echo "")

    if [[ -z "${output_dir}" ]]; then
        echo "ERROR: could not parse output_dir from JSON"
        return 1
    fi

    local engine workload concurrency_levels
    engine=$(echo "${json}" | python3 -c "import json,sys; print(json.load(sys.stdin)['engine'])")
    workload=$(echo "${json}" | python3 -c "import json,sys; print(json.load(sys.stdin)['workload'])")
    concurrency_levels=$(echo "${json}" | python3 -c "import json,sys; print(' '.join(map(str,json.load(sys.stdin)['concurrency_levels'])))")

    # Ensure engine is running
    local engine_status
    engine_status=$(ensure_engine_running "${engine}")
    echo "Engine ${engine}: ${engine_status}"

    if [[ "${engine_status}" == *"FAILED"* ]]; then
        return 1
    fi

    # Build engine config from JSON
    local config_json
    config_json=$(echo "${json}" | python3 -c "import json,sys; print(json.dumps(json.load(sys.stdin).get('engine_config',{})))" 2>/dev/null || echo "{}")

    # Create output dir
    mkdir -p "${output_dir}"

    # Run benchmark
    cd "${SCRIPT_DIR}"
    python3 benchmark_harness.py \
        --engine "${engine}" \
        --model "${MODEL}" \
        --workload "${workload}" \
        --config "${config_json}" \
        --skip-server \
        --concurrency-levels ${concurrency_levels} \
        --output-dir "${output_dir}" \
        2>&1 | tee "${output_dir}/console.log"

    return ${PIPESTATUS[0]}
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

start_loop_log

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Autonomous LLM Serving Optimization Loop"
echo "  Model: ${MODEL}"
echo "  Max rounds: ${ROUNDS}"
echo "  Log: ${LOOP_LOG}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Initial prompt to kick off the loop
INIT_PROMPT="You are starting a fresh optimization loop. Read wiki/index.md to see the current state of hypotheses. This GPU instance has Docker containers for vLLM, SGLang, and TensorRT-LLM available. The model to optimize is: ${MODEL}. Pick the top-ranked hypothesis and plan the first benchmark. Output your plan as JSON at the end."

NEXT_PROMPT="The benchmark for this round has completed. Read the results from the output directory specified in your plan, plus any files in raw/benchmarks/. Write the experiment page, update hypothesis status, extract observations, propose new hypotheses, and update wiki/index.md and wiki/log.md. Then plan the next benchmark from the updated ranked list. Output your plan as JSON at the end. If all hypotheses have been tested or no fruitful hypotheses remain, output action 'done'."

current_prompt="${INIT_PROMPT}"

while true; do
    round=$((round + 1))

    if [[ "${ROUNDS}" != "infinity" ]] && [[ $round -gt $ROUNDS ]]; then
        echo ""
        echo "Reached max rounds (${ROUNDS}). Stopping."
        log_round "${round}" "stopped" "Reached max rounds ${ROUNDS}"
        break
    fi

    if [[ $consecutive_errors -ge $max_errors ]]; then
        echo ""
        echo "Too many consecutive errors (${consecutive_errors}). Stopping."
        log_round "${round}" "error" "Too many consecutive errors"
        break
    fi

    echo ""
    echo "━━━ Round ${round} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # --- Phase 1: Claude decides what to do ---
    echo "[Round ${round}] Asking agent to plan next experiment..."

    claude_output=$(run_claude_turn "${current_prompt}" 2>&1) || {
        echo "[Round ${round}] Agent call failed."
        log_round "${round}" "error" "Agent call failed"
        consecutive_errors=$((consecutive_errors + 1))
        current_prompt="The previous round failed (agent error). Read wiki/index.md and try the next hypothesis. Output JSON."
        continue
    }

    # Extract JSON from Claude's output
    json_block=$(echo "${claude_output}" | python3 -c "
import sys, re
text = sys.stdin.read()
match = re.search(r'\`\`\`json\s*(\{.*?\})\s*\`\`\`', text, re.DOTALL)
if match:
    print(match.group(1))
else:
    # Try to find last JSON object in output
    matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if matches:
        print(matches[-1])
    else:
        print('')
" 2>/dev/null || echo "")

    if [[ -z "${json_block}" ]]; then
        echo "[Round ${round}] Could not extract action JSON from agent output."
        echo "Agent output (last 20 lines):"
        echo "${claude_output}" | tail -20
        log_round "${round}" "error" "No JSON action in agent output"
        consecutive_errors=$((consecutive_errors + 1))
        continue
    fi

    # Parse action
    action=$(echo "${json_block}" | python3 -c "import json,sys; print(json.load(sys.stdin).get('action','error'))" 2>/dev/null || echo "error")
    summary=$(echo "${json_block}" | python3 -c "import json,sys; print(json.load(sys.stdin).get('summary',''))" 2>/dev/null || echo "")

    echo "[Round ${round}] Action: ${action}"
    echo "[Round ${round}] Summary: ${summary}"

    if [[ "${action}" == "done" ]]; then
        echo ""
        echo "Agent reports optimization complete."
        log_round "${round}" "done" "${summary}"
        break
    fi

    if [[ "${action}" == "error" ]]; then
        echo "[Round ${round}] Agent reported error."
        log_round "${round}" "error" "${summary}"
        consecutive_errors=$((consecutive_errors + 1))
        continue
    fi

    # --- Phase 2: Execute benchmark ---
    echo "[Round ${round}] Running benchmark..."

    if run_benchmark_from_json "${json_block}"; then
        echo "[Round ${round}] Benchmark succeeded."
        log_round "${round}" "benchmark_ok" "${summary}"
        consecutive_errors=0
    else
        echo "[Round ${round}] Benchmark failed."
        log_round "${round}" "benchmark_failed" "${summary}"
        consecutive_errors=$((consecutive_errors + 1))
    fi

    # --- Phase 3: Next turn asks Claude to ingest results ---
    current_prompt="${NEXT_PROMPT}"

done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Loop complete. ${round} rounds executed."
echo "  Log: ${LOOP_LOG}"
echo ""
echo "  Next steps:"
echo "    git add wiki/ raw/benchmarks/ && git commit -m 'autoresearch loop ${DATE}'"
echo "    git push"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
