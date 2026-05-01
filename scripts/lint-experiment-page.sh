#!/usr/bin/env bash
# Lints a single experiment page for required frontmatter and structure.
# Exit 0 if clean, 1 if any issue is found (with errors on stderr).
#
# Usage: ./scripts/lint-experiment-page.sh wiki/experiments/<run_slug>.md

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <experiment-page>" >&2
    exit 2
fi

PAGE="$1"

if [ ! -f "$PAGE" ]; then
    echo "lint: file not found: $PAGE" >&2
    exit 1
fi

errors=0
err() { echo "lint: $PAGE: $*" >&2; errors=$((errors + 1)); }

# Required frontmatter fields
for key in title type tags hypothesis model engine workload hardware host verdict; do
    if ! grep -qE "^${key}:" "$PAGE"; then
        err "missing frontmatter: $key"
    fi
done

# Frontmatter `type:` must be `experiment`
if ! grep -qE '^type:\s*experiment' "$PAGE"; then
    err "type must be 'experiment'"
fi

# Verdict must be one of the allowed values
verdict_line="$(grep -E '^verdict:' "$PAGE" || true)"
case "$verdict_line" in
    *supported*|*refuted*|*inconclusive*|*invalid*) : ;;
    *) err "verdict must be one of: supported|refuted|inconclusive|invalid (got: ${verdict_line:-<missing>})" ;;
esac

# Required H2 sections
for h2 in "Hypothesis under test" "Setup" "Results" "Verdict" "Sources"; do
    if ! grep -qE "^## ${h2}" "$PAGE"; then
        err "missing H2 section: $h2"
    fi
done

# If verdict != invalid, the Profile / Benchmark section must exist
if ! grep -qE '^verdict:\s*invalid' "$PAGE"; then
    if ! grep -qE '^## Profile' "$PAGE"; then
        err "non-invalid verdict requires '## Profile / Benchmark' section"
    fi
fi

# Profile path must reference raw/benchmarks/ or raw/profiles/
if grep -qE '^## Profile' "$PAGE"; then
    if ! grep -qE 'raw/(benchmarks|profiles)/' "$PAGE"; then
        err "Profile section must cite a path under raw/benchmarks/ or raw/profiles/"
    fi
fi

if [ "$errors" -gt 0 ]; then
    echo "lint: $errors error(s)" >&2
    exit 1
fi
exit 0
