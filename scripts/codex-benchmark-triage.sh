#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

PROMPT='Use the benchmark-triage skill and the benchmark_strategist agent. Determine the next highest-value benchmark or training task for this study from current artifacts, platform constraints, compute budget, and paper impact. Output exact next task order, platform, expected cost, and stop conditions.'

if [ "$#" -gt 0 ]; then
  PROMPT="$PROMPT Additional user focus: $*"
fi

exec codex --search \
  -c 'model_reasoning_effort="high"' \
  -c 'plan_mode_reasoning_effort="high"' \
  -c 'mcp_servers={}' \
  exec -m gpt-5.4 -s read-only -C "$ROOT_DIR" --include-plan-tool "$PROMPT"
