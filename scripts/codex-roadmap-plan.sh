#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

PROMPT='Use the roadmap-sync skill and the benchmark_strategist agent. Produce the next-phase master plan for this study from benchmark truth, publication strategy, compute budget, and current code state. Separate near-term publishable work from broader research work.'

if [ "$#" -gt 0 ]; then
  PROMPT="$PROMPT Additional user focus: $*"
fi

exec codex --search \
  -c 'model_reasoning_effort="high"' \
  -c 'plan_mode_reasoning_effort="high"' \
  -c 'mcp_servers={}' \
  exec -m gpt-5.4 -s read-only -C "$ROOT_DIR" --include-plan-tool "$PROMPT"
