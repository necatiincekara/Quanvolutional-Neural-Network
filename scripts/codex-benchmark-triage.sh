#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

PROMPT='Use the benchmark-triage skill and the benchmark_strategist agent. Determine the next highest-value benchmark or training task for this study from current artifacts, platform constraints, compute budget, and paper impact. Output exact next task order, platform, expected cost, and stop conditions.'

if [ "$#" -gt 0 ]; then
  PROMPT="$PROMPT Additional user focus: $*"
fi

exec codex --search \
  exec -m gpt-5.6-sol -c model_reasoning_effort=xhigh \
  -s read-only -C "$ROOT_DIR" \
  --include-plan-tool \
  --output-schema "$ROOT_DIR/schemas/codex/benchmark_triage.schema.json" \
  "$PROMPT"
