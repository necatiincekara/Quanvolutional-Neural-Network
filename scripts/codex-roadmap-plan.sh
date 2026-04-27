#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

PROMPT='Use the roadmap-sync skill and the benchmark_strategist agent. Produce the next-phase master plan for this study from benchmark truth, publication strategy, compute budget, and current code state. Separate near-term publishable work from broader research work.'

if [ "$#" -gt 0 ]; then
  PROMPT="$PROMPT Additional user focus: $*"
fi

exec codex --search \
  exec -p deep -s read-only -C "$ROOT_DIR" \
  --include-plan-tool \
  --output-schema "$ROOT_DIR/schemas/codex/roadmap.schema.json" \
  "$PROMPT"
