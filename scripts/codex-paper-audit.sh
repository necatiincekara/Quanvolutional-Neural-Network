#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

PROMPT='Use the paper-sync skill and the paper_consistency_reviewer agent. Audit paper/draft.md against the strongest available repository evidence, list stale or unsupported claims, and propose the minimum set of corrections needed before submission.'

if [ "$#" -gt 0 ]; then
  PROMPT="$PROMPT Additional user focus: $*"
fi

exec codex --search \
  -c 'model_reasoning_effort="high"' \
  -c 'plan_mode_reasoning_effort="high"' \
  -c 'mcp_servers={}' \
  exec -m gpt-5.4 -s read-only -C "$ROOT_DIR" --include-plan-tool "$PROMPT"
