#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

PROMPT='Use the artifact-pack and paper-sync skills. Refresh advisor/share/submission artifacts from current repo truth without introducing stale claims. Include Word exports when shareable Markdown documents are touched.'

if [ "$#" -gt 0 ]; then
  PROMPT="$PROMPT Additional user focus: $*"
fi

exec codex --search \
  -c 'model_reasoning_effort="high"' \
  -c 'plan_mode_reasoning_effort="high"' \
  -c 'mcp_servers={}' \
  exec -m gpt-5.4 -s read-only -C "$ROOT_DIR" --include-plan-tool "$PROMPT"
