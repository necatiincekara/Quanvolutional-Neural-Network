#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

PROMPT='Use the codex-audit and status skills with the workflow_architect agent. Map this repo''s real workflows to current Codex coverage. For each workflow, identify the entrypoint, current skill/agent/script coverage, failure mode, missing Codex feature, recommended fix, and priority.'

if [ "$#" -gt 0 ]; then
  PROMPT="$PROMPT Additional user focus: $*"
fi

exec codex --search \
  -c 'model_reasoning_effort="high"' \
  -c 'plan_mode_reasoning_effort="high"' \
  -c 'mcp_servers={}' \
  exec -m gpt-5.4 -s read-only -C "$ROOT_DIR" --include-plan-tool "$PROMPT"
