#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

PROMPT='Use the sync-colab and train skills. Plan a clean Colab handoff for the next trainable-quantum run in this repo. Include environment checks, dataset path expectations, checkpoint handling, Drive backup behavior, exact command choice, and post-run reconciliation steps.'

if [ "$#" -gt 0 ]; then
  PROMPT="$PROMPT Additional user focus: $*"
fi

exec codex --search \
  -c 'model_reasoning_effort="medium"' \
  -c 'plan_mode_reasoning_effort="medium"' \
  -c 'mcp_servers={}' \
  exec -m gpt-5.4-mini -s read-only -C "$ROOT_DIR" --include-plan-tool "$PROMPT"
