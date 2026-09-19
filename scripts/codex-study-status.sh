#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

PROMPT='Use the reconcile-results and status skills. Determine the current factual status of this study from repo artifacts, separate current findings from historical docs, and report in Turkish. Lead with the strongest supported conclusion.'

if [ "$#" -gt 0 ]; then
  PROMPT="$PROMPT Additional user focus: $*"
fi

exec codex --search \
  exec -m gpt-5.6-sol -c model_reasoning_effort=xhigh \
  -s read-only -C "$ROOT_DIR" \
  --include-plan-tool \
  --output-schema "$ROOT_DIR/schemas/codex/status.schema.json" \
  "$PROMPT"
