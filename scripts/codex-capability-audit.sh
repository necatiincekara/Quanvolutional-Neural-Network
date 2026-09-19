#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

PROMPT='Use the codex-audit skill and the workflow_architect agent. Audit official Codex capabilities against this repository''s current Codex integration. Produce a dated capability matrix that separates already-used, underused, missing, and unnecessary features. Ground capability claims in official OpenAI Codex docs only.'

if [ "$#" -gt 0 ]; then
  PROMPT="$PROMPT Additional user focus: $*"
fi

exec codex --search \
  exec -m gpt-5.6-sol -c model_reasoning_effort=xhigh \
  -s read-only -C "$ROOT_DIR" \
  --include-plan-tool \
  --output-schema "$ROOT_DIR/schemas/codex/capability_audit.schema.json" \
  "$PROMPT"
