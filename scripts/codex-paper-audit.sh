#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

PROMPT='Use the paper-sync skill and the paper_consistency_reviewer agent. Audit paper/draft.md against the strongest available repository evidence, list stale or unsupported claims, and propose the minimum set of corrections needed before submission.'

if [ "$#" -gt 0 ]; then
  PROMPT="$PROMPT Additional user focus: $*"
fi

exec codex --search \
  exec -p deep -s read-only -C "$ROOT_DIR" \
  --include-plan-tool \
  --output-schema "$ROOT_DIR/schemas/codex/paper_audit.schema.json" \
  "$PROMPT"
