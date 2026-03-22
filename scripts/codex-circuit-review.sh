#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TARGET=${1:-src/trainable_quantum_model.py}

if [ "$#" -gt 0 ]; then
  shift
fi

PROMPT="Use the review-circuit skill. Review the quantum or hybrid block in $TARGET with repository context. Lead with concrete findings, then give the smallest defensible improvements."

if [ "$#" -gt 0 ]; then
  PROMPT="$PROMPT Additional user focus: $*"
fi

exec codex exec -p review -C "$ROOT_DIR" --search --include-plan-tool "$PROMPT"
