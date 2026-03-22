#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

PROMPT='Use the reconcile-results and status skills. Determine the current factual status of this study from repo artifacts, separate current findings from historical docs, and report in Turkish. Lead with the strongest supported conclusion.'

if [ "$#" -gt 0 ]; then
  PROMPT="$PROMPT Additional user focus: $*"
fi

exec codex exec -p review -C "$ROOT_DIR" --search --include-plan-tool "$PROMPT"
