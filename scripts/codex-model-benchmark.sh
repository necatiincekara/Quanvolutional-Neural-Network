#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OUT_DIR=${1:-$(mktemp -d "/tmp/codex-model-benchmark.XXXXXX")}

mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/summary.tsv"
printf "model\ttask\tseconds\toutput\n" > "$SUMMARY"

run_case() {
  MODEL="$1"
  TASK="$2"
  EFFORT="$3"
  PROMPT="$4"
  OUT_FILE="$OUT_DIR/${MODEL}_${TASK}.txt"

  START_TS=$(date +%s)
  codex --search \
    -c "model_reasoning_effort=\"$EFFORT\"" \
    -c "plan_mode_reasoning_effort=\"$EFFORT\"" \
    -c 'mcp_servers={}' \
    exec -m "$MODEL" -s read-only -C "$ROOT_DIR" --include-plan-tool \
    --output-last-message "$OUT_FILE" "$PROMPT" >/dev/null
  END_TS=$(date +%s)
  ELAPSED=$((END_TS - START_TS))
  printf "%s\t%s\t%s\t%s\n" "$MODEL" "$TASK" "$ELAPSED" "$OUT_FILE" >> "$SUMMARY"
}

RECONCILE_PROMPT='Use the reconcile-results and status skills. Determine the current factual status of this study from repo artifacts and report the strongest supported conclusion.'
PAPER_PROMPT='Use the paper-sync skill and the paper_consistency_reviewer agent. Audit paper/draft.md against current repository evidence and list the minimum corrections needed before submission.'
TRIAGE_PROMPT='Use the benchmark-triage skill and the benchmark_strategist agent. Determine the next highest-value benchmark or training task for this study.'

for MODEL in gpt-5.4 gpt-5.3-codex gpt-5.4-mini; do
  EFFORT=high
  if [ "$MODEL" = "gpt-5.4-mini" ]; then
    EFFORT=medium
  fi
  run_case "$MODEL" reconcile "$EFFORT" "$RECONCILE_PROMPT"
  run_case "$MODEL" paper_audit "$EFFORT" "$PAPER_PROMPT"
  run_case "$MODEL" benchmark_triage "$EFFORT" "$TRIAGE_PROMPT"
done

echo "Wrote $SUMMARY"
echo "Artifacts in $OUT_DIR"
