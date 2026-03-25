#!/usr/bin/env bash
set -euo pipefail

if ! command -v pandoc >/dev/null 2>&1; then
  echo "pandoc not found on PATH" >&2
  exit 1
fi

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <markdown-file> [<markdown-file> ...]" >&2
  exit 1
fi

for src in "$@"; do
  if [ ! -f "$src" ]; then
    echo "Missing file: $src" >&2
    exit 1
  fi

  case "$src" in
    *.md) ;;
    *)
      echo "Not a markdown file: $src" >&2
      exit 1
      ;;
  esac

  out="${src%.md}.docx"
  pandoc "$src" \
    --from gfm \
    --to docx \
    --resource-path . \
    --output "$out"
  echo "Wrote $out"
done
