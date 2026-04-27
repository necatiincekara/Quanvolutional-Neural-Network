#!/usr/bin/env python3
import json
import sys


def build_context(prompt: str) -> str | None:
    lowered = prompt.lower()
    notes: list[str] = []

    if any(token in lowered for token in ["paper", "draft", "abstract", "conclusion", "submission", "advisor", "docs/"]):
        notes.append("Before writing paper/docs/share material, route through reconcile-results or paper-sync.")
    if any(token in lowered for token in ["quantum advantage", "advantage", "best model", "winner"]):
        notes.append("Current repository evidence does not support a generic quantum-advantage claim.")
    if any(token in lowered for token in ["train_v7", "colab", "rerun", "benchmark", "next run"]):
        notes.append("The V7 confirmatory rerun is complete; prioritize artifact provenance cleanup over redundant trainable-quantum reruns.")

    if not notes:
        return None
    return " ".join(notes)


def main() -> int:
    data = json.load(sys.stdin)
    extra = build_context(data.get("prompt", ""))
    if not extra:
        return 0

    payload = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": extra
        }
    }
    sys.stdout.write(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
