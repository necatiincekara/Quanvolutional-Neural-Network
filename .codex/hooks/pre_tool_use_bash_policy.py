#!/usr/bin/env python3
import json
import sys


def system_message(command: str) -> str | None:
    lowered = command.lower()

    if "train_v7.py" in lowered:
        return "Reminder: V7 trainable-quantum work is a Colab-first path. Confirm that this run is worth the compute before continuing."
    if "train_ablation_local.py" in lowered and "non_trainable_quantum" in lowered:
        return "Reminder: cached non-trainable quantum timing excludes the one-time cache/precompute stage."
    if "paper/draft.md" in lowered or "docs/" in lowered:
        return "Reminder: if this shell command is part of paper/docs editing, reconcile current results before changing claims."
    return None


def main() -> int:
    data = json.load(sys.stdin)
    command = data.get("tool_input", {}).get("command", "")
    message = system_message(command)
    if not message:
        return 0
    sys.stdout.write(json.dumps({"systemMessage": message}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
