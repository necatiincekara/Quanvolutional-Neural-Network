#!/usr/bin/env python3
import json
import sys


def system_message(command: str) -> str | None:
    lowered = command.lower()

    if "train_v7.py" in lowered:
        return "Reminder: V7 trainable-quantum work is a Colab-first path. Confirm that this run is worth the compute before continuing."
    if "train_modern_baselines.py" in lowered:
        return "Reminder: modern-classical runs can change the reviewer-proof upper-bound row. Reconcile artifacts before changing benchmark claims."
    if "train_thesis_models.py" in lowered:
        return "Reminder: thesis-faithful runs must stay separate from current-local matched-budget baselines in summaries."
    if " -m src.train" in lowered or lowered.startswith("python -m src.train") or lowered.startswith("python3 -m src.train"):
        return "Reminder: src.train is the older V4/V6 baseline path. Treat new output as historical-path evidence unless reconciled."
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
