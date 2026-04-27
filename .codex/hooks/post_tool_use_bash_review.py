#!/usr/bin/env python3
import json
import sys


def system_message(command: str) -> str | None:
    lowered = command.lower()

    if any(token in lowered for token in ["train_thesis_models.py", "train_ablation_local.py", "train_v7.py", "train_modern_baselines.py"]) or " -m src.train" in lowered:
        return "Result-affecting run finished. Reconcile artifacts and refresh benchmark summaries before changing narrative docs."
    if "aggregate_benchmarks.py" in lowered:
        return "Benchmark summary refreshed. Sync any publication-facing docs that depend on the updated table."
    if "export_docx.sh" in lowered or "advisor_share" in lowered:
        return "Shareable artifacts refreshed. Check that no stale claim was copied into the packet."
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
