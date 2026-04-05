#!/usr/bin/env python3
import json
import sys


def main() -> int:
    payload = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": (
                "Repository truth hierarchy: experiments/*.json > docs/EXPERIMENTS.md > paper/draft.md > current code > historical docs. "
                "Keep thesis-faithful, current-local matched-budget, and trainable-quantum case-study families separate. "
                "Current strongest supported models: thesis_cnniiii=85.26±0.97, classical_conv=81.40±1.06. "
                "V7 remains an engineering case-study, not the benchmark leader."
            )
        }
    }
    sys.stdout.write(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
