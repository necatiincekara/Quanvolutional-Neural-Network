#!/usr/bin/env python3
"""Build or execute the low-data scaling pilot command grid."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ABLATION_MODELS = {"classical_conv", "non_trainable_quantum", "param_linear"}
THESIS_MODELS = {"thesis_cnn3", "thesis_cnniiii", "thesis_hqnn2", "thesis_hqnn3"}
DEFAULT_MODELS = ["classical_conv", "non_trainable_quantum", "thesis_cnniiii", "thesis_hqnn2"]
DEFAULT_FRACTIONS = [0.10, 0.25, 0.50, 1.00]


def model_entrypoint(model: str) -> str:
    if model in ABLATION_MODELS:
        return "train_ablation_local.py"
    if model in THESIS_MODELS:
        return "train_thesis_models.py"
    known = ", ".join(sorted(ABLATION_MODELS | THESIS_MODELS))
    raise ValueError(f"Unknown model {model!r}. Known models: {known}")


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    commands: list[list[str]] = []
    for model in args.models:
        script = model_entrypoint(model)
        for fraction in args.fractions:
            for seed in args.seeds:
                cmd = [
                    args.python,
                    script,
                    "--model",
                    model,
                    "--seed",
                    str(seed),
                    "--split-seed",
                    str(args.split_seed),
                    "--protocol-version",
                    args.protocol_version,
                    "--train-fraction",
                    f"{fraction:.2f}",
                    "--fraction-seed",
                    str(args.fraction_seed if args.fraction_seed is not None else args.split_seed),
                    "--device",
                    args.device,
                ]
                if args.epochs is not None:
                    cmd.extend(["--epochs", str(args.epochs)])
                commands.append(cmd)
    return commands


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or print the low-data scaling pilot grid")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--fractions", nargs="+", type=float, default=DEFAULT_FRACTIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--fraction-seed", type=int, default=None)
    parser.add_argument("--protocol-version", default="low_data_pilot_v1")
    parser.add_argument("--device", choices=["cpu", "auto"], default="cpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    commands = build_commands(args)
    print(f"Low-data grid contains {len(commands)} commands.")
    for idx, cmd in enumerate(commands, start=1):
        print(f"[{idx:02d}/{len(commands):02d}] {shlex.join(cmd)}")

    if not args.execute:
        print("Dry run only. Add --execute to run the grid.")
        return 0

    for idx, cmd in enumerate(commands, start=1):
        print(f"\n=== Running {idx}/{len(commands)}: {shlex.join(cmd)} ===")
        completed = subprocess.run(cmd, cwd=repo_root, check=False)
        if completed.returncode != 0:
            print(f"Command failed with exit code {completed.returncode}: {shlex.join(cmd)}")
            if not args.continue_on_error:
                return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
