#!/usr/bin/env python3
"""Reconstruct low-data result rows from captured Colab notebook output.

The original JSON rows remain authoritative when they are available. This tool
exists because the notebook captures the complete command, epoch metrics,
best-validation updates, test accuracy, and original result path for seeds
43--47, while those Drive-backed JSON files are not all present locally.
Every reconstructed row is labelled explicitly and can later be replaced by
the original Drive JSON without changing filenames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


RUN_PATTERN = re.compile(
    r"^=== Running \d+/\d+: (?P<command>.+?) ===\n"
    r"(?P<body>.*?)(?=^=== Running \d+/\d+:|\Z)",
    re.MULTILINE | re.DOTALL,
)
COMMAND_PATTERN = re.compile(
    r"--model (?P<model>\S+).*?"
    r"--seed (?P<seed>\d+).*?"
    r"--split-seed (?P<split_seed>\d+).*?"
    r"--protocol-version (?P<protocol>\S+).*?"
    r"--train-fraction (?P<fraction>[0-9.]+).*?"
    r"--fraction-seed (?P<fraction_seed>\d+)"
)
EPOCH_PATTERN = re.compile(
    r"Train Loss: (?P<train_loss>[0-9.]+), Train Acc: (?P<train_acc>[0-9.]+)%.*?"
    r"Val Loss: (?P<val_loss>[0-9.]+), Val Acc: (?P<val_acc>[0-9.]+)%",
    re.DOTALL,
)


def notebook_stdout(path: Path) -> str:
    with path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    chunks: list[str] = []
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            text = output.get("text") if isinstance(output, dict) else None
            if isinstance(text, list):
                chunks.extend(text)
    return "".join(chunks)


def extract_rows(notebook_path: Path) -> list[dict[str, Any]]:
    text = notebook_stdout(notebook_path)
    rows = []
    for run_match in RUN_PATTERN.finditer(text):
        command = run_match.group("command")
        body = run_match.group("body")
        command_match = COMMAND_PATTERN.search(command)
        if not command_match:
            continue
        meta = command_match.groupdict()
        if meta["model"] not in {"classical_conv", "non_trainable_quantum"}:
            continue

        dataset_match = re.search(
            r"Dataset sizes: \{'train': (\d+), 'val': (\d+), 'test': (\d+)\}",
            body,
        )
        test_match = re.search(r"Test Accuracy: ([0-9.]+)%", body)
        saved_match = re.search(r"Results saved to (\S+\.json)", body)
        epoch_matches = list(EPOCH_PATTERN.finditer(body))
        if not dataset_match or not test_match or not saved_match or not epoch_matches:
            raise ValueError(f"Incomplete captured output for command: {command}")

        epochs = []
        for index, epoch_match in enumerate(epoch_matches, start=1):
            values = epoch_match.groupdict()
            epochs.append(
                {
                    "epoch": index,
                    "train_loss": float(values["train_loss"]),
                    "train_acc": float(values["train_acc"]),
                    "val_loss": float(values["val_loss"]),
                    "val_acc": float(values["val_acc"]),
                }
            )
        best_val_acc = max(epoch["val_acc"] for epoch in epochs)
        model = meta["model"]
        seed = int(meta["seed"])
        split_seed = int(meta["split_seed"])
        fraction_seed = int(meta["fraction_seed"])
        fraction = float(meta["fraction"])
        dataset_sizes = [int(value) for value in dataset_match.groups()]
        total_params = 88045 if model == "classical_conv" else 88488
        fraction_tag = f"frac{int(round(fraction * 100)):03d}"
        row = {
            "model": model,
            "source": "captured-notebook-output-reconstruction",
            "family": "current-local",
            "protocol_version": meta["protocol"],
            "platform": "colab-NVIDIA L4",
            "train_seed": seed,
            "split_seed": split_seed,
            "params": {"total": total_params, "trainable": total_params},
            "total_params": total_params,
            "trainable_params": total_params,
            "epochs": epochs,
            "best_val_acc": round(best_val_acc, 2),
            "test_loss": None,
            "test_acc": float(test_match.group(1)),
            "runtime_seconds": None,
            "runtime_minutes": None,
            "benchmark_axis": "low-data-scaling",
            "train_fraction": fraction,
            "fraction_seed": fraction_seed,
            "dataset_sizes": {
                "train": dataset_sizes[0],
                "val": dataset_sizes[1],
                "test": dataset_sizes[2],
            },
            "low_data": {
                "enabled": fraction < 1.0,
                "train_fraction": fraction,
                "fraction_seed": fraction_seed,
                "selected_train_size": dataset_sizes[0],
                "classes_present": 44,
            },
            "run_id": (
                f"{model}__{meta['protocol']}__{fraction_tag}__"
                f"seed{seed}__split{split_seed}__fraction{fraction_seed}"
            ),
            "artifact_status": "reconstructed-from-captured-output",
            "provenance": {
                "source_notebook": str(notebook_path),
                "captured_command": command,
                "original_remote_json_path": saved_match.group(1),
                "original_remote_json_expected": True,
                "reconstruction_limits": [
                    "test_loss and runtime were not printed in the captured output",
                    "original Drive JSON should replace this row when synced",
                ],
            },
        }
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", default="colab_low_data_confirm.ipynb")
    parser.add_argument("--output-dir", default="experiments/low_data")
    parser.add_argument(
        "--manifest-out",
        default="experiments/low_data_reconstruction_manifest_20260728.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    notebook_path = Path(args.notebook)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = extract_rows(notebook_path)
    if len(rows) != 40:
        raise RuntimeError(f"Expected 40 captured seed-43--47 rows, found {len(rows)}")

    entries = []
    for row in rows:
        output_path = output_dir / Path(
            row["provenance"]["original_remote_json_path"]
        ).name
        if output_path.exists() and not args.overwrite:
            with output_path.open(encoding="utf-8") as handle:
                existing = json.load(handle)
            if existing.get("artifact_status") != "reconstructed-from-captured-output":
                entries.append(
                    {
                        "path": str(output_path),
                        "status": "original-or-preexisting-kept",
                    }
                )
                continue
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(row, handle, indent=2, ensure_ascii=False)
        entries.append({"path": str(output_path), "status": row["artifact_status"]})

    notebook_sha256 = hashlib.sha256(notebook_path.read_bytes()).hexdigest()
    manifest = {
        "artifact": "low-data-notebook-reconstruction-manifest",
        "created_date": "2026-07-28",
        "source_notebook": str(notebook_path),
        "source_notebook_sha256": notebook_sha256,
        "captured_rows": len(rows),
        "seeds": sorted({row["train_seed"] for row in rows}),
        "models": sorted({row["model"] for row in rows}),
        "fractions": sorted({row["train_fraction"] for row in rows}),
        "drive_folder": {
            "path": "MyDrive/quanv_results/low_data_confirm_v2_20260517",
            "id": "1kILcZ9o358QQykuqk0COxoOpFn4cLIKN",
            "url": "https://drive.google.com/drive/folders/1kILcZ9o358QQykuqk0COxoOpFn4cLIKN",
            "observed_json_count": 56,
        },
        "entries": entries,
        "provenance_policy": (
            "Reconstructed rows are supported by captured notebook output but are not byte-identical "
            "copies of the original Drive JSON. Replace them with the originals when available."
        ),
    }
    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(f"Reconstructed/verified {len(entries)} rows")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
