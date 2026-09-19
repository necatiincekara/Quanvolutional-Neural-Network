#!/usr/bin/env python3
"""Reconcile downloaded Drive artifacts into canonical publication paths.

This is a provenance operation only. It performs no training and never edits a
checkpoint. Before copying byte-original low-data JSON files over notebook-
reconstructed counterparts, it verifies that all result-bearing fields agree.
It then copies the downloaded low-data checkpoints into ``models/low_data`` and
writes a SHA-256 reconciliation manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DRIVE_ROOT = ROOT / "experiments" / "Drive" / "quanv_results"
V2_ROOT = DRIVE_ROOT / "low_data_confirm_v2_20260517"
V1_ROOT = DRIVE_ROOT / "low_data_confirm_20260502"
CANONICAL_JSON_ROOT = ROOT / "experiments" / "low_data"
CANONICAL_CHECKPOINT_ROOT = ROOT / "models" / "low_data"
OUTPUT = ROOT / "experiments" / "drive_artifact_reconciliation_20260809.json"

RESULT_FIELDS = (
    "model",
    "family",
    "train_seed",
    "split_seed",
    "protocol_version",
    "params",
    "total_params",
    "trainable_params",
    "epochs",
    "dataset_sizes",
    "benchmark_axis",
    "train_fraction",
    "fraction_seed",
    "run_id",
    "best_val_acc",
    "test_acc",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def inventory(path: Path) -> dict[str, Any]:
    return {
        "path": relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def verify_grid(paths: list[Path]) -> None:
    experiment_paths = [path for path in paths if path.parent.name == "experiments_low_data"]
    if len(experiment_paths) != 56:
        raise RuntimeError(f"Expected 56 low-data JSON rows, found {len(experiment_paths)}")

    records = [read_json(path) for path in experiment_paths]
    current = [row for row in records if row.get("family") == "current-local"]
    thesis = [row for row in records if row.get("family") == "thesis-faithful"]
    if len(current) != 48 or len(thesis) != 8:
        raise RuntimeError(
            f"Unexpected family counts: current-local={len(current)}, thesis-faithful={len(thesis)}"
        )
    combinations = {
        (row["model"], float(row["train_fraction"]), int(row["train_seed"]))
        for row in current
    }
    expected = {
        (model, fraction, seed)
        for model in ("classical_conv", "non_trainable_quantum")
        for fraction in (0.10, 0.25, 0.50, 1.00)
        for seed in range(42, 48)
    }
    if combinations != expected:
        missing = sorted(expected - combinations)
        extra = sorted(combinations - expected)
        raise RuntimeError(f"Current-local grid mismatch; missing={missing}, extra={extra}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Copy verified originals/checkpoints into canonical paths and write the manifest.",
    )
    args = parser.parse_args()

    source_json_root = V2_ROOT / "experiments_low_data"
    source_json = sorted(source_json_root.glob("*.json"))
    verify_grid(source_json)

    reconciliation: list[dict[str, Any]] = []
    for source in source_json:
        destination = CANONICAL_JSON_ROOT / source.name
        if not destination.exists():
            raise FileNotFoundError(f"Missing canonical counterpart: {destination}")
        original = read_json(source)
        current = read_json(destination)
        mismatches = [
            field for field in RESULT_FIELDS if original.get(field) != current.get(field)
        ]
        if mismatches:
            raise RuntimeError(f"Result mismatch for {source.name}: {mismatches}")
        entry = {
            "source": relative(source),
            "destination": relative(destination),
            "source_sha256": sha256(source),
            "previous_destination_sha256": sha256(destination),
            "result_fields_verified_equal": True,
            "previous_artifact_status": current.get("artifact_status", "original"),
            "final_status": "canonical-byte-original-drive-json",
        }
        reconciliation.append(entry)
        if args.apply:
            shutil.copy2(source, destination)
            entry["destination_sha256_after_copy"] = sha256(destination)
            if entry["destination_sha256_after_copy"] != entry["source_sha256"]:
                raise RuntimeError(f"Copy verification failed for {destination}")

    checkpoint_sources = sorted((V1_ROOT / "models_low_data").glob("*.pth")) + sorted(
        (V2_ROOT / "models_low_data").glob("*.pth")
    )
    if len(checkpoint_sources) != 40 or len({path.name for path in checkpoint_sources}) != 40:
        raise RuntimeError(
            "Expected 40 unique seed-43--47 low-data checkpoints across the two Drive folders"
        )

    checkpoint_entries: list[dict[str, Any]] = []
    for source in checkpoint_sources:
        destination = CANONICAL_CHECKPOINT_ROOT / source.name
        entry = {
            "source": relative(source),
            "destination": relative(destination),
            "source_sha256": sha256(source),
            "bytes": source.stat().st_size,
            "final_status": "canonical-byte-original-drive-checkpoint",
        }
        checkpoint_entries.append(entry)
        if args.apply:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            entry["destination_sha256_after_copy"] = sha256(destination)
            if entry["destination_sha256_after_copy"] != entry["source_sha256"]:
                raise RuntimeError(f"Copy verification failed for {destination}")

    all_drive_files = sorted(path for path in DRIVE_ROOT.rglob("*") if path.is_file())
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "operation": "drive-artifact-reconciliation",
        "applied": args.apply,
        "scope_note": (
            "Low-data JSON metrics were verified against the prior reconstructed rows before "
            "replacement. V7 checkpoints were inventoried but not copied over the generic local "
            "V7 paths; the Drive folders still contain no raw V7 result JSON."
        ),
        "drive_inventory": {
            "root": relative(DRIVE_ROOT),
            "artifact_count": len(all_drive_files),
            "total_bytes": sum(path.stat().st_size for path in all_drive_files),
            "artifacts": [inventory(path) for path in all_drive_files],
        },
        "low_data_json_reconciliation": {
            "source_count": len(source_json),
            "byte_identical_before_copy": sum(
                entry["source_sha256"] == entry["previous_destination_sha256"]
                for entry in reconciliation
            ),
            "reconstructed_rows_replaced": sum(
                entry["previous_artifact_status"] == "reconstructed-from-captured-output"
                for entry in reconciliation
            ),
            "entries": reconciliation,
        },
        "low_data_checkpoint_reconciliation": {
            "source_count": len(checkpoint_sources),
            "entries": checkpoint_entries,
        },
        "v7": {
            "resumed_folder": relative(DRIVE_ROOT / "v7"),
            "clean_folder": relative(DRIVE_ROOT / "v7_clean_20260427"),
            "checkpoint_count": len(list((DRIVE_ROOT / "v7").glob("*.pth")))
            + len(list((DRIVE_ROOT / "v7_clean_20260427").glob("*.pth"))),
            "raw_result_json_found": False,
        },
    }

    if args.apply:
        OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {relative(OUTPUT)}")
    print(
        "Verified "
        f"{len(source_json)} low-data JSON rows, {len(checkpoint_sources)} checkpoints, "
        f"and {len(all_drive_files)} total Drive artifacts (apply={args.apply})."
    )


if __name__ == "__main__":
    main()
