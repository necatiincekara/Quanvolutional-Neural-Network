#!/usr/bin/env python3
"""Build a deterministic SHA-256 manifest for publication evidence.

This script performs no training and does not modify source artifacts. It records
the byte identity of result JSON, checkpoints, notebooks, figures, the manuscript,
and the current dataset snapshot so provenance can be audited before submission.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "submission_artifact_manifest_20260728.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def collect(patterns: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path for path in ROOT.glob(pattern) if path.is_file())
    paths.discard(OUTPUT)
    return sorted(paths, key=lambda path: path.relative_to(ROOT).as_posix())


def dataset_snapshot() -> dict[str, object]:
    files = collect(["set/train/**/*", "set/test/**/*"])
    digest = hashlib.sha256()
    split_counts = {"train": 0, "test": 0}
    total_bytes = 0
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        file_hash = sha256(path)
        digest.update(f"{relative}\t{path.stat().st_size}\t{file_hash}\n".encode())
        total_bytes += path.stat().st_size
        if relative.startswith("set/train/"):
            split_counts["train"] += 1
        elif relative.startswith("set/test/"):
            split_counts["test"] += 1
    return {
        "file_count": len(files),
        "split_file_counts": split_counts,
        "total_bytes": total_bytes,
        "combined_sha256": digest.hexdigest(),
        "method": "sha256 of sorted '<relative_path>\\t<size>\\t<file_sha256>\\n' rows",
    }


def main() -> None:
    evidence_paths = collect(
        [
            "experiments/*.json",
            "experiments/low_data/*.json",
            "models/*.pth",
            "models/low_data/*.pth",
            "*.ipynb",
            "paper/figures/*.png",
            "paper/figures/*.pdf",
            "paper/draft.md",
            "docs/EXPERIMENTS.md",
            "docs/LOW_DATA_SUMMARY.md",
            "docs/STATISTICAL_EVIDENCE_2026-05-17.md",
            "docs/CLASSIFICATION_METRICS_2026-07-28.md",
            "docs/LOW_DATA_CLASSIFICATION_METRICS_2026-08-09.md",
            "docs/ARTIFACT_PROVENANCE_2026-07-28.md",
            "docs/SUBMISSION_READINESS_CHECKLIST_2026-05-17.md",
            "requirements-publication-lock.txt",
        ]
    )
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/build_submission_manifest.py",
        "scope_note": (
            "Byte-identity manifest only. Canonical low-data rows are reconciled "
            "byte-original Drive artifacts. Reconstructed V7 rows retain their explicit "
            "artifact_status fields and are not converted into original artifacts."
        ),
        "artifact_count": len(evidence_paths),
        "artifacts": [record(path) for path in evidence_paths],
        "dataset_snapshot": dataset_snapshot(),
        "dataset_source": {
            "title": "Ottoman Turkish Characters",
            "authors": ["Alperen Özer", "Alp Bintuğ Uzun"],
            "url": "https://www.kaggle.com/datasets/alpbintuuzun/ottoman-turkish-characters",
            "kaggle_license_label": "GPL 2",
            "official_version": 1,
            "official_zip_sha256_verified_2026_08_09": "35b68d7f7e677e591d2305c573ce350914042f0f7e5b2fa79d2cb16415563885",
            "identity_check": "3894 local PNG files content-identical; no missing, extra, or mismatched images",
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)} with {len(evidence_paths)} artifacts")


if __name__ == "__main__":
    main()
