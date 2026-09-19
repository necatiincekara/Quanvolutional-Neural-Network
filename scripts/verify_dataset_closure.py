"""Verify the bounded CUDA/data closure and write its final additive inventory.

No numerical experiment or training is launched. Run once after report completion.
"""
import ast
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "research/cuda_dataset_closure_20260916"
REPORT = ROOT / "research/ASTRA_CUDA_DATASET_CLOSURE_REPORT.md"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def main():
    wall, cpu = time.perf_counter(), time.process_time()
    final = EVIDENCE / "final_verification.json"
    inventory = EVIDENCE / "artifact_inventory.json"
    assert not final.exists() and not inventory.exists(), "Refuse to replace final evidence"
    before = read(EVIDENCE / "before_state.json")
    changed = [r["path"] for r in before["protected_files"]
               if not (ROOT / r["path"]).is_file() or sha(ROOT / r["path"]) != r["sha256"]]
    assert not changed, changed
    patch = subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)
    assert hashlib.sha256(patch).hexdigest() == before["diff_sha256"]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert head == before["head"]

    source = ROOT / "scripts/investigate_dataset_closure.py"
    config = read(EVIDENCE / "investigation_config.json")
    assert sha(source) == config["script_sha256"] == sha(EVIDENCE / "executed_investigator_source.py")
    ast.parse(source.read_text())
    ast.parse(Path(__file__).read_text())
    base = ROOT / "research/e0_e3_20260915"
    manifest = read(base / "dataset_manifest.json")
    protocol = read(base / "protocol.json")
    assert sha(base / "dataset_manifest.json") == config["manifest_sha256"]
    assert sha(base / "protocol.json") == config["protocol_sha256"]
    byid = {r["record_id"]: r for r in manifest["records"]}
    metadata = read(EVIDENCE / "kaggle_metadata.json")
    source_map = {f"{int(k):02d}": v.strip()
                  for k, v in re.findall(r"(\d+)\s*:\s*([^,\n`]+)", metadata["description"])}
    assert source_map == manifest["label_map"] and len(source_map) == 44
    archive = read(EVIDENCE / "source_archive_inventory.json")
    old_source = read(ROOT / "experiments/submission_artifact_manifest_20260728.json")["dataset_source"]
    assert archive["sha256"] == old_source["official_zip_sha256_verified_2026_08_09"]
    assert len(archive["entries"]) == 3894 and not archive["non_png_entries"]
    assert {r["path"] for r in archive["entries"]} == {"set/" + r["relative_path"] for r in manifest["records"]}

    review = read(EVIDENCE / "adjudication_review.json")
    conflicts = read(EVIDENCE / "conflict_records.json")["groups"]
    assert len(conflicts) == 5 and len(review["conflicts"]) == 5
    for group in conflicts:
        assert len(group["records"]) == 2 and group["sample_id"] not in protocol["representatives"]
        assert len({r["label"] for r in group["records"]}) == 2
        arrays = []
        for r in group["records"]:
            old = byid[r["record_id"]]
            assert all(r[k] == v for k, v in old.items())
            assert r["class_name"] == source_map[r["label_code"]]
            path = ROOT / "set" / r["relative_path"]
            assert sha(path) == r["byte_sha256"]
            arrays.append(cv2.imread(str(path), 0))
        assert np.array_equal(*arrays)
    assert all(r["corrected_label"] is None for r in review["conflicts"])
    invalid = read(EVIDENCE / "invalid_sample.json")
    assert invalid["record"]["label_code"] == "00" and invalid["corrected_label"] is None
    assert invalid["record"]["sample_id"] not in protocol["representatives"]
    assert not invalid["exact_other_matches"] and review["invalid"]["corrected_label"] is None

    queue = read(EVIDENCE / "near_duplicate_queue.json")
    assert len(queue["pairs"]) == 4
    membership = {sid: split for split in ["train", "validation", "test"] for sid in protocol[split + "_ids"]}
    for pair in queue["pairs"]:
        a, b = [cv2.imread(str(ROOT / "set" / p), 0).astype(np.float64) for p in pair["representatives"]]
        assert abs(float(np.sqrt(np.mean((a-b)**2))) - pair["rmse_gray"]) < 1e-12
        assert int(np.count_nonzero(a-b)) == pair["different_pixels"]
        assert [membership.get(s, "quarantine") for s in pair["sample_ids"]] == pair["protocol_splits"]
    assert sum(p["protocol_cross_split"] for p in queue["pairs"]) == 3
    assert set(queue["review_order"]) == {r["pair_id"] for r in review["near_duplicates"]}
    env = read(EVIDENCE / "environment.json")
    assert not env["cuda_available"] and env["numerical_experiments_executed"] == 0
    assert env["gpu_seconds"] == 0
    assert read(EVIDENCE / "investigation_runtime.json")["training_runs"] == 0

    report = REPORT.read_text()
    assert re.findall(r"^## (\d+)\.", report, re.M) == [str(i) for i in range(1, 18)]
    assert report.rstrip().endswith("STOP — CUDA/data closure complete. Awaiting authorization for E4 or another explicitly approved stage.")
    links = [REPORT.parent / p for p in re.findall(r"\]\(([^)]+)\)", report) if not p.startswith("https://")]
    assert all(p.exists() or p in [final, inventory] for p in links)
    for path in EVIDENCE.glob("*.json"):
        read(path)
    result = {"completion_date": "2026-09-17", "head": head, "branch": before["branch"],
              "protected_files_checked": len(before["protected_files"]), "protected_files_changed": changed,
              "tracked_patch_unchanged": True, "tracked_patch_sha256": hashlib.sha256(patch).hexdigest(),
              "source_identity_verified": True, "source_map_verified_classes": 44,
              "conflict_groups_verified": 5, "conflict_occurrences_quarantined": 10,
              "invalid_sample_quarantined": True, "near_pairs_metric_verified": 4,
              "cross_protocol_candidate_pairs": 3, "report_numbered_sections": 17,
              "gpu_seconds": 0, "training_runs": 0,
              "status": subprocess.check_output(["git", "status", "--short", "--untracked-files=all"], cwd=ROOT, text=True),
              "process_cpu_seconds": time.process_time()-cpu, "wall_seconds": time.perf_counter()-wall}
    write(final, result)

    purposes = {
        "before_state.json": "Starting worktree and 5075 protected-file hashes",
        "before_tracked.patch": "Preserved pre-existing tracked changes",
        "adjudication_review.json": "Visual adjudication, confidence, null corrected labels and foreground-statistic addendum",
        "author_repository_listing.json": "Bounded original-author repository metadata search response",
        "source_catalog.json": "Exact primary-source URLs, access date, scope and failures",
        "source_archive_inventory.json": "Original release ZIP hash and complete 3894-entry inventory, without repackaged images",
        "primary_publication_access.json": "Publisher PDF retrieval identity and access record",
        "kaggle_metadata.json": "Original uploader release description, class map and license label",
        "kaggle_search_response.json": "Original Kaggle dataset discovery response",
        "environment.json": "CUDA availability and installed version probe",
        "environment_fidelity.json": "Qualified historical/current environment comparison",
        "historical_environment_extract.json": "Notebook output excerpts with cell/output indexes",
        "cuda_handoff.json": "Static unexecuted minimum CUDA diagnostic specification",
        "conflict_records.json": "Every conflicting-content occurrence, identity, label and initial quarantine proposal",
        "conflict_contact_sheet.png": "Five identity-conflict groups for visual inspection",
        "invalid_sample.json": "Invalid 00 identity, metadata, filename neighbors and nearest valid rasters",
        "invalid_contact_sheet.png": "Invalid raster and bounded visual comparison set",
        "filename_grouping.json": "Explicitly provisional filename-prefix inventory, not writer assignments",
        "near_duplicate_queue.json": "Four conservative aligned-distance review candidates and counts",
        "near_duplicate_review_1.png": "Visual review of all four candidate pairs",
        "investigation_config.json": "Pre-execution thresholds, cap and source hashes",
        "investigation_runtime.json": "Measured CPU/wall time; zero GPU and training runs",
        "executed_investigator_source.py": "Exact executed investigator snapshot",
        "historical_config.txt": "Pre-fix Git config snapshot",
        "historical_enhanced_training.txt": "Pre-fix Git trainer snapshot",
        "historical_trainable_quantum_model.txt": "Pre-fix Git circuit/model snapshot",
        "commands.md": "Reproduction commands, access methods, failed attempts and scope boundary",
        "final_verification.json": "Completed integrity/evidence checks and completion worktree state",
        "artifact_inventory.json": "Complete new-deliverable index; self hash intentionally null",
        "ASTRA_CUDA_DATASET_CLOSURE_REPORT.md": "Required 17-section final scientific report",
        "investigate_dataset_closure.py": "Read-only data/environment diagnostic entrypoint",
        "verify_dataset_closure.py": "This final preservation/evidence verifier"}
    files = sorted([*EVIDENCE.glob("*"), REPORT, source, Path(__file__).resolve(), inventory])
    entries = []
    for path in files:
        assert path.name in purposes, path
        entries.append({"path": str(path.relative_to(ROOT)), "purpose": purposes[path.name],
                        "bytes": None if path == inventory else path.stat().st_size,
                        "sha256": None if path == inventory else sha(path)})
    write(inventory, {"completion_date": "2026-09-17", "preexisting_files_modified": [], "files": entries,
                      "temporary_reading_intermediates": "Publisher PDF/renderings and thesis text in /private/tmp are not deliverables; source URL/hash retained",
                      "runtime_exclusions": "Interpreter bytecode and library caches"})
    assert all(p.exists() for p in links)
    print(json.dumps({k:v for k,v in result.items() if k != "status"}, indent=2))
    print(f"New deliverable files: {len(entries)}")


if __name__ == "__main__":
    main()
