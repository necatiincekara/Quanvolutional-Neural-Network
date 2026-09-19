"""Read-only dataset investigation; exclusive additive outputs, no split/label changes.

Run: python scripts/investigate_dataset_closure.py --output NEW_DIRECTORY
The default output is this task's already-created evidence directory. Individual
outputs are exclusive writes, so reruns must use a fresh directory.
"""
import argparse
import collections
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "research/e0_e3_20260915"


def write(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def prefix(name):
    # These are candidate filename groups, NOT verified writers or pages.
    patterns = [r"TBMM-\d{3}", r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                r"koc_img\d{5}", r"talik_\d{2}", r"K\d{4}"]
    for pat in patterns:
        m = re.match(pat, name)
        if m:
            return m[0]
    return re.sub(r"0*\d{2}$", "", Path(name).stem)


def sheet(path, rows, title):
    im = Image.new("RGB", (1000, 48 + 166 * len(rows)), "white")
    d = ImageDraw.Draw(im)
    d.text((12, 12), title, fill="black")
    for k, row in enumerate(rows):
        y = 48 + 166 * k
        for j, rec in enumerate(row[:2]):
            pic = Image.open(ROOT / "set" / rec["relative_path"]).convert("L")
            im.paste(pic.resize((128, 128), Image.Resampling.NEAREST), (12 + j * 490, y))
            text = rec["relative_path"].replace("/", "/\n", 1)
            # Long zero runs are rendered as a separate full filename record in JSON.
            d.text((148 + j * 490, y + 8), text[:90], fill="black")
            d.text((148 + j * 490, y + 55), f'label {rec["label_code"]}; {rec["sample_id"][-12:]}', fill="black")
        d.text((12, y + 132), f"Row {k + 1}", fill="black")
    if path.exists():
        raise FileExistsError(path)
    im.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=ROOT / "research/cuda_dataset_closure_20260916")
    args = ap.parse_args(); out = args.output; out.mkdir(parents=True, exist_ok=True)
    wall = time.perf_counter(); cpu = time.process_time()
    config = {"scope": "diagnostic only; no training, label or split mutation", "manifest": str(BASE.relative_to(ROOT) / "dataset_manifest.json"),
              "manifest_sha256": sha(BASE / "dataset_manifest.json"), "protocol_sha256": sha(BASE / "protocol.json"),
              "script_sha256": sha(Path(__file__)), "access_date": "2026-09-16",
              "near_duplicate_method": "Exhaustive aligned native 32x32 grayscale L2 on distinct valid-label content groups; float32 BLAS shortlist with 1 MSE unit guard then float64 verification.",
              "thresholds_gray_0_255": {"tight_rmse": 5.0, "broad_rmse": 10.0},
              "review_limit_pairs": 24, "review_rule": "tight cross-protocol pairs first, then lowest RMSE cross-historical pairs; not automatic removal",
              "cpu_budget_seconds": 120, "threads": 1,
              "limitations": "No alignment correction, rotations, learned representations or writer inference. Pixel similarity is not shared source identity."}
    write(out / "investigation_config.json", config)
    import torch
    env = {"python": platform.python_version(), "os": platform.platform(), "torch": torch.__version__,
           "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "cuda_device_count": torch.cuda.device_count(),
           "nvidia_smi": shutil.which("nvidia-smi"), "gpu_model": None, "nvidia_driver": None,
           "mps_available": torch.backends.mps.is_available(), "versions": {}}
    for pkg in ["pennylane", "pennylane-lightning", "pennylane-lightning-gpu", "numpy", "opencv-python", "Pillow"]:
        try: env["versions"][pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError: env["versions"][pkg] = None
    env.update({"part_a_status": "BLOCKED_NO_CUDA" if not env["cuda_available"] else "CUDA_PRESENT_NOT_EXECUTED_BY_THIS_SCRIPT",
                "cuda_autocast_fp16": "not tested; no CUDA" if not env["cuda_available"] else "not tested",
                "cuda_bfloat16_operations": "not tested", "quantum_backend_executed": None, "differentiation_executed": None,
                "numerical_experiments_executed": 0, "gpu_seconds": 0})
    write(out / "environment.json", env)
    m = json.loads((BASE / "dataset_manifest.json").read_text()); p = json.loads((BASE / "protocol.json").read_text())
    records = m["records"]; byid = {r["record_id"]: r for r in records}
    arrays = {}
    metadata = {}
    for r in records:
        path = ROOT / "set" / r["relative_path"]
        assert sha(path) == r["byte_sha256"]
        arrays[r["record_id"]] = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        with Image.open(path) as im:
            metadata[r["record_id"]] = {"format": im.format, "mode": im.mode, "info": {k: str(v) for k,v in im.info.items()}}
    conflicts = []
    for g in m["duplicates"]["pixel_sha256"]:
        if not g["label_conflict"]: continue
        rr = [byid[i] for i in g["record_ids"]]
        assert all(np.array_equal(arrays[rr[0]["record_id"]], arrays[r["record_id"]]) for r in rr)
        conflicts.append({"group_id": f"C{len(conflicts)+1}", "sample_id": rr[0]["sample_id"],
                          "records": [{**r, "filename": Path(r["relative_path"]).name, "class_name": m["label_map"][r["label_code"]], "png_metadata": metadata[r["record_id"]]} for r in rr],
                          "exact_pixel_identity_verified": True, "exact_byte_identity": len({r["byte_sha256"] for r in rr}) == 1,
                          "cross_historical_split": g["cross_split"], "in_current_protocol": rr[0]["sample_id"] in p["representatives"],
                          "visual_status": "contact sheet generated; see separate manual review", "decision": "quarantine; insufficient evidence for corrected class", "corrected_label": None,
                          "confidence": {"identity_conflict": "high", "label_adjudication": "insufficient", "quarantine": "high"},
                          "rationale": "Same pixels and bytes carry incompatible valid source labels; no crop-to-source annotation or independent authoritative label is available."})
    write(out / "conflict_records.json", {"annotation_layer_only": True, "label_map": m["label_map"], "groups": conflicts})
    sheet(out / "conflict_contact_sheet.png", [g["records"] for g in conflicts], "Five identical-content label conflicts: original pixels enlarged 4x")
    groups = collections.defaultdict(list)
    for r in records: groups[prefix(Path(r["relative_path"]).name)].append(r)
    membership = {sid: split for split in ["train", "validation", "test"] for sid in p[split + "_ids"]}
    filename_evidence = []
    for k, rr in sorted(groups.items()):
        filename_evidence.append({"candidate_prefix": k, "count": len(rr), "label_codes": sorted({r["label_code"] for r in rr}),
                                 "original_splits": dict(collections.Counter(r["original_split"] for r in rr)),
                                 "new_protocol_splits": dict(collections.Counter(membership.get(r["sample_id"], "quarantine") for r in rr)),
                                 "record_ids": [r["record_id"] for r in rr], "classification": "plausible grouping requiring validation"})
    write(out / "filename_grouping.json", {"rules": "regex prefixes in script; koc_img + first 5 digits is explicitly provisional; prefixes are not asserted pages/writers", "groups": filename_evidence,
                                           "png_metadata_nonempty": {k:v for k,v in metadata.items() if v["info"]}})
    invalid = [r for r in records if not r["valid"]]
    rr = invalid[0]; x = arrays[rr["record_id"]].astype(np.float64)
    valid = [r for r in records if r["valid"]]
    nearest = sorted([(float(np.sqrt(np.mean((x-arrays[r["record_id"]])**2))),r) for r in valid], key=lambda t:(t[0],t[1]["relative_path"]))[:10]
    neighbors = sorted([r for r in records if prefix(Path(r["relative_path"]).name) == "TBMM-007"], key=lambda r:r["relative_path"])
    at = next(i for i,r in enumerate(neighbors) if r["record_id"] == rr["record_id"])
    write(out / "invalid_sample.json", {"record": rr, "png_metadata": metadata[rr["record_id"]], "pixel_distribution": {"min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()), "std": float(x.std()), "ink_pixels_below_128": int((x<128).sum())},
                                        "exact_other_matches": [r for r in records if r["sample_id"] == rr["sample_id"] and r["record_id"] != rr["record_id"]],
                                        "filename_neighbors": neighbors[max(0,at-3):at+4], "nearest_valid_by_aligned_rmse": [{"rmse_gray":d,"record":r} for d,r in nearest],
                                        "decision": "retain invalid/quarantined; insufficient evidence", "corrected_label": None, "confidence": "high for retaining quarantine; no justified class assignment"})
    sheet(out / "invalid_contact_sheet.png", [[rr, r] for _,r in nearest[:5]], "Invalid 00 at left; five nearest pixels at right (not ground truth)")
    # One representative for each distinct valid-label content. Keep conflicts in
    # the historical investigation, marking their absence from the new protocol.
    content = collections.defaultdict(list)
    for r in valid: content[r["sample_id"]].append(r)
    ids = sorted(content); reps = [min(content[i],key=lambda r:r["relative_path"]) for i in ids]
    a = np.stack([arrays[r["record_id"]] for r in reps]).reshape(len(ids),-1).astype(np.float32)
    n2 = np.sum(a*a,axis=1); candidates = []; checks = []
    near_cpu = time.process_time(); near_wall = time.perf_counter()
    with threadpool_limits(limits=1):
        for start in range(0,len(ids),128):
            if time.process_time()-near_cpu > 120: raise RuntimeError("Near-duplicate CPU cap exceeded; stop rather than silently truncate")
            d = (n2[start:start+128,None] + n2[None,:] - 2*a[start:start+128]@a.T)/1024
            ii,jj = np.where(d <= 101)
            for i0,j in zip(ii.tolist(),jj.tolist()):
                i=start+i0
                if j<=i:continue
                diff=a[i].astype(np.float64)-a[j].astype(np.float64); rmse=float(np.sqrt(np.mean(diff**2)))
                if rmse>10:continue
                historical_cross = any(r["original_split"]!=s["original_split"] for r in content[ids[i]] for s in content[ids[j]])
                sp = [membership.get(ids[i],"quarantine"),membership.get(ids[j],"quarantine")]
                ink1=a[i]<128;ink2=a[j]<128
                candidates.append({"sample_ids":[ids[i],ids[j]], "representatives":[reps[i]["relative_path"],reps[j]["relative_path"]],
                                   "record_ids":[[r["record_id"] for r in content[ids[i]]],[r["record_id"] for r in content[ids[j]]]],
                                   "labels":[sorted({r["label_code"] for r in content[ids[i]]}),sorted({r["label_code"] for r in content[ids[j]]})],
                                   "rmse_gray":rmse,"max_abs_gray":float(abs(diff).max()), "different_pixels":int(np.count_nonzero(diff)),
                                   "ink_iou":float((ink1&ink2).sum()/max(1,(ink1|ink2).sum())),
                                   "historical_cross_split":historical_cross,"protocol_splits":sp,
                                   "protocol_cross_split":sp[0]!=sp[1] and 'quarantine' not in sp, "review_status":"unreviewed"})
            # Independent exact distance checks of the matrix calculation.
            for i in [start,min(start+127,len(ids)-1)]:
                j=(i+29)%len(ids)
                exact=float(np.mean((a[i].astype(np.float64)-a[j])**2))
                checks.append(abs(float(d[i-start,j])-exact))
    candidates.sort(key=lambda c:(c["rmse_gray"],c["sample_ids"]))
    for i,c in enumerate(candidates):c["pair_id"]=f"ND{i+1:04d}"
    review = sorted(candidates,key=lambda c:(not c["protocol_cross_split"],not c["historical_cross_split"],c["rmse_gray"]))[:24]
    for c in review:c["review_status"]="contact sheet generated; see separate manual review"
    def summary(cc):
        return {"pairs":len(cc),"content_ids":len({s for c in cc for s in c["sample_ids"]}),
                "cross_historical_pairs":sum(c["historical_cross_split"] for c in cc),"cross_protocol_pairs":sum(c["protocol_cross_split"] for c in cc),
                "protocol_pair_counts":dict(collections.Counter('/'.join(sorted(c["protocol_splits"])) for c in cc)),
                "cross_protocol_affected_ids":{s:len({sid for c in cc if c["protocol_cross_split"] for sid,sp in zip(c["sample_ids"],c["protocol_splits"]) if sp==s}) for s in ['train','validation','test']}}
    write(out / "near_duplicate_queue.json", {"unique_valid_content_count":len(ids),"exhaustive_pair_count":len(ids)*(len(ids)-1)//2,"threshold_5":summary([c for c in candidates if c['rmse_gray']<=5]),"threshold_10":summary(candidates),"pairs":candidates,
                                              "review_order":[c['pair_id'] for c in review],"matrix_independent_check_max_mse_error":max(checks),"checked_matrix_distances":len(checks),
                                              "cpu_seconds":time.process_time()-near_cpu,"wall_seconds":time.perf_counter()-near_wall})
    recbypath={r['relative_path']:r for r in records}
    for chunk in range(0,len(review),8):
        sheet(out/f"near_duplicate_review_{chunk//8+1}.png",[[recbypath[f] for f in c['representatives']] for c in review[chunk:chunk+8]], "Near duplicate queue: " + ', '.join(c['pair_id'] for c in review[chunk:chunk+8]))
    write(out / "investigation_runtime.json", {"wall_seconds":time.perf_counter()-wall,"process_cpu_seconds":time.process_time()-cpu,"gpu_seconds":0,"training_runs":0})
    print(json.dumps({"conflict_groups":len(conflicts),"filename_groups":len(groups),"tight":summary([c for c in candidates if c['rmse_gray']<=5]),"broad":summary(candidates),"runtime":time.perf_counter()-wall}))


if __name__ == "__main__":
    main()
