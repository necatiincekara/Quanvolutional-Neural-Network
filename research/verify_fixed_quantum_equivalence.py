"""Audit the current fixed RX/Rot/CNOT-chain features without training.

Run from the repository root with venv/bin/python. The reference functions are
extracted from actual source ASTs, not independently reimplemented. This proves
only the specified expectation map's equivalence, not V7 re-uploading or OCR
performance. Writes a new diagnostic JSON; refuses to replace existing evidence.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import pennylane as qml
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_reference(relative_path, name):
    path = ROOT / relative_path
    node = next(n for n in ast.walk(ast.parse(path.read_text()))
                if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {"qml": qml, "n_qubits": 4,
                 "config": type("Config", (), {"N_QUBITS": 4})}
    exec(compile(module, str(path), "exec"), namespace)
    return qml.QNode(namespace[name], qml.device("default.qubit", wires=4),
                     interface="torch", diff_method="backprop")


def classical_map(x, weights):
    """Exact measured map; final Rot omega angles cannot affect these Zs."""
    phi, theta = weights[:, 0], weights[:, 1]
    local_z = (torch.cos(theta) * torch.cos(x)
               - torch.sin(theta) * torch.sin(phi) * torch.sin(x))
    return torch.cumprod(local_z, dim=-1)


def reference_values(circuit, x, weights):
    return torch.stack(circuit(x, weights), dim=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    start = time.perf_counter()
    torch.set_num_threads(1)
    refs = {
        "fixed_preprocessor": extract_reference("train_ablation_local.py", "fixed_circuit"),
        "current_base_circuit": extract_reference("src/model.py", "quanv_circuit"),
    }
    rng = np.random.default_rng(20260908)
    real_patches, image_manifest = [], []
    # Training pixels only; use original 2x2 crops, no test labels or selection.
    paths = sorted((ROOT / "set/train").glob("*.png"))[:16]
    for path in paths:
        array = np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0
        if min(array.shape) >= 2:
            row, col = (array.shape[0] - 2) // 2, (array.shape[1] - 2) // 2
            real_patches.append(array[row:row+2, col:col+2].reshape(4))
            image_manifest.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
    cases = {
        "raw_range": rng.uniform(0, 1, (16, 4)),
        "feature_range": rng.uniform(-4*np.pi, 4*np.pi, (16, 4)),
        "edge_cases": np.array([[0]*4, [1]*4, [np.pi/2]*4, [-np.pi]*4]),
        "training_image_crops": np.asarray(real_patches),
    }
    errors = {key: {"forward_max_abs": 0.0, "input_jacobian_max_abs": 0.0,
                    "weight_jacobian_max_abs": 0.0, "omega_gradient_max_abs": 0.0}
              for key in refs}
    total_instances = 0
    for seed in range(42, 48):
        # Same RandomState sequence and distribution as precompute_quantum_features.
        circuit_rng = np.random.RandomState(seed)
        for _ in range(4):
            weights = torch.tensor(circuit_rng.uniform(-np.pi, np.pi, (4, 3)), dtype=torch.float64)
            for values in cases.values():
                x = torch.tensor(values, dtype=torch.float64)
                expected = classical_map(x, weights)
                total_instances += len(x)
                for name, circuit in refs.items():
                    observed = reference_values(circuit, x, weights)
                    err = float((expected-observed).abs().max())
                    errors[name]["forward_max_abs"] = max(errors[name]["forward_max_abs"], err)
            x = torch.tensor(rng.uniform(-np.pi, np.pi, 4), dtype=torch.float64)
            cj = torch.autograd.functional.jacobian(classical_map, (x, weights))
            for name, circuit in refs.items():
                qj = torch.autograd.functional.jacobian(
                    lambda a, b: reference_values(circuit, a, b), (x, weights))
                for key, left, right in zip(
                    ["input_jacobian_max_abs", "weight_jacobian_max_abs"], cj, qj):
                    errors[name][key] = max(errors[name][key], float((left-right).abs().max()))
                errors[name]["omega_gradient_max_abs"] = max(
                    errors[name]["omega_gradient_max_abs"], float(qj[1][:, :, 2].abs().max()))
    passed = all(value < 1e-10 for row in errors.values() for value in row.values())
    sources = ["train_ablation_local.py", "src/model.py", str(Path(__file__).relative_to(ROOT))]
    result = {
        "experiment_id": "astra_fixed_feature_equivalence_20260908",
        "kind": "algebraic_map_verification_not_training", "passed": passed,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {p: sha256(ROOT/p) for p in sources},
        "scope": "RX encoding; local Rot; forward CNOT chain; single-wire Z expectations",
        "excluded": ["V7 reuploading", "full OCR prediction replay", "historical cache byte equality", "hardware noise"],
        "formula": "cumprod(cos(theta)*cos(x)-sin(theta)*sin(phi)*sin(x))",
        "dtype": "float64", "backend": "default.qubit", "diff_method": "backprop",
        "circuit_seeds": list(range(42, 48)), "filters_per_seed": 4,
        "input_seed": 20260908, "instances_per_reference": total_instances,
        "jacobian_cases_per_reference": 24, "threshold": 1e-10,
        "errors": errors, "training_image_crops": image_manifest,
        "versions": {"torch": torch.__version__, "pennylane": qml.__version__,
                     "numpy": np.__version__, "python": platform.python_version()},
        "platform": platform.platform(), "runtime_seconds": time.perf_counter()-start,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: result[k] for k in ["passed", "instances_per_reference", "errors", "runtime_seconds"]}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
