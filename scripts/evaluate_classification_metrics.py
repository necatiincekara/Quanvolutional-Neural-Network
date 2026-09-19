#!/usr/bin/env python3
"""Evaluate saved publication checkpoints with class-aware OCR metrics.

This script performs inference only. It never trains or modifies checkpoints.
It verifies each reconstructed top-1 accuracy against the corresponding result
JSON before writing aggregate macro-F1, balanced accuracy, weighted-F1,
per-class recall, and confusion-matrix evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import config
from src.ablation_models import (
    ClassicalBaselineNet,
    NonTrainableQuantumClassicalNet,
    ParamMatchedLinearNet,
)
from src.benchmark_protocol import load_raw_tensors, set_global_seed
from src.modern_baselines import create_modern_baseline
from src.thesis_models import create_thesis_model


DEFAULT_MODELS = [
    "classical_conv",
    "param_linear",
    "non_trainable_quantum",
    "thesis_cnn3",
    "thesis_cnniiii",
    "thesis_hqnn2",
    "resnet18_cifar_gray",
]
DEFAULT_SEEDS = [42, 43, 44]

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "classical_conv": {
        "family": "current-local",
        "result": "experiments/ablation_classical_conv_seed{seed}_split42.json",
        "checkpoint": "models/best_ablation_classical_conv_seed{seed}_split42.pth",
        "builder": lambda: ClassicalBaselineNet(num_classes=config.NUM_CLASSES),
        "input": "raw",
    },
    "param_linear": {
        "family": "current-local",
        "result": "experiments/ablation_param_linear_seed{seed}_split42.json",
        "checkpoint": "models/best_ablation_param_linear_seed{seed}_split42.pth",
        "builder": lambda: ParamMatchedLinearNet(num_classes=config.NUM_CLASSES),
        "input": "raw",
    },
    "non_trainable_quantum": {
        "family": "current-local",
        "result": "experiments/ablation_non_trainable_quantum_seed{seed}_split42.json",
        "checkpoint": "models/best_ablation_non_trainable_quantum_seed{seed}_split42.pth",
        "builder": lambda: NonTrainableQuantumClassicalNet(
            in_channels=16, num_classes=config.NUM_CLASSES
        ),
        "input": "cached",
    },
    "thesis_cnn3": {
        "family": "thesis-faithful",
        "result": "experiments/publication_thesis_cnn3_seed{seed}_split42.json",
        "checkpoint": "models/best_publication_thesis_cnn3_seed{seed}_split42.pth",
        "builder": lambda: create_thesis_model("thesis_cnn3", config.NUM_CLASSES),
        "input": "raw",
    },
    "thesis_cnniiii": {
        "family": "thesis-faithful",
        "result": "experiments/publication_thesis_cnniiii_seed{seed}_split42.json",
        "checkpoint": "models/best_publication_thesis_cnniiii_seed{seed}_split42.pth",
        "builder": lambda: create_thesis_model("thesis_cnniiii", config.NUM_CLASSES),
        "input": "raw",
    },
    "thesis_hqnn2": {
        "family": "thesis-faithful",
        "result": "experiments/publication_thesis_hqnn2_seed{seed}_split42.json",
        "checkpoint": "models/best_publication_thesis_hqnn2_seed{seed}_split42.pth",
        "builder": lambda: create_thesis_model("thesis_hqnn2", config.NUM_CLASSES),
        "input": "cached",
    },
    "resnet18_cifar_gray": {
        "family": "modern-classical",
        "result": "experiments/publication_resnet18_cifar_gray_seed{seed}_split42.json",
        "checkpoint": "models/best_publication_resnet18_cifar_gray_seed{seed}_split42.pth",
        "builder": lambda: create_modern_baseline(
            "resnet18_cifar_gray", config.NUM_CLASSES
        ),
        "input": "raw",
    },
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def class_names() -> list[str]:
    return [config.TAGS[f"{index:02d}"] for index in range(1, config.NUM_CLASSES + 1)]


def cached_test_tensors(result: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    cache_key = result.get("cache_metadata", {}).get("cache_key")
    if not cache_key:
        raise ValueError(f"Missing cache key for {result.get('model')} seed {result.get('train_seed')}")
    root = Path("experiments/quantum_cache")
    features = root / f"{cache_key}_test_features.npy"
    labels = root / f"{cache_key}_test_labels.npy"
    if not features.exists() or not labels.exists():
        raise FileNotFoundError(f"Missing cached test tensors for cache key {cache_key}")
    return (
        torch.tensor(np.load(features), dtype=torch.float32),
        torch.tensor(np.load(labels), dtype=torch.long),
    )


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray, classes: int) -> np.ndarray:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    return matrix


def metrics_from_confusion(matrix: np.ndarray) -> dict[str, Any]:
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=support > 0,
    )
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=predicted > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(recall),
        where=(precision + recall) > 0,
    )
    total = int(matrix.sum())
    accuracy = float(true_positive.sum() / total) if total else 0.0
    weighted_f1 = float(np.average(f1, weights=support)) if support.sum() else 0.0
    return {
        "accuracy": round(100.0 * accuracy, 4),
        "macro_f1": round(100.0 * float(f1.mean()), 4),
        "balanced_accuracy": round(100.0 * float(recall.mean()), 4),
        "weighted_f1": round(100.0 * weighted_f1, 4),
        "per_class_precision": [round(100.0 * float(value), 4) for value in precision],
        "per_class_recall": [round(100.0 * float(value), 4) for value in recall],
        "per_class_f1": [round(100.0 * float(value), 4) for value in f1],
        "support": [int(value) for value in support],
    }


def evaluate_checkpoint(
    builder: Callable[[], torch.nn.Module],
    checkpoint: Path,
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray]:
    model = builder().to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    loader = DataLoader(TensorDataset(features, labels), batch_size=batch_size, shuffle=False)
    predicted_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for images, batch_labels in loader:
            logits = model(images.to(device))
            predicted_batches.append(logits.argmax(dim=1).cpu())
            label_batches.append(batch_labels.cpu())
    all_predictions = torch.cat(predicted_batches).numpy()
    all_labels = torch.cat(label_batches).numpy()
    matrix = confusion_matrix(all_labels, all_predictions, config.NUM_CLASSES)
    return metrics_from_confusion(matrix), matrix


def mean_std(values: list[float]) -> tuple[float, float]:
    return round(mean(values), 2), round(stdev(values), 2) if len(values) > 1 else 0.0


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_names = ["accuracy", "macro_f1", "balanced_accuracy", "weighted_f1"]
    aggregate: dict[str, Any] = {}
    for name in scalar_names:
        metric_mean, metric_std = mean_std([float(run[name]) for run in runs])
        aggregate[f"{name}_mean"] = metric_mean
        aggregate[f"{name}_std"] = metric_std

    recall_by_class = list(zip(*(run["per_class_recall"] for run in runs)))
    aggregate["per_class_recall_mean"] = [round(mean(values), 2) for values in recall_by_class]
    aggregate["per_class_recall_std"] = [
        round(stdev(values), 2) if len(values) > 1 else 0.0 for values in recall_by_class
    ]
    aggregate["confusion_matrix_sum"] = np.sum(
        [np.asarray(run["confusion_matrix"], dtype=np.int64) for run in runs], axis=0
    ).tolist()
    return aggregate


def metric_text(row: dict[str, Any], name: str) -> str:
    return f"{row[f'{name}_mean']:.2f} ± {row[f'{name}_std']:.2f}"


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Class-Aware Classification Evidence",
        "",
        f"**Date:** {report['generated_date']}",
        "",
        "Generated by inference from the saved best-validation checkpoints. No training is performed.",
        "Top-1 accuracy is required to match the corresponding experiment JSON within the configured tolerance before a row is accepted.",
        "",
        "## Three-Seed Test Metrics",
        "",
        "| Family | Model | Runs | Accuracy | Macro-F1 | Balanced Accuracy | Weighted-F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["summary"]:
        lines.append(
            "| {family} | `{model}` | {runs} | {accuracy} | {macro_f1} | {balanced} | {weighted} |".format(
                family=row["family"],
                model=row["model"],
                runs=row["runs"],
                accuracy=metric_text(row, "accuracy"),
                macro_f1=metric_text(row, "macro_f1"),
                balanced=metric_text(row, "balanced_accuracy"),
                weighted=metric_text(row, "weighted_f1"),
            )
        )

    names = report["class_names"]
    lines.extend(
        [
            "",
            "## Lowest-Recall Classes",
            "",
            "The five lowest mean-recall classes are listed per model to expose class-imbalance and morphology-sensitive failure modes.",
            "",
            "| Model | Lowest-recall classes (mean recall %) |",
            "|---|---|",
        ]
    )
    for row in report["summary"]:
        pairs = sorted(enumerate(row["per_class_recall_mean"]), key=lambda item: item[1])[:5]
        formatted = ", ".join(f"{names[index]}: {value:.2f}" for index, value in pairs)
        lines.append(f"| `{row['model']}` | {formatted} |")

    lines.extend(
        [
            "",
            "## Provenance",
            "",
            "- Predictions come from the repository's saved per-seed best-validation checkpoints.",
            "- Raw-image models use the fixed repository test split.",
            "- Quantum-preprocessing models use the cache key recorded in each experiment JSON.",
            "- Per-run confusion matrices and all 44 per-class metrics are stored in the companion JSON.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_SPECS), default=DEFAULT_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["cpu", "auto"], default="cpu")
    parser.add_argument("--accuracy-tolerance", type=float, default=0.03)
    parser.add_argument("--json-out", default="experiments/classification_metrics_20260728.json")
    parser.add_argument("--md-out", default="docs/CLASSIFICATION_METRICS_2026-07-28.md")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    _, _, raw_test_features, raw_test_labels = load_raw_tensors(config.IMAGE_SIZE)
    class_labels = class_names()
    run_rows: list[dict[str, Any]] = []

    for model_name in args.models:
        spec = MODEL_SPECS[model_name]
        for seed in args.seeds:
            set_global_seed(seed)
            result_path = Path(spec["result"].format(seed=seed, split=args.split_seed))
            checkpoint_path = Path(spec["checkpoint"].format(seed=seed, split=args.split_seed))
            if not result_path.exists() or not checkpoint_path.exists():
                raise FileNotFoundError(f"Missing result/checkpoint pair: {result_path}, {checkpoint_path}")
            result = read_json(result_path)
            if result.get("protocol_version") != "publication_v1":
                raise ValueError(f"Unexpected protocol in {result_path}")
            if int(result.get("split_seed")) != args.split_seed:
                raise ValueError(f"Unexpected split seed in {result_path}")

            if spec["input"] == "cached":
                features, labels = cached_test_tensors(result)
            else:
                features, labels = raw_test_features, raw_test_labels

            metrics, matrix = evaluate_checkpoint(
                spec["builder"], checkpoint_path, features, labels, args.batch_size, device
            )
            artifact_accuracy = float(result["test_acc"])
            if abs(metrics["accuracy"] - artifact_accuracy) > args.accuracy_tolerance:
                raise RuntimeError(
                    f"Accuracy mismatch for {model_name} seed {seed}: "
                    f"checkpoint={metrics['accuracy']:.4f}, JSON={artifact_accuracy:.4f}"
                )
            run_rows.append(
                {
                    "family": spec["family"],
                    "model": model_name,
                    "seed": seed,
                    "split_seed": args.split_seed,
                    "protocol_version": result["protocol_version"],
                    "result_json": str(result_path),
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_accuracy_verified_against_json": True,
                    **metrics,
                    "confusion_matrix": matrix.tolist(),
                }
            )
            print(
                f"Verified {model_name} seed {seed}: "
                f"accuracy={metrics['accuracy']:.2f}, macro-F1={metrics['macro_f1']:.2f}"
            )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in run_rows:
        grouped[(run["family"], run["model"])].append(run)

    summary = []
    for (family, model_name), runs in grouped.items():
        aggregate = aggregate_runs(runs)
        summary.append(
            {
                "family": family,
                "model": model_name,
                "runs": len(runs),
                "seeds": sorted(run["seed"] for run in runs),
                **aggregate,
            }
        )

    family_order = {"modern-classical": 0, "thesis-faithful": 1, "current-local": 2}
    summary.sort(
        key=lambda row: (
            family_order.get(row["family"], 99),
            -row["accuracy_mean"],
        )
    )
    report = {
        "generated_date": "2026-07-28",
        "artifact": "class-aware-inference-evaluation",
        "protocol_version": "publication_v1",
        "split_seed": args.split_seed,
        "class_names": class_labels,
        "source_priority": "saved checkpoints + experiment JSON + cached quantum features",
        "training_performed": False,
        "summary": summary,
        "runs": run_rows,
    }

    json_out = Path(args.json_out)
    md_out = Path(args.md_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    with json_out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    with md_out.open("w", encoding="utf-8") as handle:
        handle.write(to_markdown(report))
    print(f"Wrote {json_out}")
    print(f"Wrote {md_out}")


if __name__ == "__main__":
    main()
