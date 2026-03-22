#!/usr/bin/env python3
"""
Aggregate publication benchmark JSON results into summary tables.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from statistics import mean, stdev


REFERENCE_RECORDS = [
    {
        "model": "V7_trainable_quantum_documented",
        "source": "docs/notebook",
        "family": "trainable-quantum-case-study",
        "best_val_acc": 67.35,
        "test_acc": 65.02,
        "total_params": 87798,
        "n_runs": 1,
        "notes": "Documented stabilized V7 result from docs/notebook output.",
    },
    {
        "model": "V4_historical_reference",
        "source": "historical-docs",
        "family": "historical-reference",
        "best_val_acc": 8.75,
        "test_acc": None,
        "total_params": None,
        "n_runs": 1,
        "notes": "Historical non-trainable baseline reference.",
    },
]

LEGACY_MODEL_DEFAULTS = {
    "classical_conv": {"source": "repo-local-ablation", "family": "current-local"},
    "param_linear": {"source": "repo-local-ablation", "family": "current-local"},
    "non_trainable_quantum": {"source": "repo-local-ablation", "family": "current-local"},
    "thesis_cnn3": {"source": "thesis-faithful", "family": "thesis-faithful"},
    "thesis_cnniiii": {"source": "thesis-faithful", "family": "thesis-faithful"},
    "thesis_hqnn2": {"source": "thesis-faithful", "family": "thesis-faithful"},
    "thesis_hqnn3": {"source": "thesis-faithful", "family": "thesis-faithful"},
}


def load_result_files():
    records = []
    dedupe = {}
    for path in sorted(glob.glob("experiments/*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if "model" not in data:
            continue
        defaults = LEGACY_MODEL_DEFAULTS.get(data["model"], {})
        data.setdefault("source", defaults.get("source", "unknown"))
        data.setdefault("family", defaults.get("family", "unknown"))
        key = data.get("run_id", path)
        preferred = dedupe.get(key)
        if preferred is None or ("_seed" in path and "_seed" not in preferred["__path"]):
            data["__path"] = path
            dedupe[key] = data
    records.extend(dedupe.values())
    for ref in REFERENCE_RECORDS:
        ref = dict(ref)
        ref["__path"] = "reference"
        records.append(ref)
    return records


def summarize(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["model"]].append(record)

    summary_rows = []
    for model, items in sorted(grouped.items()):
        test_vals = [item["test_acc"] for item in items if item.get("test_acc") is not None]
        val_vals = [item["best_val_acc"] for item in items if item.get("best_val_acc") is not None]
        row = {
            "model": model,
            "family": items[0].get("family", "unknown"),
            "source": items[0].get("source", "unknown"),
            "runs": len(items),
            "best_val_acc_mean": round(mean(val_vals), 2) if val_vals else None,
            "best_val_acc_std": round(stdev(val_vals), 2) if len(val_vals) > 1 else 0.0 if val_vals else None,
            "test_acc_mean": round(mean(test_vals), 2) if test_vals else None,
            "test_acc_std": round(stdev(test_vals), 2) if len(test_vals) > 1 else 0.0 if test_vals else None,
            "total_params": items[0].get("total_params") or items[0].get("params", {}).get("total"),
            "notes": items[0].get("notes", ""),
        }
        summary_rows.append(row)
    return summary_rows


def family_sections(summary_rows):
    buckets = defaultdict(list)
    for row in summary_rows:
        buckets[row["family"]].append(row)
    return buckets


def fmt_metric(mean_value, std_value):
    if mean_value is None:
        return "-"
    if std_value is None:
        return f"{mean_value:.2f}"
    return f"{mean_value:.2f} ± {std_value:.2f}"


def to_markdown(summary_rows):
    sections = family_sections(summary_rows)
    lines = [
        "# Benchmark Summary",
        "",
        "Auto-generated from `experiments/*.json` plus documented reference rows.",
        "",
    ]
    for family in sorted(sections):
        lines.append(f"## {family}")
        lines.append("")
        lines.append("| Model | Source | Runs | Best Val | Test | Params | Notes |")
        lines.append("|---|---|---:|---:|---:|---:|---|")
        for row in sections[family]:
            lines.append(
                "| {model} | {source} | {runs} | {val} | {test} | {params} | {notes} |".format(
                    model=row["model"],
                    source=row["source"],
                    runs=row["runs"],
                    val=fmt_metric(row["best_val_acc_mean"], row["best_val_acc_std"]),
                    test=fmt_metric(row["test_acc_mean"], row["test_acc_std"]),
                    params=row["total_params"] if row["total_params"] is not None else "-",
                    notes=row["notes"] or "-",
                )
            )
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate benchmark result JSON files")
    parser.add_argument("--json-out", default="experiments/benchmark_summary.json")
    parser.add_argument("--md-out", default="docs/BENCHMARK_SUMMARY.md")
    args = parser.parse_args()

    records = load_result_files()
    summary_rows = summarize(records)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
    with open(args.md_out, "w", encoding="utf-8") as f:
        f.write(to_markdown(summary_rows))
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.md_out}")


if __name__ == "__main__":
    main()
