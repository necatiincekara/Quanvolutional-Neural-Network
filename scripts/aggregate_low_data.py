#!/usr/bin/env python3
"""Aggregate low-data scaling pilot results into JSON and Markdown summaries."""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from statistics import mean, stdev
from typing import Any


COMPARISONS = [
    ("current-local", "classical_conv", "non_trainable_quantum"),
    ("thesis-faithful", "thesis_cnniiii", "thesis_hqnn2"),
]


def load_records(pattern: str) -> list[dict[str, Any]]:
    records = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("benchmark_axis") != "low-data-scaling":
            continue
        records.append(data | {"__path": path})
    return records


def metric(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return round(mean(values), 2), round(stdev(values), 2) if len(values) > 1 else 0.0


def fmt(value: float | None, std_value: float | None = None) -> str:
    if value is None:
        return "-"
    if std_value is None:
        return f"{value:.2f}"
    return f"{value:.2f} ± {std_value:.2f}"


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["model"], float(record["train_fraction"]))].append(record)

    rows = []
    for (model, fraction), items in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        test_vals = [item["test_acc"] for item in items if item.get("test_acc") is not None]
        val_vals = [item["best_val_acc"] for item in items if item.get("best_val_acc") is not None]
        train_sizes = [item.get("dataset_sizes", {}).get("train") for item in items]
        protocol_versions = sorted({item.get("protocol_version") for item in items if item.get("protocol_version")})
        test_mean, test_std = metric(test_vals)
        val_mean, val_std = metric(val_vals)
        rows.append(
            {
                "family": items[0].get("family", "unknown"),
                "model": model,
                "train_fraction": fraction,
                "runs": len(items),
                "seeds": sorted(item.get("train_seed") for item in items),
                "train_size": train_sizes[0] if train_sizes else None,
                "best_val_acc_mean": val_mean,
                "best_val_acc_std": val_std,
                "test_acc_mean": test_mean,
                "test_acc_std": test_std,
                "total_params": items[0].get("total_params"),
                "source": items[0].get("source", "unknown"),
                "protocol_version": protocol_versions[0] if len(protocol_versions) == 1 else "mixed",
                "protocol_versions": protocol_versions,
            }
        )
    return rows


def row_index(rows: list[dict[str, Any]]) -> dict[tuple[str, float], dict[str, Any]]:
    return {(row["model"], row["train_fraction"]): row for row in rows}


def comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = row_index(rows)
    out = []
    for family, classical_model, quantum_model in COMPARISONS:
        fractions = sorted(
            {
                fraction
                for model, fraction in indexed
                if model in {classical_model, quantum_model}
            }
        )
        full_gap = None
        if (classical_model, 1.0) in indexed and (quantum_model, 1.0) in indexed:
            classical_full = indexed[(classical_model, 1.0)]["test_acc_mean"]
            quantum_full = indexed[(quantum_model, 1.0)]["test_acc_mean"]
            if classical_full is not None and quantum_full is not None:
                full_gap = round(classical_full - quantum_full, 2)

        for fraction in fractions:
            classical = indexed.get((classical_model, fraction))
            quantum = indexed.get((quantum_model, fraction))
            if not classical or not quantum:
                continue
            c_test = classical["test_acc_mean"]
            q_test = quantum["test_acc_mean"]
            if c_test is None or q_test is None:
                continue
            gap = round(c_test - q_test, 2)
            quantum_wins = q_test > c_test
            within_two = gap <= 2.0
            low_fraction_gap_halved = (
                fraction in {0.10, 0.25}
                and full_gap is not None
                and full_gap > 0
                and gap <= full_gap * 0.5
            )
            out.append(
                {
                    "family": family,
                    "train_fraction": fraction,
                    "classical_model": classical_model,
                    "classical_test_acc": c_test,
                    "quantum_model": quantum_model,
                    "quantum_test_acc": q_test,
                    "gap_classical_minus_quantum": gap,
                    "full_data_gap": full_gap,
                    "colab_signal": quantum_wins or within_two or low_fraction_gap_halved,
                    "signal_reason": ", ".join(
                        reason
                        for reason, active in [
                            ("quantum_wins", quantum_wins),
                            ("within_2_points", within_two),
                            ("low_fraction_gap_halved", low_fraction_gap_halved),
                        ]
                        if active
                    )
                    or "none",
                }
            )
    return out


def to_markdown(summary_rows: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> str:
    lines = [
        "# Low-Data Scaling Summary",
        "",
        "Auto-generated from `experiments/low_data/*.json`.",
        "Rows with `Runs = 1` are pilot evidence only; do not treat them as multi-seed claims.",
        "",
    ]
    if not summary_rows:
        lines.extend(["No low-data results found yet.", ""])
        return "\n".join(lines)

    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        families[row["family"]].append(row)

    for family in sorted(families):
        lines.append(f"## {family}")
        lines.append("")
        lines.append("| Model | Fraction | Runs | Seeds | Train Size | Best Val | Test | Params |")
        lines.append("|---|---:|---:|---|---:|---:|---:|---:|")
        for row in sorted(families[family], key=lambda item: (item["model"], item["train_fraction"])):
            lines.append(
                "| {model} | {fraction:.2f} | {runs} | {seeds} | {train_size} | {val} | {test} | {params} |".format(
                    model=row["model"],
                    fraction=row["train_fraction"],
                    runs=row["runs"],
                    seeds=",".join(str(seed) for seed in row["seeds"]),
                    train_size=row["train_size"],
                    val=fmt(row["best_val_acc_mean"], row["best_val_acc_std"]),
                    test=fmt(row["test_acc_mean"], row["test_acc_std"]),
                    params=row["total_params"],
                )
            )
        lines.append("")

    lines.append("## Colab Decision Signals")
    lines.append("")
    if not comparisons:
        lines.append("No paired comparison rows are complete yet.")
        lines.append("")
    else:
        lines.append("| Family | Fraction | Classical | Test | Quantum | Test | Gap C-Q | Signal | Reason |")
        lines.append("|---|---:|---|---:|---|---:|---:|---|---|")
        for row in comparisons:
            lines.append(
                "| {family} | {fraction:.2f} | {classical} | {c_test:.2f} | {quantum} | {q_test:.2f} | {gap:.2f} | {signal} | {reason} |".format(
                    family=row["family"],
                    fraction=row["train_fraction"],
                    classical=row["classical_model"],
                    c_test=row["classical_test_acc"],
                    quantum=row["quantum_model"],
                    q_test=row["quantum_test_acc"],
                    gap=row["gap_classical_minus_quantum"],
                    signal="yes" if row["colab_signal"] else "no",
                    reason=row["signal_reason"],
                )
            )
        lines.append("")
        confirmed_rows = [
            row
            for row in comparisons
            if row["colab_signal"]
            and all(
                summary.get("runs", 0) >= 3
                for summary in summary_rows
                if summary["model"] in {row["classical_model"], row["quantum_model"]}
                and summary["train_fraction"] == row["train_fraction"]
            )
        ]
        if confirmed_rows:
            lines.append("Decision: low-data confirmation is complete for the flagged multi-seed rows.")
        elif any(row["colab_signal"] for row in comparisons):
            lines.append("Decision: Colab follow-up is justified for the flagged model pair/fraction rows only.")
        else:
            lines.append("Decision: no Colab follow-up is justified by the current low-data rows.")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate low-data benchmark result JSON files")
    parser.add_argument("--input-glob", default="experiments/low_data/*.json")
    parser.add_argument("--json-out", default="experiments/low_data_summary.json")
    parser.add_argument("--md-out", default="docs/LOW_DATA_SUMMARY.md")
    args = parser.parse_args()

    records = load_records(args.input_glob)
    summary_rows = summarize(records)
    comparisons = comparison_rows(summary_rows)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary_rows, "comparisons": comparisons}, f, indent=2)
    with open(args.md_out, "w", encoding="utf-8") as f:
        f.write(to_markdown(summary_rows, comparisons))
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.md_out}")


if __name__ == "__main__":
    main()
