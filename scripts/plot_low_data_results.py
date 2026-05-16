#!/usr/bin/env python3
"""Create paper-ready low-data scaling figures from the aggregate summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_LABELS = {
    "classical_conv": "Classical conv",
    "non_trainable_quantum": "Non-trainable quantum",
    "thesis_cnniiii": "Thesis CNN-IIII",
    "thesis_hqnn2": "Thesis HQNN-II",
}

FAMILY_PAIRS = {
    "current-local": ("classical_conv", "non_trainable_quantum"),
    "thesis-faithful": ("thesis_cnniiii", "thesis_hqnn2"),
}

COLORS = {
    "classical_conv": "#2f5d8c",
    "non_trainable_quantum": "#c4512d",
    "thesis_cnniiii": "#5c6f2f",
    "thesis_hqnn2": "#7a4fa3",
}


def load_summary(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def index_summary(rows: list[dict[str, Any]]) -> dict[tuple[str, float], dict[str, Any]]:
    return {(row["model"], float(row["train_fraction"])): row for row in rows}


def family_rows(rows: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("family") == family]


def plot_accuracy_panel(ax: plt.Axes, rows: list[dict[str, Any]], family: str) -> None:
    indexed = index_summary(rows)
    for model in FAMILY_PAIRS[family]:
        model_rows = sorted(
            [row for row in rows if row["model"] == model],
            key=lambda row: row["train_fraction"],
        )
        x_vals = [row["train_fraction"] * 100 for row in model_rows]
        y_vals = [row["test_acc_mean"] for row in model_rows]
        y_err = [row["test_acc_std"] if row.get("runs", 0) > 1 else 0.0 for row in model_rows]
        ax.errorbar(
            x_vals,
            y_vals,
            yerr=y_err,
            marker="o",
            linewidth=2.2,
            capsize=4,
            color=COLORS[model],
            label=MODEL_LABELS[model],
        )

    title = "Current-local confirmation (3 seeds)" if family == "current-local" else "Thesis-faithful pilot (seed 42)"
    ax.set_title(title)
    ax.set_xlabel("Training fraction (%)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xticks([10, 25, 50, 100])
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="lower right")

    if family == "current-local":
        for fraction in [0.10, 0.25, 0.50, 1.00]:
            classical = indexed[("classical_conv", fraction)]["test_acc_mean"]
            quantum = indexed[("non_trainable_quantum", fraction)]["test_acc_mean"]
            ax.annotate(
                f"+{quantum - classical:.2f}",
                xy=(fraction * 100, quantum),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=COLORS["non_trainable_quantum"],
            )


def plot_gap_panel(ax: plt.Axes, comparisons: list[dict[str, Any]]) -> None:
    for family, color, marker in [
        ("current-local", "#c4512d", "o"),
        ("thesis-faithful", "#5c6f2f", "s"),
    ]:
        rows = sorted(
            [row for row in comparisons if row.get("family") == family],
            key=lambda row: row["train_fraction"],
        )
        x_vals = [row["train_fraction"] * 100 for row in rows]
        y_vals = [row["gap_classical_minus_quantum"] for row in rows]
        label = "Current-local: classical - quantum" if family == "current-local" else "Thesis-faithful: classical - quantum"
        ax.plot(x_vals, y_vals, marker=marker, linewidth=2.2, color=color, label=label)

    ax.axhline(0, color="#333333", linewidth=1.0)
    ax.set_title("Paired test-accuracy gap")
    ax.set_xlabel("Training fraction (%)")
    ax.set_ylabel("Gap in test accuracy points")
    ax.set_xticks([10, 25, 50, 100])
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper right")


def create_figure(summary: dict[str, Any], output: Path) -> None:
    rows = summary["summary"]
    comparisons = summary["comparisons"]

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    plot_accuracy_panel(axes[0], family_rows(rows, "current-local"), "current-local")
    plot_accuracy_panel(axes[1], family_rows(rows, "thesis-faithful"), "thesis-faithful")
    plot_gap_panel(axes[2], comparisons)
    fig.suptitle("Low-data scaling: current-local confirmation and thesis-faithful pilot", fontsize=13)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot low-data scaling results")
    parser.add_argument("--summary", default="experiments/low_data_summary.json", type=Path)
    parser.add_argument("--out", default="paper/figures/low_data_scaling.png", type=Path)
    parser.add_argument(
        "--pdf-out",
        default="paper/figures/low_data_scaling.pdf",
        type=Path,
        help="Optional PDF copy for paper workflows. Use an empty string to skip.",
    )
    args = parser.parse_args()

    summary = load_summary(args.summary)
    create_figure(summary, args.out)
    print(f"Wrote {args.out}")
    if str(args.pdf_out):
        create_figure(summary, args.pdf_out)
        print(f"Wrote {args.pdf_out}")


if __name__ == "__main__":
    main()
