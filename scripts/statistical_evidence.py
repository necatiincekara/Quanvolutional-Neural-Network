#!/usr/bin/env python3
"""Generate uncertainty and exploratory pairwise statistics for benchmark artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    from scipy import stats
except Exception:  # pragma: no cover - scipy is expected in the project venv
    stats = None


FULL_DATA_COMPARISONS = [
    ("thesis_cnniiii", "thesis_hqnn2", "thesis-faithful best classical vs best quantum"),
    ("thesis_cnn3", "thesis_hqnn2", "thesis-faithful CNN-III vs HQNN-II"),
    ("classical_conv", "non_trainable_quantum", "current-local strongest classical vs non-trainable quantum"),
    ("param_linear", "non_trainable_quantum", "current-local matched linear replacement vs non-trainable quantum"),
    ("classical_conv", "param_linear", "current-local convolution vs matched linear replacement"),
    ("resnet18_cifar_gray", "thesis_cnniiii", "modern classical upper bound vs thesis-faithful best"),
    ("resnet18_cifar_gray", "classical_conv", "modern classical upper bound vs current-local best"),
]

LOW_DATA_COMPARISONS = [
    ("current-local", "non_trainable_quantum", "classical_conv"),
]


def read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_seed_index(experiments_dir: str) -> dict[str, list[int]]:
    seeds: dict[str, list[int]] = {}
    for path in sorted(Path(experiments_dir).glob("*.json")):
        data = read_json(str(path))
        if not isinstance(data, dict):
            continue
        model = data.get("model")
        train_seed = data.get("train_seed")
        if not model or train_seed is None:
            continue
        if data.get("benchmark_axis"):
            continue
        if data.get("protocol_version") != "publication_v1":
            continue
        seeds.setdefault(model, []).append(int(train_seed))
    return {model: sorted(set(values)) for model, values in seeds.items()}


def t_critical(df: float, confidence: float = 0.95) -> float:
    if df <= 0:
        return float("nan")
    if stats is None:
        return 1.96
    return float(stats.t.ppf((1.0 + confidence) / 2.0, df))


def confidence_interval(mean: float | None, std: float | None, n: int) -> dict[str, float | None]:
    if mean is None:
        return {"low": None, "high": None, "half_width": None}
    if std is None or n < 2:
        return {"low": None, "high": None, "half_width": None}
    half_width = t_critical(n - 1) * std / math.sqrt(n)
    return {
        "low": round(mean - half_width, 2),
        "high": round(mean + half_width, 2),
        "half_width": round(half_width, 2),
    }


def ci_text(ci: dict[str, float | None]) -> str:
    if ci["low"] is None or ci["high"] is None:
        return "-"
    return f"[{ci['low']:.2f}, {ci['high']:.2f}]"


def metric_text(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "-"
    if std is None:
        return f"{mean:.2f}"
    return f"{mean:.2f} ± {std:.2f}"


def row_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["model"]: row for row in rows}


def pooled_cohens_d(mean_a: float, std_a: float, n_a: int, mean_b: float, std_b: float, n_b: int) -> float | None:
    if n_a < 2 or n_b < 2:
        return None
    denom_df = n_a + n_b - 2
    if denom_df <= 0:
        return None
    pooled_var = ((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / denom_df
    if pooled_var <= 0:
        return None
    return (mean_a - mean_b) / math.sqrt(pooled_var)


def welch_comparison(left: dict[str, Any], right: dict[str, Any], label: str) -> dict[str, Any]:
    mean_l = left.get("test_acc_mean")
    mean_r = right.get("test_acc_mean")
    std_l = left.get("test_acc_std")
    std_r = right.get("test_acc_std")
    n_l = int(left.get("runs", 0))
    n_r = int(right.get("runs", 0))

    result = {
        "label": label,
        "left_model": left["model"],
        "right_model": right["model"],
        "left_test_acc_mean": mean_l,
        "left_test_acc_std": std_l,
        "left_runs": n_l,
        "right_test_acc_mean": mean_r,
        "right_test_acc_std": std_r,
        "right_runs": n_r,
        "difference_left_minus_right": None,
        "difference_ci_low": None,
        "difference_ci_high": None,
        "welch_t": None,
        "welch_df": None,
        "welch_p_two_sided": None,
        "cohens_d": None,
        "test_status": "not_tested",
        "caveat": "",
    }

    if mean_l is None or mean_r is None:
        result["caveat"] = "missing test accuracy"
        return result
    diff = mean_l - mean_r
    result["difference_left_minus_right"] = round(diff, 2)
    if n_l < 2 or n_r < 2 or std_l is None or std_r is None:
        result["caveat"] = "requires at least two runs per side"
        return result

    var_l = std_l**2 / n_l
    var_r = std_r**2 / n_r
    se = math.sqrt(var_l + var_r)
    if se == 0:
        result["test_status"] = "degenerate"
        result["caveat"] = "zero standard error"
        return result

    numerator = (var_l + var_r) ** 2
    denominator = (var_l**2 / (n_l - 1)) + (var_r**2 / (n_r - 1))
    df = numerator / denominator if denominator else float("nan")
    t_value = diff / se
    half_width = t_critical(df) * se
    p_value = None
    if stats is not None and not math.isnan(df):
        p_value = float(2.0 * stats.t.sf(abs(t_value), df))
    d_value = pooled_cohens_d(mean_l, std_l, n_l, mean_r, std_r, n_r)

    result.update(
        {
            "difference_ci_low": round(diff - half_width, 2),
            "difference_ci_high": round(diff + half_width, 2),
            "welch_t": round(t_value, 3),
            "welch_df": round(df, 2),
            "welch_p_two_sided": round(p_value, 4) if p_value is not None else None,
            "cohens_d": round(d_value, 2) if d_value is not None else None,
            "test_status": "exploratory",
            "caveat": "small-n descriptive Welch test; not a definitive inferential claim",
        }
    )
    return result


def full_data_intervals(summary_rows: list[dict[str, Any]], seed_index: dict[str, list[int]]) -> list[dict[str, Any]]:
    rows = []
    for row in summary_rows:
        mean_value = row.get("test_acc_mean")
        if mean_value is None:
            continue
        runs = int(row.get("runs", 0))
        std_value = row.get("test_acc_std")
        ci = confidence_interval(mean_value, std_value, runs)
        rows.append(
            {
                "family": row.get("family", "unknown"),
                "model": row["model"],
                "runs": runs,
                "seeds": seed_index.get(row["model"], []),
                "test_acc_mean": mean_value,
                "test_acc_std": std_value,
                "test_acc_ci95_low": ci["low"],
                "test_acc_ci95_high": ci["high"],
                "test_acc_ci95_half_width": ci["half_width"],
                "total_params": row.get("total_params"),
            }
        )
    return rows


def full_data_comparisons(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = row_lookup(summary_rows)
    comparisons = []
    for left_model, right_model, label in FULL_DATA_COMPARISONS:
        if left_model not in indexed or right_model not in indexed:
            continue
        comparisons.append(welch_comparison(indexed[left_model], indexed[right_model], label))
    return comparisons


def low_data_outputs(low_data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows = low_data.get("summary", [])
    intervals = []
    for row in summary_rows:
        mean_value = row.get("test_acc_mean")
        if mean_value is None:
            continue
        runs = int(row.get("runs", 0))
        std_value = row.get("test_acc_std")
        ci = confidence_interval(mean_value, std_value, runs)
        intervals.append(
            {
                "family": row.get("family", "unknown"),
                "model": row["model"],
                "train_fraction": row.get("train_fraction"),
                "runs": runs,
                "seeds": row.get("seeds", []),
                "test_acc_mean": mean_value,
                "test_acc_std": std_value,
                "test_acc_ci95_low": ci["low"],
                "test_acc_ci95_high": ci["high"],
                "test_acc_ci95_half_width": ci["half_width"],
            }
        )

    indexed = {
        (row.get("family"), row["model"], row.get("train_fraction")): row
        for row in summary_rows
    }
    comparisons = []
    for family, quantum_model, classical_model in LOW_DATA_COMPARISONS:
        fractions = sorted(
            {
                row.get("train_fraction")
                for row in summary_rows
                if row.get("family") == family and row.get("model") in {quantum_model, classical_model}
            }
        )
        for fraction in fractions:
            quantum = indexed.get((family, quantum_model, fraction))
            classical = indexed.get((family, classical_model, fraction))
            if not quantum or not classical:
                continue
            comparisons.append(
                welch_comparison(
                    quantum,
                    classical,
                    f"{family} low-data quantum vs classical at fraction {fraction:.2f}",
                )
            )
    return intervals, comparisons


def p_text(value: float | None) -> str:
    if value is None:
        return "-"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def comparison_ci_text(row: dict[str, Any]) -> str:
    low = row.get("difference_ci_low")
    high = row.get("difference_ci_high")
    if low is None or high is None:
        return "-"
    return f"[{low:.2f}, {high:.2f}]"


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Statistical Evidence Summary",
        "",
        f"**Date:** {report['generated_date']}",
        "",
        "This report is generated from repository benchmark artifacts. It is intended to support manuscript wording, not to create stronger claims than the artifacts justify.",
        "",
        "## Method Notes",
        "",
        "- 95% confidence intervals use the Student t distribution around the reported mean test accuracy.",
        "- Pairwise rows use two-sided Welch tests and standardized mean differences from summary statistics.",
        "- Because most multi-seed groups have only `n=3`, p-values are descriptive reviewer aids rather than definitive inferential evidence.",
        "- Low-data current-local tests are approximate because seed-43 and seed-44 raw JSON files are Drive-backed remote artifacts; the local repository currently stores the confirmed aggregate summary and Drive manifest.",
        "- Thesis-faithful low-data rows are seed-42 pilot evidence only, so no confidence interval or significance test is reported for that axis.",
        "",
        "## Full-Data Test Accuracy Intervals",
        "",
        "| Family | Model | Runs | Seeds | Test | 95% CI |",
        "|---|---|---:|---|---:|---:|",
    ]
    for row in report["full_data_intervals"]:
        if row["runs"] < 3 and row["family"] != "trainable-quantum-case-study":
            continue
        seeds = ",".join(str(seed) for seed in row.get("seeds", [])) or "-"
        ci = ci_text(
            {
                "low": row["test_acc_ci95_low"],
                "high": row["test_acc_ci95_high"],
                "half_width": row["test_acc_ci95_half_width"],
            }
        )
        lines.append(
            "| {family} | `{model}` | {runs} | {seeds} | {test} | {ci} |".format(
                family=row["family"],
                model=row["model"],
                runs=row["runs"],
                seeds=seeds,
                test=metric_text(row["test_acc_mean"], row["test_acc_std"]),
                ci=ci,
            )
        )

    lines.extend(
        [
            "",
            "## Full-Data Pairwise Comparisons",
            "",
            "Positive differences mean the left model has higher mean test accuracy.",
            "",
            "| Comparison | Left | Right | Difference | 95% CI | Welch p | Cohen's d | Interpretation |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["full_data_comparisons"]:
        diff = row.get("difference_left_minus_right")
        interpretation = "exploratory only"
        if diff is not None:
            if diff > 0:
                interpretation = "left higher on mean test accuracy"
            elif diff < 0:
                interpretation = "right higher on mean test accuracy"
            else:
                interpretation = "same mean test accuracy"
        lines.append(
            "| {label} | `{left}` | `{right}` | {diff} | {ci} | {p} | {d} | {interp} |".format(
                label=row["label"],
                left=row["left_model"],
                right=row["right_model"],
                diff=f"{diff:.2f}" if diff is not None else "-",
                ci=comparison_ci_text(row),
                p=p_text(row.get("welch_p_two_sided")),
                d=f"{row['cohens_d']:.2f}" if row.get("cohens_d") is not None else "-",
                interp=interpretation,
            )
        )

    lines.extend(
        [
            "",
            "## Low-Data Current-Local Pairwise Comparisons",
            "",
            "Positive differences mean `non_trainable_quantum` has higher mean test accuracy than `classical_conv`.",
            "",
            "| Fraction | Quantum Test | Classical Test | Difference | 95% CI | Welch p | Interpretation |",
            "|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["low_data_comparisons"]:
        diff = row.get("difference_left_minus_right")
        lines.append(
            "| {fraction:.2f} | {qtest} | {ctest} | {diff} | {ci} | {p} | {interp} |".format(
                fraction=float(row["label"].split()[-1]),
                qtest=metric_text(row["left_test_acc_mean"], row["left_test_acc_std"]),
                ctest=metric_text(row["right_test_acc_mean"], row["right_test_acc_std"]),
                diff=f"{diff:.2f}" if diff is not None else "-",
                ci=comparison_ci_text(row),
                p=p_text(row.get("welch_p_two_sided")),
                interp="quantum higher on mean test accuracy" if diff and diff > 0 else "no quantum mean lead",
            )
        )

    lines.extend(
        [
            "",
            "## Manuscript-Safe Interpretation",
            "",
            "- Full-data RQ1 remains classical-favored: the largest and most stable leads belong to `resnet18_cifar_gray` and `thesis_cnniiii`.",
            "- Current-local full-data differences among `classical_conv`, `param_linear`, and `non_trainable_quantum` are small relative to the low `n=3` uncertainty.",
            "- The May 2026 low-data result supports a narrow current-local signal for `non_trainable_quantum`, strongest at the 25% fraction and weakest at full data.",
            "- No row in this report supports a generic quantum-advantage claim.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate statistical evidence report from benchmark summaries")
    parser.add_argument("--benchmark-summary", default="experiments/benchmark_summary.json")
    parser.add_argument("--low-data-summary", default="experiments/low_data_summary.json")
    parser.add_argument("--experiments-dir", default="experiments")
    parser.add_argument("--json-out", default="experiments/statistical_evidence_2026-05-17.json")
    parser.add_argument("--md-out", default="docs/STATISTICAL_EVIDENCE_2026-05-17.md")
    parser.add_argument("--generated-date", default="May 17, 2026")
    args = parser.parse_args()

    benchmark_rows = read_json(args.benchmark_summary)
    low_data = read_json(args.low_data_summary)
    seed_index = load_seed_index(args.experiments_dir)
    low_intervals, low_comparisons = low_data_outputs(low_data)
    report = {
        "generated_date": args.generated_date,
        "methods": {
            "ci": "Student t 95% confidence interval for means",
            "pairwise": "two-sided Welch tests from summary statistics",
            "caveat": "small-n descriptive analysis; most multi-seed rows have n=3",
        },
        "full_data_intervals": full_data_intervals(benchmark_rows, seed_index),
        "full_data_comparisons": full_data_comparisons(benchmark_rows),
        "low_data_intervals": low_intervals,
        "low_data_comparisons": low_comparisons,
    }

    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(args.md_out, "w", encoding="utf-8") as f:
        f.write(to_markdown(report))
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.md_out}")


if __name__ == "__main__":
    main()
