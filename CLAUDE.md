# CLAUDE.md

This file provides project guidance for Claude Code when working in this repository.

## Project Reality

This repository contains a hybrid quantum-classical OCR study for Ottoman-Turkish handwritten character recognition. It includes both historical thesis-era material and newer post-thesis experimental artifacts. Do not assume the oldest planning documents still reflect the current scientific status.

As of April 27, 2026:

- `src/model.py` and `src/train.py` represent the older V4/V6-style path.
- `src/trainable_quantum_model.py`, `src/enhanced_training.py`, and `train_v7.py` represent the newer stabilized trainable-quantum path.
- Multi-seed classical artifacts currently outperform the trainable and non-trainable quantum variants on test accuracy.
- The current repo evidence supports a stronger "fair benchmark + hybrid QML engineering and failure-analysis" story than a clean "quantum advantage" story.

## Source Of Truth Order

When results disagree, use this order:

1. `experiments/*.json`, saved checkpoints under `models/`, and notebook outputs with captured metrics
2. `docs/EXPERIMENTS.md`
3. `paper/draft.md`
4. current code in `src/` and training entrypoints
5. historical planning docs such as `README.md`, `docs/AUDIT_REPORT.md`, `docs/IMPLEMENTATION_GUIDE.md`, and older roadmap text

## Current Known Results

- Modern-classical upper bound: `resnet18_cifar_gray` at `88.13 ± 0.82` test and `92.98 ± 0.29` best validation.
- Strongest thesis-faithful reproduction: `thesis_cnniiii` at `85.26 ± 0.97` test.
- Current-local matched-budget results:
  - `classical_conv`: `81.40 ± 1.06` test
  - `param_linear`: `81.12 ± 2.27` test
  - `non_trainable_quantum`: `80.40 ± 0.69` test
- Trainable-quantum case study: fresh V7 Colab rerun at `72.89%` best validation and `72.53%` test; older documented V7 remains `67.35%` best validation and `65.02%` test.
- Thesis-best quantum result was `HQNN-II` at `82.40%` in the thesis table, but the current thesis-faithful reproduction `thesis_hqnn2` is `78.61 ± 0.69` and should be kept separate from the Henderson-style current-local non-trainable quantum ablation.

## High-Value Files

- Core models:
  - `src/model.py`
  - `src/trainable_quantum_model.py`
- Training:
  - `src/train.py`
  - `src/enhanced_training.py`
  - `train_v7.py`
  - `train_ablation_local.py`
- Modern classical baselines:
  - `train_modern_baselines.py`
  - `src/modern_baselines.py`
- Publication benchmark training:
  - `train_thesis_models.py`
  - `src/thesis_models.py`
  - `src/benchmark_protocol.py`
  - `src/benchmark_training.py`
- Data:
  - `src/dataset.py`
  - `src/config.py`
- Results:
  - `docs/EXPERIMENTS.md`
  - `experiments/*.json`
  - `paper/draft.md`
  - `train_v7_colab.ipynb`
- Publication planning:
  - `docs/PUBLICATION_STRATEGY_2026-03-22.md`
  - `docs/RESEARCH_ROADMAP.md`
  - `docs/BENCHMARK_MATRIX_2026-03-22.md`
  - `docs/BENCHMARK_SUMMARY.md`

## Important Repo Caveats

- `README.md`, `docs/AUDIT_REPORT.md`, and parts of the historical roadmap are stale relative to post-2025 results.
- `experiments/run_experiments.py` is stale relative to the current enhanced training path.
- `train_v7.py` exposes `--target`, but the current enhanced training path does not fully thread that value through.
- The dataset loader skips malformed filenames with unknown label codes. One known file in `set/train` resolves to label code `00` and is ignored.

## Working Rules

- Reconcile artifacts before writing study summaries, README text, or paper claims.
- For scientific summaries, use exact dates and exact metric sources.
- Explicitly separate current claims from historical claims.
- Do not write publication-facing text from `CLAUDE.md` alone.
- Before paper work, use the publication strategy doc and current result artifacts.

## Common Commands

```bash
# Base training
python -m src.train

# Enhanced V7 training
python train_v7.py

# Local ablations
python train_ablation_local.py --help

# Thesis-faithful publication benchmarks
python train_thesis_models.py --help

# Aggregate benchmark tables
python scripts/aggregate_benchmarks.py

# Quick non-trainable quantum smoke test
python train_ablation_local.py --model non_trainable_quantum --test
```

## Claude Skill Guidance

Prefer the repo skills when the task matches:

- `status`: current project snapshot
- `reconcile-results`: determine what is factually true right now
- `compare`: compare baselines, versions, or circuits
- `paper-sync`: synchronize paper/thesis text with actual results
- `performance-debug`: analyze slowness, scheduler mistakes, dead quantum signal, or collapse
- `review-circuit`: inspect quantum circuit design
- `train`: orchestrate training
- `log-result`: update result logs after verification

## Publication Reality

This study is still publishable, but the current strongest angle is not "quantum beats classical." The stronger current angle is:

- rigorous comparison of trainable vs non-trainable quantum variants
- failure analysis and stabilization lessons for hybrid QML
- honest benchmarking on a 44-class Ottoman OCR task

For publication planning, use `docs/PUBLICATION_STRATEGY_2026-03-22.md` as the main reference.
