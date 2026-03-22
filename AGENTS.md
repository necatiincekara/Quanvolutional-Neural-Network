# AGENTS.md

Repository instructions for Codex working in this study.

## Project Reality

This repository tracks a hybrid quantum-classical OCR study for Ottoman-Turkish handwritten character recognition. It contains both historical planning material and newer experimental artifacts. Do not assume the oldest narrative docs are still correct.

As of March 22, 2026:

- `src/model.py` and `src/train.py` are the older V4/V6 baseline path.
- `src/trainable_quantum_model.py`, `src/enhanced_training.py`, and `train_v7.py` are the newer stabilized trainable-quantum path.
- Local ablation artifacts currently outperform the documented V7 result on test accuracy.
- Several human-written docs still describe V4 as the current best model. Treat that as historical unless re-verified.

## Source Of Truth Order

When results disagree, use this priority order:

1. `experiments/*.json`, saved checkpoints under `models/`, and notebook outputs with captured metrics
2. `docs/EXPERIMENTS.md`
3. `paper/draft.md`
4. Current code in `src/` and training entrypoints
5. Historical planning docs such as `README.md`, `CLAUDE.md`, `docs/AUDIT_REPORT.md`, and `docs/IMPLEMENTATION_GUIDE.md`

## Current Known Study Status

- Documented stabilized V7 result: about `67.35%` best validation and `65.02%` test accuracy in repo docs/notebook outputs.
- Stronger local ablation test results currently exist:
  - `classical_conv`: `82.62%` test
  - `param_linear`: `81.76%` test
  - `non_trainable_quantum`: `80.47%` test
- Therefore, the current repo evidence supports a stronger "hybrid QML engineering and stabilization" story than a clean "quantum advantage" story.

## Important Repo Caveats

- `README.md`, `CLAUDE.md`, and `docs/AUDIT_REPORT.md` are partially stale relative to post-November 2025 results.
- `experiments/run_experiments.py` is stale relative to the current `EnhancedTrainer` API.
- `train_v7.py` exposes `--target`, but the current enhanced training path does not thread that value through.
- The dataset loader skips malformed filenames with unknown label codes. One known file in `set/train` resolves to label code `00` and is ignored.
- Preserve `.claude/*` files. Codex integration in this repo is additive, not a replacement for Claude workflows.

## Preferred Working Style In This Repo

- Reconcile artifacts before writing status reports, paper text, or README updates.
- For architecture work, inspect both the base path and the trainable-quantum path before proposing changes.
- For scientific summaries, use exact dates and exact metric sources.
- For current-vs-historical comparisons, explicitly label which claims are current and which are legacy.
- For web research about Codex, OpenAI, Claude Code, or other live tooling, verify against official docs before making claims.

## High-Value Files

- Core models: `src/model.py`, `src/trainable_quantum_model.py`
- Training: `src/train.py`, `src/enhanced_training.py`, `train_v7.py`, `train_ablation_local.py`
- Data: `src/dataset.py`, `src/config.py`
- Results: `docs/EXPERIMENTS.md`, `experiments/*.json`, `paper/draft.md`, `train_v7_colab.ipynb`
- Historical guidance: `CLAUDE.md`, `docs/AUDIT_REPORT.md`, `docs/IMPLEMENTATION_GUIDE.md`
- Publication planning: `docs/PUBLICATION_STRATEGY_2026-03-22.md`, `docs/RESEARCH_ROADMAP.md`
- Codex integration: `.codex/config.toml`, `.codex/agents/*.toml`, `.agents/skills/*/SKILL.md`

## Repo-Specific Skills

Prefer the repository skills when the task matches:

- `status`: current study snapshot
- `reconcile-results`: resolve mismatches between docs, checkpoints, notebooks, and JSON logs
- `compare`: compare versions, circuits, or baselines
- `review-circuit`: quantum circuit review with the custom QML reviewer agent
- `performance-debug`: slowness, dead quantum signal, scheduler mistakes, or learning collapse analysis
- `paper-sync`: synchronize `paper/draft.md`, docs, and study claims before writing or editing the paper
- `experiment`: run or plan an experiment and document it
- `log-result`: update result logs after verification
- `train`: training orchestration
- `setup-env`: local or Colab environment setup
- `sync-colab`: local/Colab sync workflow
- `architecture`: design or analyze model changes
- `gradient-check`: debug vanishing or exploding gradients

## Repo-Specific Custom Agents

- `quantum_ml_reviewer`: read-only quantum ML reviewer for circuits and hybrid training logic
- `result_reconciler`: read-only artifact reconciler for "what is actually true right now?"
- `paper_consistency_reviewer`: read-only reviewer for stale or unsupported claims in paper/docs
- `paper_writer`: write-focused academic drafting agent for `paper/draft.md` and closely related study docs

## Practical Commands

- Base training: `python -m src.train`
- Enhanced V7 training: `python train_v7.py`
- Local ablations: `python train_ablation_local.py --help`
- Codex non-interactive study status: `./scripts/codex-study-status.sh`
- Codex paper audit: `./scripts/codex-paper-audit.sh`
- Codex circuit review: `./scripts/codex-circuit-review.sh src/trainable_quantum_model.py`
- Review local changes: use Codex `/review`
- Generate or refresh project instructions: use Codex `/init` only if you are intentionally regenerating this file

## Recommended Codex Profiles

- Default top-level config: general repository work
- `paper`: high-rigor paper drafting and claim checks
- `review`: read-only audit mode for repo review and circuit inspection
- `fast_local`: lighter local iteration when you do not need maximum reasoning depth

## Codex-Specific Notes

- Trust the repository so project-scoped `.codex/` settings load.
- If a session is not behaving as expected, check `/status` and `/debug-config`.
- If the user asks for a Turkish study report, prefer `reconcile-results` first and then write the report in Turkish.
