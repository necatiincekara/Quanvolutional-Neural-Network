# src/AGENTS.md

Scoped instructions for Codex when working under `src/`.

## Purpose

This directory contains model, training, dataset, and benchmark infrastructure code for the study.

## Required Behavior

- Inspect the active training path before proposing code changes:
  - `src/model.py` / `src/train.py` for the older path
  - `src/trainable_quantum_model.py` / `src/enhanced_training.py` for V7
  - `src/thesis_models.py` / `src/benchmark_training.py` for publication benchmark work
- Keep thesis-faithful HQNN models separate from the current Henderson-style local non-trainable quantum baseline.
- Prefer Colab/CUDA for heavy trainable-quantum work; prefer the M4 only for M4-feasible classical and cached non-trainable work.
- If code changes affect metrics, training procedure, or benchmark semantics, update the relevant docs after verification.

## Preferred Workflow

1. Reconcile the current benchmark truth.
2. Inspect the relevant training path.
3. Apply the smallest code change that resolves the issue.
4. Verify with a smoke test or the lightest meaningful run.

## Useful Skills And Agents

- Skills: `train`, `experiment`, `review-circuit`, `performance-debug`, `benchmark-triage`
- Agents: `quantum_ml_reviewer`, `benchmark_strategist`
