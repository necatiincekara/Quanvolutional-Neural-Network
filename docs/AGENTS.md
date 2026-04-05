# docs/AGENTS.md

Scoped instructions for Codex when working under `docs/`.

## Purpose

This directory holds benchmark summaries, strategy notes, integration docs, advisor/share materials, and roadmap documents.

## Required Behavior

- Narrative docs must follow artifact truth, not historical planning text.
- Prefer updating current operational docs over older historical docs.
- Keep the benchmark hierarchy explicit:
  - `thesis_cnniiii` is the strongest thesis-faithful reproduced model
  - `classical_conv` is the strongest current-local matched-budget model
  - `V7` is an engineering case-study, not the benchmark leader
- If a document is shareable or advisor-facing, refresh the `.docx` export after editing.

## Preferred Workflow

1. Start from `AGENTS.md`.
2. Reconcile against `experiments/*.json`, `docs/BENCHMARK_SUMMARY.md`, and `docs/SUBMISSION_BENCHMARK_2026-03-25.md`.
3. Update only the docs needed for the current task.
4. Regenerate Word copies for touched shareable docs.

## Useful Skills And Agents

- Skills: `status`, `reconcile-results`, `artifact-pack`, `roadmap-sync`
- Agents: `result_reconciler`, `benchmark_strategist`, `workflow_architect`
