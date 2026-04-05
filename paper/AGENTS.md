# paper/AGENTS.md

Scoped instructions for Codex when working under `paper/`.

## Purpose

This directory is for manuscript writing, claim synchronization, and submission-facing paper edits.

## Required Behavior

- Reconcile before writing. Use `paper-sync` or `reconcile-results` before changing scientific claims.
- Treat the thesis as the foundation of the paper, not as something to invalidate.
- Keep benchmark families separate:
  - thesis-faithful reproductions
  - current-local matched-budget ablations
  - trainable-quantum engineering case-study
- Do not describe V7 as the best current model.
- Do not write generic "quantum advantage" language unless newly verified artifacts explicitly support it.

## Preferred Workflow

1. Reconcile artifacts.
2. Audit stale or unsupported claims.
3. Write the smallest defensible paper change.
4. If results or framing change substantially, refresh the corresponding `.docx` output.

## Useful Skills And Agents

- Skills: `paper-sync`, `reconcile-results`, `compare`
- Agents: `paper_consistency_reviewer`, `paper_writer`, `result_reconciler`
