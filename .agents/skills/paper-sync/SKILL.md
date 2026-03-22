---
name: paper-sync
description: Synchronize paper and thesis writing with the actual experimental evidence in this repo. Use before drafting, revising, or finishing the paper.
---

# Paper Sync

Use this skill before editing `paper/draft.md`, thesis text, or any publication-facing summary.

## Workflow

1. Read `AGENTS.md` and the current `paper/draft.md`.
2. Reconcile the paper against:
   - `experiments/*.json`
   - `docs/EXPERIMENTS.md`
   - notebook outputs that contain captured metrics
   - `README.md`, `CLAUDE.md`, and `docs/AUDIT_REPORT.md` for stale claims
3. Build a short inconsistency list:
   - current-best model claim
   - validation and test metrics
   - pending vs completed experiments
   - unsupported narrative claims
4. When writing results, prefer this academic structure:
   - hypothesis or research question
   - configuration
   - quantitative results
   - conclusion
5. If the task is substantial, spawn `paper_consistency_reviewer` first.
6. If the task involves drafting or revising the paper itself, use `paper_writer`.

Repository note: this repo contains historical planning text that is no longer fully aligned with current experimental artifacts. Do not write the paper from `README.md` or `CLAUDE.md` alone.
