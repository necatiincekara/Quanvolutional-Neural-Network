---
name: reconcile-results
description: Reconcile contradictory metrics across docs, papers, notebooks, JSON logs, and checkpoints to determine the current factual status of the study.
---

# Reconcile Results

Use this skill before writing any serious study summary, README update, or paper claim.

## Workflow

1. Read `AGENTS.md`.
2. Gather evidence from:
   - `experiments/*.json`
   - `models/`
   - `docs/EXPERIMENTS.md`
   - `paper/draft.md`
   - `README.md`
   - `CLAUDE.md`
   - relevant notebook outputs
3. Build a conflict list:
   - current-best claim
   - validation/test metrics
   - pending-vs-completed experiments
   - reproducibility gaps
4. Prefer machine-generated artifacts over narrative text.
5. Produce:
   - current supported status
   - stale claims and their file locations
   - documentation sync actions
   - open questions that still require reruns

If deeper evidence gathering is needed, spawn the `result_reconciler` agent.
