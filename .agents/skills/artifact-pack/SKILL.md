---
name: artifact-pack
description: Assemble advisor, submission, or sharing packets from current repo artifacts without introducing stale claims.
---

# Artifact Pack

Use this skill when the task is to bundle shareable outputs such as advisor updates, submission packets, benchmark snapshots, or Word exports.

## Workflow

1. Read `AGENTS.md`.
2. Reconcile the current benchmark truth before packaging anything.
3. Prefer these sources:
   - `docs/SUBMISSION_BENCHMARK_2026-03-25.md`
   - `docs/BENCHMARK_SUMMARY.md`
   - `paper/draft.md`
   - `experiments/*.json`
4. Regenerate `.docx` versions for any touched shareable Markdown docs.
5. If the pack includes paper-facing text, use `paper-sync` first.
6. If the task spans multiple documents or requires structural review, spawn `paper_consistency_reviewer`.

## Repository Caveats

- Share packets must preserve the thesis-to-paper continuity framing.
- Do not mix thesis-faithful and current-local families into a single misleading table.
