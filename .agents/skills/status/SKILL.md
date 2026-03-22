---
name: status
description: Produce a current project status snapshot for this study, including code state, strongest results, stale docs, and next steps.
---

# Status

Use this skill when the user asks for the current state of the study or the repo.

## Workflow

1. Start from `AGENTS.md`.
2. Check repo state:
   - `git status`
   - recent commits if needed
   - available checkpoints and experiment artifacts
3. Reconcile current study results before summarizing them.
4. Report:
   - active code paths
   - strongest supported results
   - stale or conflicting docs
   - practical next steps
5. If the user wants a formal report, switch to `reconcile-results` first and then write the report in the requested language.
