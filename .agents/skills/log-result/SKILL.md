---
name: log-result
description: Log validated experimental results and update the study narrative without spreading stale claims.
---

# Log Result

Use this skill after metrics have been verified from actual outputs.

## Workflow

1. Read the latest entries in `docs/EXPERIMENTS.md`.
2. Verify the new numbers from the strongest available source:
   - experiment JSON
   - checkpoint evaluation output
   - notebook output
3. Add or update the result in `docs/EXPERIMENTS.md`.
4. If the result changes the study narrative, also check:
   - `paper/draft.md`
   - `README.md`
   - `CLAUDE.md`
5. Explicitly mark when a historical claim is being superseded by a newer result.
