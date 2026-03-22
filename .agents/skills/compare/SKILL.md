---
name: compare
description: Compare model versions, circuit variants, or training configurations in this repo. Produces evidence-backed tables and explains speed, stability, and accuracy tradeoffs.
---

# Compare

Use this skill when the user asks which version, circuit, or setup is better.

## Workflow

1. Start from `AGENTS.md`.
2. Collect evidence from:
   - `docs/EXPERIMENTS.md`
   - `experiments/*.json`
   - `paper/draft.md`
   - relevant source files
3. If numbers conflict, prefer local artifacts over narrative docs.
4. Produce a compact comparison table covering:
   - feature map size
   - circuit type
   - trainable vs fixed quantum layer
   - runtime or quantum-call cost
   - validation/test performance
   - stability notes
5. End with a clear recommendation and state whether it is a current conclusion or a historical one.
