---
name: codex-audit
description: Audit official Codex capabilities against this repo's current integration so you can identify underused or stale Codex layers safely.
---

# Codex Audit

Use this skill when the task is about Codex itself: official capability research, repo integration review, hooks/rules coverage, profile strategy, or stale Codex guidance.

## Workflow

1. Read `AGENTS.md`, `.codex/config.toml`, `docs/CODEX_INTEGRATION_2026-03-22.md`, and `docs/CODEX_WORKFLOWS.md`.
2. Verify Codex capability claims against official OpenAI Codex docs before making recommendations.
3. Build a matrix with:
   - already used
   - underused
   - missing
   - not worth using in this repo
4. Pay special attention to:
   - nested `AGENTS.md`
   - rules
   - hooks
   - non-interactive `codex exec`
   - profile/model strategy
   - MCP use
5. If the task spans many files or requires a repo-process review, spawn `workflow_architect`.

## Repository Caveats

- `docs/CODEX_INTEGRATION_2026-03-22.md` is historical and may understate newer Codex capabilities.
- Do not confuse shell approval rules with general writing-policy enforcement; those live in `AGENTS.md`, hooks, and skills.
