# Codex Capability Audit

**Date:** April 5, 2026

This document audits current Codex capabilities against the repository's actual integration layer. It is the authoritative capability baseline for future Codex changes in this repo.

> Update, April 27, 2026:
> High-rigor Codex workflows now use `gpt-5.5` with `xhigh` reasoning through the `deep`, `paper`, `review`, and `benchmark` profiles. Recurring audit/triage scripts now use `--output-schema`, and capability/gap audits no longer force MCP off.

## 1. Summary

The repo already uses the core Codex surfaces well enough to support serious research work:

- root `AGENTS.md`
- repository skills
- repository subagents
- config profiles
- non-interactive `codex exec` scripts

However, several high-leverage capabilities were underused or absent before this sprint:

- nested `AGENTS.md`
- hooks
- rules
- workflow-specific audit/triage scripts
- explicit model/profile evaluation

The practical conclusion is not that the repo needs every Codex feature. It needs the **right small set of Codex features used consistently**.

## 2. Capability Matrix

| Capability | Officially Supported | Repo Status Before | Repo Status Now | Leverage |
|---|---|---|---|---|
| Root `AGENTS.md` | Yes | Used | Used | High |
| Nested `AGENTS.md` | Yes | Missing | Added for `paper/`, `docs/`, `src/` | High |
| Skills | Yes | Used | Expanded | High |
| Subagents | Yes | Used | Expanded | High |
| Profiles in `.codex/config.toml` | Yes | Partially used | Expanded to `benchmark` and `colab` | High |
| Slash commands | Yes | Lightly documented | Retained; operational guidance improved | Medium |
| Hooks | Yes | Not used | Repo-local hooks added | High |
| Rules | Yes | Not used | Repo-local shell rules added | Medium |
| MCP | Yes | Available, not repo-central | Kept optional | Low |
| Non-interactive `codex exec` | Yes | Used | Expanded | High |
| Model overrides per task | Yes | Available, underused | Structured benchmark script added | Medium |

## 3. What The Repo Already Used Well

- A clear root `AGENTS.md` with the study truth hierarchy.
- Repository skills for result reconciliation, paper sync, circuit review, training, and Colab setup.
- Read-heavy subagents for paper consistency and factual reconciliation.
- Reusable `codex exec` scripts for status, paper audit, and circuit review.
- Profile separation between paper, review, and fast local work.

## 4. What Was Underused

### 4.1 Nested Instructions

The repo previously relied on one global `AGENTS.md`. That was enough for general orientation, but not enough to narrow behavior for:

- paper-writing surfaces
- docs/reporting surfaces
- training/code surfaces

### 4.2 Hooks

Hooks were previously treated as effectively absent in this repo. That understated current Codex capability. The repo now uses hooks for:

- session-start context
- prompt-time reminders
- Bash pre/post reminders around training, benchmark, and writing workflows

### 4.3 Rules

Rules were previously postponed. They are now used as **shell approval guardrails**, which is their correct scope. They do not replace AGENTS/skills for scientific writing behavior.

### 4.4 Workflow-Specific Automation

The earlier script layer was useful but narrow. The missing workflows were:

- capability audit
- gap audit
- benchmark triage
- roadmap generation
- Colab handoff planning
- artifact-pack refresh

## 5. What Is Still Optional Or Low Priority

These capabilities are real, but not central to the current repo strategy:

- heavy MCP-driven workflow redesign
- CI/GitHub automation for Codex in this sprint
- maximizing slash-command surface usage for its own sake
- global user-home customization as a requirement for repo correctness

## 6. Chosen Model/Profile Strategy

Current defaults after this sprint:

- `paper`: `gpt-5.5` / `xhigh`
- `review`: `gpt-5.5` / `xhigh`
- `deep`: `gpt-5.5` / `xhigh`
- `benchmark`: `gpt-5.5` / `xhigh`
- `fast_local`: `gpt-5.4-mini`
- `colab`: `gpt-5.4-mini`

This keeps high-rigor research workflows on the strongest configured model while preserving lighter local and Colab handoff profiles. A dedicated benchmark script still exists to compare:

- `gpt-5.4`
- `gpt-5.3-codex`
- `gpt-5.4-mini`

If the benchmark shows that `gpt-5.3-codex` is equal or better for review/benchmark work, the repo can update those defaults in one controlled change instead of ad hoc per-session switching.

## 7. Repo-Local High-Leverage Priorities

1. Always route writing through reconciliation.
2. Keep benchmark families separate in every summary and script.
3. Use hooks for lightweight reminders, not heavy control.
4. Use rules for shell safety only.
5. Prefer non-interactive scripts for recurring high-value tasks.
6. Reserve Colab-centric orchestration for the V7 trainable-quantum path and later extension work.

## 8. Official Sources

OpenAI Codex docs used as the official capability baseline:

- `https://developers.openai.com/codex/overview`
- `https://developers.openai.com/codex/guides/agents-md`
- `https://developers.openai.com/codex/skills`
- `https://developers.openai.com/codex/subagents`
- `https://developers.openai.com/codex/cli/slash-commands`
- `https://developers.openai.com/codex/config-reference`
- `https://developers.openai.com/codex/rules`
- `https://developers.openai.com/codex/hooks`
