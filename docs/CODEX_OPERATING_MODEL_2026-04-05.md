# Codex Operating Model

**Date:** April 5, 2026

This document defines how Codex should be used in this repository after the April 2026 upgrade sprint.

## 1. Core Principle

The goal is not to use every Codex feature. The goal is to use a **small, high-leverage, repo-specific Codex stack** that improves research throughput and publication safety.

## 2. Instruction Stack

Instruction precedence in this repo:

1. root `AGENTS.md`
2. nearest nested `AGENTS.md`
3. repository skills
4. repository subagents
5. hooks as reminder/context layer
6. rules as shell approval guardrails

Scoped instruction files:

- `paper/AGENTS.md`
- `docs/AGENTS.md`
- `src/AGENTS.md`

## 3. Profile Policy

| Profile | Default Use |
|---|---|
| `paper` | manuscript editing, claim-sensitive drafting |
| `review` | read-only audits, circuit review, paper consistency review |
| `deep` | high-rigor non-interactive reconciliation, paper audit, workflow audit, benchmark triage, and roadmap planning |
| `fast_local` | lighter local iteration |
| `benchmark` | result reconciliation, next-run decision, roadmap work |
| `colab` | Colab handoff and remote-run planning |

As of April 27, 2026, high-rigor non-interactive scripts use the repo-local `deep`
profile and attach JSON output schemas where the result should be machine-checkable.
They still pin read-only sandboxing at the script level when the workflow is an audit.

## 4. Skill Routing

Use these skills by default for the following jobs:

- current repo state: `status`
- metric/truth reconciliation: `reconcile-results`
- Codex capability or integration review: `codex-audit`
- next-run choice: `benchmark-triage`
- next-phase planning: `roadmap-sync`
- paper synchronization: `paper-sync`
- artifact/share packet refresh: `artifact-pack`
- training orchestration: `train`
- Colab prep: `sync-colab` + `setup-env`

## 5. Subagent Routing

Use subagents only when the task is large enough to justify them.

- `result_reconciler`: factual status across artifacts
- `paper_consistency_reviewer`: stale or unsupported paper/docs claims
- `paper_writer`: targeted academic drafting
- `quantum_ml_reviewer`: circuit/training-path review
- `workflow_architect`: Codex integration and workflow coverage
- `benchmark_strategist`: benchmark order, platform, compute, paper impact

## 6. Hooks And Rules

### Hooks

Hooks provide:

- session-start context
- prompt-time reminders
- Bash pre/post reminders for training, benchmarking, and writing workflows

Hooks are reminder-level only. They should not replace reconciliation or scientific judgment.

### Rules

Rules are used for shell safety and approval friction only.

They protect:

- benchmark summary refresh commands
- share/artifact refresh commands
- expensive or platform-sensitive training commands
- destructive shell commands

Rules are not a substitute for paper claim enforcement. That belongs in:

- `AGENTS.md`
- skills
- hooks
- review workflows

## 7. Script Layer

The main non-interactive workflow entrypoints are:

- `scripts/codex-capability-audit.sh`
- `scripts/codex-gap-audit.sh`
- `scripts/codex-study-status.sh`
- `scripts/codex-paper-audit.sh`
- `scripts/codex-circuit-review.sh`
- `scripts/codex-benchmark-triage.sh`
- `scripts/codex-roadmap-plan.sh`
- `scripts/codex-colab-handoff.sh`
- `scripts/codex-artifact-pack.sh`
- `scripts/codex-model-benchmark.sh`

For the current local CLI version, these scripts are the preferred non-interactive
entrypoints because they combine the appropriate profile, sandbox, and output schema.

## 8. Default Operational Sequence

When the repo truth is uncertain:

1. `codex-study-status.sh`
2. `codex-gap-audit.sh` if the question is workflow-related
3. `codex-benchmark-triage.sh` if the question is "what next?"

When editing the paper:

1. `codex-paper-audit.sh`
2. `paper-sync`
3. `paper_writer` if drafting is substantial

When planning the next scientific phase:

1. `codex-benchmark-triage.sh`
2. `codex-roadmap-plan.sh`
3. `codex-colab-handoff.sh` if the next task is a Colab run

## 9. Anti-Patterns

Avoid these:

- writing paper claims directly from `README.md` or `CLAUDE.md`
- mixing thesis-faithful and current-local model families into one misleading leaderboard
- treating rules as if they were narrative-writing policies
- running heavy trainable-quantum work on the Mac by default
- adding more automation layers before the current ones are actually used
