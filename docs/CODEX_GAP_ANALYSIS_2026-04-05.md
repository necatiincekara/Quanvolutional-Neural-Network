# Codex Gap Analysis

**Date:** April 5, 2026

This document maps the repo's real workflows to Codex coverage and identifies the highest-leverage gaps that needed to be closed.

## 1. Workflow Coverage Matrix

| Workflow | Current Entrypoint | Skill Coverage | Agent Coverage | Script Coverage | Main Failure Mode | Missing Piece | Priority |
|---|---|---|---|---|---|---|---|
| Current study status reporting | `AGENTS.md`, `docs/BENCHMARK_SUMMARY.md` | `status`, `reconcile-results` | `result_reconciler` | `codex-study-status.sh` | stale docs can leak in | better context + hooks | High |
| Result reconciliation | `experiments/*.json`, `docs/EXPERIMENTS.md` | `reconcile-results` | `result_reconciler` | partial | manual prompting drift | stronger operating model | High |
| Paper sync | `paper/draft.md` | `paper-sync` | `paper_consistency_reviewer`, `paper_writer` | `codex-paper-audit.sh` | unsupported claims can return | nested instructions + hooks | High |
| Training orchestration | `train_v7.py`, `train_ablation_local.py`, `train_thesis_models.py` | `train`, `experiment`, `performance-debug` | `quantum_ml_reviewer` | none | next-task choice is ad hoc | benchmark triage | High |
| Colab handoff / sync | `train_v7.py`, notebook, Drive backup | `sync-colab`, `setup-env` | none | none | run prep not standardized | colab script + profile | High |
| Benchmark aggregation | `scripts/aggregate_benchmarks.py` | indirect only | none | native Python script only | aggregation exists but decision layer is missing | triage workflow | Medium |
| Roadmap planning | publication docs + manual reasoning | none before this sprint | none before this sprint | none | roadmap drift | roadmap skill + strategist | High |
| Stale-claim detection | paper/docs audits | `paper-sync`, `reconcile-results` | `paper_consistency_reviewer` | `codex-paper-audit.sh` | only paper-focused | broader docs workflow layer | Medium |
| Advisor/share/submission packet generation | docs + export script | none before this sprint | none before this sprint | `export_docx.sh` only | pack assembly is manual | artifact-pack skill/script | Medium |

## 2. Key Gaps That Mattered

### 2.1 Skill Discoverability and Coverage

The repo already had strong core skills, but some high-value workflows still depended on manual prompting:

- next-run decision making
- roadmap generation
- Codex capability review itself
- share/submission pack assembly

### 2.2 Review-Heavy Agent Bias

The existing subagent set was strong at reviewing and reconciling, but weak on:

- workflow architecture
- benchmark prioritization
- compute-aware sequencing

### 2.3 Missing Runtime Guardrails

Before this sprint, the repo lacked:

- hooks for reminder-level runtime guidance
- rules for shell approval guardrails
- scoped instructions for paper/docs/src surfaces

That made it too easy for a session to do the right high-level thing but still take the wrong local path.

### 2.4 Script Layer Was Too Narrow

The previous script layer covered:

- status
- paper audit
- circuit review

The missing recurring workflows were exactly the ones needed for the next study phase:

- capability audit
- gap audit
- benchmark triage
- roadmap replanning
- Colab handoff
- artifact-pack refresh

## 3. Prioritization

### High Leverage

- nested `AGENTS.md`
- benchmark-triage skill
- roadmap-sync skill
- workflow_architect agent
- benchmark_strategist agent
- repo hooks
- Colab handoff script

### Medium Leverage

- shell rules
- artifact-pack workflow
- model benchmark script
- expanded workflow docs

### Low Leverage Or Not Worth It Right Now

- CI/GitHub Automation in this sprint
- making MCP central to the study workflow
- adding many more narrow subagents
- adding many more skills than the workflow truly needs

## 4. Stale Guidance To Retire Or Downgrade

- `docs/CODEX_INTEGRATION_2026-03-22.md` should no longer be treated as the only Codex reference.
- `docs/CODEX_WORKFLOWS.md` should no longer be read as the full active workflow surface.
- any repo text implying Codex hooks are absent should be treated as stale.
- any repo text implying rules are only hypothetical should be treated as stale.

## 5. Resulting Direction

The repo should use Codex as a layered system:

1. `AGENTS.md` + nested `AGENTS.md` for behavior
2. skills for task packages
3. subagents for large read-heavy work
4. hooks for light reminders
5. rules for shell guardrails
6. scripts for repeatable high-value workflows

That combination is enough to support the next publication and benchmark phase without turning the repo into an over-automated maze.
