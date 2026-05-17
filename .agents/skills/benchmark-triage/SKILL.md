---
name: benchmark-triage
description: Decide the next benchmark or training action using current artifacts, compute budget, platform constraints, and paper impact.
---

# Benchmark Triage

Use this skill when the user asks "what should we run next?", "should this run be on Mac or Colab?", or "what is the highest-value next benchmark?"

## Workflow

1. Read `AGENTS.md`.
2. Reconcile current metrics from:
   - `experiments/*.json`
   - `docs/BENCHMARK_SUMMARY.md`
   - `docs/BENCHMARK_MATRIX_2026-03-22.md`
   - `docs/PUBLICATION_STRATEGY_2026-03-22.md`
3. Decide:
   - next task
   - platform
   - expected cost/time
   - paper impact
   - stop condition
4. Keep benchmark families separate and explicitly label which family is being optimized.
5. For next quantum training, V8, or Colab spend, use `literature-review` first and require a literature-backed paper-impact gate.
6. If the task is broad or roadmap-level, spawn `benchmark_strategist` after `result_reconciler`.

## Repository Caveats

- Do not default to more V7 reruns. The current V7 evidence supports an engineering case-study, while future quantum work needs a matched classical comparator, gradient-health check, and source-backed publication rationale.
- `runtime_seconds` for cached non-trainable quantum runs does not include the one-time cache/precompute stage.
