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
5. If the task is broad or roadmap-level, spawn `benchmark_strategist` after `result_reconciler`.

## Repository Caveats

- The next heavy scientifically meaningful task is usually the Colab V7 confirmatory rerun, not more redundant Mac reruns.
- `runtime_seconds` for cached non-trainable quantum runs does not include the one-time cache/precompute stage.
