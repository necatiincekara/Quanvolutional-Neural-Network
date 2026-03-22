---
name: performance-debug
description: Diagnose training slowness, dead quantum signal, scheduler mistakes, and hybrid learning collapse in this repo.
---

# Performance Debug

Use this skill when the problem is broader than gradients alone.

## Workflow

1. Inspect the active training path and the model it uses.
2. Separate the failure mode:
   - slow first epoch only
   - persistent slowness
   - loss not decreasing
   - unstable or collapsed accuracy
3. Check the known repo-specific causes:
   - first-epoch JIT or kernel compilation cost
   - too many quantum patches entering the circuit
   - dead quantum signal via very small `q_out.std()`
   - tiny or `NaN` gradients
   - scheduler stepped in the wrong place
   - dtype or AMP boundary issues around the quantum layer
4. Prefer evidence from the current code path, not from historical architecture notes.
5. Return:
   - observed bottleneck or failure mode
   - strongest likely cause
   - minimal fix
   - whether a rerun or ablation is needed

Use `gradient-check` when the issue is clearly gradient-specific. Use this skill when the failure could also be architectural, scheduler-related, or runtime-related.
