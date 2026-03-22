# Benchmark Summary

Auto-generated from `experiments/*.json` plus documented reference rows.

## current-local

| Model | Source | Runs | Best Val | Test | Params | Notes |
|---|---|---:|---:|---:|---:|---|
| classical_conv | repo-local-ablation | 1 | 84.50 ± 0.00 | 82.62 ± 0.00 | 88045 | - |
| non_trainable_quantum | repo-local-ablation | 1 | 88.89 ± 0.00 | 80.47 ± 0.00 | 88488 | - |
| param_linear | repo-local-ablation | 1 | 88.30 ± 0.00 | 81.76 ± 0.00 | 87798 | - |

## historical-reference

| Model | Source | Runs | Best Val | Test | Params | Notes |
|---|---|---:|---:|---:|---:|---|
| V4_historical_reference | historical-docs | 1 | 8.75 ± 0.00 | - | - | Historical non-trainable baseline reference. |

## trainable-quantum-case-study

| Model | Source | Runs | Best Val | Test | Params | Notes |
|---|---|---:|---:|---:|---:|---|
| V7_trainable_quantum_documented | docs/notebook | 1 | 67.35 ± 0.00 | 65.02 ± 0.00 | 87798 | Documented stabilized V7 result from docs/notebook output. |
