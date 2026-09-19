# Statistical Evidence Summary

**Date:** August 9, 2026

This report is generated from repository benchmark artifacts. It is intended to support manuscript wording, not to create stronger claims than the artifacts justify.

## Method Notes

- 95% confidence intervals use the Student t distribution around the reported mean test accuracy.
- Full-data pairwise rows use two-sided Welch tests and standardized mean differences from summary statistics.
- Full-data groups have `n=3`; their Welch p-values are descriptive reviewer aids rather than definitive inferential evidence.
- Low-data current-local rows use the paired seed design (`n=6`, seeds 42--47); paired t and exact sign-flip tests are primary, with Holm correction across four fractions.
- Seed 42 used the local 3,427-image parser state, while Colab seeds 43--47 used 3,428 loaded images. Training subset sizes therefore differ by one image; the low-data analysis is near-matched and exploratory.
- Byte-original Drive JSON for seeds 43--47 was reconciled into the canonical low-data directory on August 9, 2026; the former notebook reconstructions remain documented in a superseded provenance manifest.
- Thesis-faithful low-data rows are seed-42 pilot evidence only, so no confidence interval or significance test is reported for that axis.

## Full-Data Test Accuracy Intervals

| Family | Model | Runs | Seeds | Test | 95% CI |
|---|---|---:|---|---:|---:|
| trainable-quantum-case-study | `V7_trainable_quantum_clean_20260427` | 1 | - | 65.88 ± 0.00 | - |
| trainable-quantum-case-study | `V7_trainable_quantum_documented` | 1 | - | 65.02 ± 0.00 | - |
| trainable-quantum-case-study | `V7_trainable_quantum_rerun` | 1 | - | 72.53 ± 0.00 | - |
| current-local | `classical_conv` | 3 | 42,43,44 | 81.40 ± 1.06 | [78.77, 84.03] |
| current-local | `non_trainable_quantum` | 3 | 42,43,44 | 80.40 ± 0.69 | [78.69, 82.11] |
| current-local | `param_linear` | 3 | 42,43,44 | 81.12 ± 2.27 | [75.48, 86.76] |
| modern-classical | `resnet18_cifar_gray` | 3 | 42,43,44 | 88.13 ± 0.82 | [86.09, 90.17] |
| thesis-faithful | `thesis_cnn3` | 3 | 42,43,44 | 79.33 ± 1.26 | [76.20, 82.46] |
| thesis-faithful | `thesis_cnniiii` | 3 | 42,43,44 | 85.26 ± 0.97 | [82.85, 87.67] |
| thesis-faithful | `thesis_hqnn2` | 3 | 42,43,44 | 78.61 ± 0.69 | [76.90, 80.32] |

## Full-Data Pairwise Comparisons

Positive differences mean the left model has higher mean test accuracy.

| Comparison | Left | Right | Difference | 95% CI | Welch p | Cohen's d | Interpretation |
|---|---|---|---:|---:|---:|---:|---|
| thesis-faithful best classical vs best quantum | `thesis_cnniiii` | `thesis_hqnn2` | 6.65 | [4.66, 8.64] | 0.001 | 7.90 | left higher on mean test accuracy |
| thesis-faithful CNN-III vs HQNN-II | `thesis_cnn3` | `thesis_hqnn2` | 0.72 | [-1.87, 3.31] | 0.447 | 0.71 | left higher on mean test accuracy |
| current-local strongest classical vs non-trainable quantum | `classical_conv` | `non_trainable_quantum` | 1.00 | [-1.17, 3.17] | 0.254 | 1.12 | left higher on mean test accuracy |
| current-local matched linear replacement vs non-trainable quantum | `param_linear` | `non_trainable_quantum` | 0.72 | [-4.38, 5.82] | 0.644 | 0.43 | left higher on mean test accuracy |
| current-local convolution vs matched linear replacement | `classical_conv` | `param_linear` | 0.28 | [-4.48, 5.04] | 0.860 | 0.16 | left higher on mean test accuracy |
| modern classical upper bound vs thesis-faithful best | `resnet18_cifar_gray` | `thesis_cnniiii` | 2.87 | [0.81, 4.93] | 0.018 | 3.20 | left higher on mean test accuracy |
| modern classical upper bound vs current-local best | `resnet18_cifar_gray` | `classical_conv` | 6.73 | [4.53, 8.93] | 0.001 | 7.10 | left higher on mean test accuracy |

## Low-Data Current-Local Pairwise Comparisons

Positive differences mean `non_trainable_quantum` has higher mean test accuracy than `classical_conv`.

| Fraction | Seeds | Q-C Difference | Paired 95% CI | Paired p | Exact sign-flip p | Holm p | Interpretation |
|---:|---|---:|---:|---:|---:|---:|---|
| 0.10 | 42,43,44,45,46,47 | 1.79 | [-2.93, 6.50] | 0.375 | 0.375 | 1.000 | quantum mean higher; CI crosses zero |
| 0.25 | 42,43,44,45,46,47 | 1.29 | [-1.88, 4.45] | 0.344 | 0.438 | 1.000 | quantum mean higher; CI crosses zero |
| 0.50 | 42,43,44,45,46,47 | 0.65 | [-1.43, 2.72] | 0.461 | 0.469 | 1.000 | quantum mean higher; CI crosses zero |
| 1.00 | 42,43,44,45,46,47 | 0.28 | [-0.44, 1.00] | 0.359 | 0.406 | 1.000 | quantum mean higher; CI crosses zero |

## Manuscript-Safe Interpretation

- Full-data RQ1 remains classical-favored: the largest and most stable leads belong to `resnet18_cifar_gray` and `thesis_cnniiii`.
- Current-local full-data differences among `classical_conv`, `param_linear`, and `non_trainable_quantum` are small relative to the low `n=3` uncertainty.
- The May 2026 low-data result supports only a narrow exploratory current-local signal: the largest mean gap is at 10%, but every paired 95% interval crosses zero.
- No row in this report supports a generic quantum-advantage claim.
