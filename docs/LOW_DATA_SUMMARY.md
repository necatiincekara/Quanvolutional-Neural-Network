# Low-Data Scaling Summary

Auto-generated from `experiments/low_data/*.json`.
Rows with `Runs = 1` are pilot evidence only; do not treat them as multi-seed claims.

## current-local

| Model | Fraction | Runs | Seeds | Train Size | Best Val | Test | Params |
|---|---:|---:|---|---:|---:|---:|---:|
| classical_conv | 0.10 | 1 | 42 | 308 | 49.71 ± 0.00 | 47.42 ± 0.00 | 88045 |
| classical_conv | 0.25 | 1 | 42 | 771 | 68.42 ± 0.00 | 68.24 ± 0.00 | 88045 |
| classical_conv | 0.50 | 1 | 42 | 1542 | 78.36 ± 0.00 | 76.39 ± 0.00 | 88045 |
| classical_conv | 1.00 | 1 | 42 | 3085 | 86.26 ± 0.00 | 80.69 ± 0.00 | 88045 |
| non_trainable_quantum | 0.10 | 1 | 42 | 308 | 56.73 ± 0.00 | 50.21 ± 0.00 | 88488 |
| non_trainable_quantum | 0.25 | 1 | 42 | 771 | 71.35 ± 0.00 | 69.31 ± 0.00 | 88488 |
| non_trainable_quantum | 0.50 | 1 | 42 | 1542 | 78.36 ± 0.00 | 77.04 ± 0.00 | 88488 |
| non_trainable_quantum | 1.00 | 1 | 42 | 3085 | 86.55 ± 0.00 | 81.33 ± 0.00 | 88488 |

## thesis-faithful

| Model | Fraction | Runs | Seeds | Train Size | Best Val | Test | Params |
|---|---:|---:|---|---:|---:|---:|---:|
| thesis_cnniiii | 0.10 | 1 | 42 | 308 | 70.18 ± 0.00 | 65.88 ± 0.00 | 1378124 |
| thesis_cnniiii | 0.25 | 1 | 42 | 771 | 85.67 ± 0.00 | 79.61 ± 0.00 | 1378124 |
| thesis_cnniiii | 0.50 | 1 | 42 | 1542 | 88.30 ± 0.00 | 82.40 ± 0.00 | 1378124 |
| thesis_cnniiii | 1.00 | 1 | 42 | 3085 | 92.40 ± 0.00 | 85.19 ± 0.00 | 1378124 |
| thesis_hqnn2 | 0.10 | 1 | 42 | 308 | 55.56 ± 0.00 | 50.43 ± 0.00 | 248428 |
| thesis_hqnn2 | 0.25 | 1 | 42 | 771 | 66.67 ± 0.00 | 62.45 ± 0.00 | 248428 |
| thesis_hqnn2 | 0.50 | 1 | 42 | 1542 | 79.24 ± 0.00 | 72.10 ± 0.00 | 248428 |
| thesis_hqnn2 | 1.00 | 1 | 42 | 3085 | 84.21 ± 0.00 | 78.33 ± 0.00 | 248428 |

## Colab Decision Signals

| Family | Fraction | Classical | Test | Quantum | Test | Gap C-Q | Signal | Reason |
|---|---:|---|---:|---|---:|---:|---|---|
| current-local | 0.10 | classical_conv | 47.42 | non_trainable_quantum | 50.21 | -2.79 | yes | quantum_wins, within_2_points |
| current-local | 0.25 | classical_conv | 68.24 | non_trainable_quantum | 69.31 | -1.07 | yes | quantum_wins, within_2_points |
| current-local | 0.50 | classical_conv | 76.39 | non_trainable_quantum | 77.04 | -0.65 | yes | quantum_wins, within_2_points |
| current-local | 1.00 | classical_conv | 80.69 | non_trainable_quantum | 81.33 | -0.64 | yes | quantum_wins, within_2_points |
| thesis-faithful | 0.10 | thesis_cnniiii | 65.88 | thesis_hqnn2 | 50.43 | 15.45 | no | none |
| thesis-faithful | 0.25 | thesis_cnniiii | 79.61 | thesis_hqnn2 | 62.45 | 17.16 | no | none |
| thesis-faithful | 0.50 | thesis_cnniiii | 82.40 | thesis_hqnn2 | 72.10 | 10.30 | no | none |
| thesis-faithful | 1.00 | thesis_cnniiii | 85.19 | thesis_hqnn2 | 78.33 | 6.86 | no | none |

Decision: Colab follow-up is justified for the flagged model pair/fraction rows only.
