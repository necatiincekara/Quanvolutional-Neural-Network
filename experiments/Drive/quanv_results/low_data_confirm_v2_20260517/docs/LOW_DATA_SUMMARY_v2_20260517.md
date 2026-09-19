# Low-Data Scaling Summary

Auto-generated from `experiments/low_data/*.json`.
Rows with `Runs = 1` are pilot evidence only; do not treat them as multi-seed claims.

## current-local

| Model | Fraction | Runs | Seeds | Train Size | Best Val | Test | Params |
|---|---:|---:|---|---:|---:|---:|---:|
| classical_conv | 0.10 | 6 | 42,43,44,45,46,47 | 308 | 52.05 ± 2.23 | 49.14 ± 2.49 | 88045 |
| classical_conv | 0.25 | 6 | 42,43,44,45,46,47 | 771 | 71.25 ± 1.75 | 67.88 ± 2.13 | 88045 |
| classical_conv | 0.50 | 6 | 42,43,44,45,46,47 | 1542 | 80.50 ± 1.66 | 75.36 ± 1.43 | 88045 |
| classical_conv | 1.00 | 6 | 42,43,44,45,46,47 | 3085 | 87.23 ± 1.12 | 80.62 ± 0.44 | 88045 |
| non_trainable_quantum | 0.10 | 6 | 42,43,44,45,46,47 | 308 | 56.38 ± 3.00 | 50.93 ± 2.72 | 88488 |
| non_trainable_quantum | 0.25 | 6 | 42,43,44,45,46,47 | 771 | 72.03 ± 1.07 | 69.17 ± 1.12 | 88488 |
| non_trainable_quantum | 0.50 | 6 | 42,43,44,45,46,47 | 1542 | 82.41 ± 2.85 | 76.00 ± 1.36 | 88488 |
| non_trainable_quantum | 1.00 | 6 | 42,43,44,45,46,47 | 3085 | 86.50 ± 1.15 | 80.90 ± 0.95 | 88488 |

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
| current-local | 0.10 | classical_conv | 49.14 | non_trainable_quantum | 50.93 | -1.79 | yes | quantum_wins, within_2_points |
| current-local | 0.25 | classical_conv | 67.88 | non_trainable_quantum | 69.17 | -1.29 | yes | quantum_wins, within_2_points |
| current-local | 0.50 | classical_conv | 75.36 | non_trainable_quantum | 76.00 | -0.64 | yes | quantum_wins, within_2_points |
| current-local | 1.00 | classical_conv | 80.62 | non_trainable_quantum | 80.90 | -0.28 | yes | quantum_wins, within_2_points |
| thesis-faithful | 0.10 | thesis_cnniiii | 65.88 | thesis_hqnn2 | 50.43 | 15.45 | no | none |
| thesis-faithful | 0.25 | thesis_cnniiii | 79.61 | thesis_hqnn2 | 62.45 | 17.16 | no | none |
| thesis-faithful | 0.50 | thesis_cnniiii | 82.40 | thesis_hqnn2 | 72.10 | 10.30 | no | none |
| thesis-faithful | 1.00 | thesis_cnniiii | 85.19 | thesis_hqnn2 | 78.33 | 6.86 | no | none |

Decision: low-data confirmation is complete for the flagged multi-seed rows.
