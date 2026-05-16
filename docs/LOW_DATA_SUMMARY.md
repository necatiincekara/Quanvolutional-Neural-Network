# Low-Data Scaling Summary

Auto-generated in Colab from Drive-backed `experiments/low_data/*.json`; synced locally from `quanv_results/low_data_confirm_20260502`.
Drive raw-artifact IDs for the confirmation rows are recorded in `experiments/low_data_drive_manifest_20260502.json`.
Paper figure artifacts are available at `paper/figures/low_data_scaling.png` and `paper/figures/low_data_scaling.pdf`.
Rows with `Runs = 1` are pilot evidence only; do not treat them as multi-seed claims.

## current-local

| Model | Fraction | Runs | Seeds | Train Size | Best Val | Test | Params |
|---|---:|---:|---|---:|---:|---:|---:|
| classical_conv | 0.10 | 3 | 42,43,44 | 308 | 50.88 ± 2.28 | 48.42 ± 2.31 | 88045 |
| classical_conv | 0.25 | 3 | 42,43,44 | 771 | 70.17 ± 1.52 | 66.24 ± 1.78 | 88045 |
| classical_conv | 0.50 | 3 | 42,43,44 | 1542 | 79.53 ± 1.34 | 75.61 ± 1.02 | 88045 |
| classical_conv | 1.00 | 3 | 42,43,44 | 3085 | 87.04 ± 1.10 | 80.47 ± 0.57 | 88045 |
| non_trainable_quantum | 0.10 | 3 | 42,43,44 | 308 | 57.02 ± 0.77 | 50.71 ± 2.93 | 88488 |
| non_trainable_quantum | 0.25 | 3 | 42,43,44 | 771 | 71.74 ± 1.50 | 69.88 ± 0.99 | 88488 |
| non_trainable_quantum | 0.50 | 3 | 42,43,44 | 1542 | 81.87 ± 4.02 | 76.75 ± 0.50 | 88488 |
| non_trainable_quantum | 1.00 | 3 | 42,43,44 | 3085 | 86.36 ± 1.47 | 80.76 ± 0.99 | 88488 |

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
| current-local | 0.10 | classical_conv | 48.42 | non_trainable_quantum | 50.71 | -2.29 | yes | quantum_wins, within_2_points |
| current-local | 0.25 | classical_conv | 66.24 | non_trainable_quantum | 69.88 | -3.64 | yes | quantum_wins, within_2_points |
| current-local | 0.50 | classical_conv | 75.61 | non_trainable_quantum | 76.75 | -1.14 | yes | quantum_wins, within_2_points |
| current-local | 1.00 | classical_conv | 80.47 | non_trainable_quantum | 80.76 | -0.29 | yes | quantum_wins, within_2_points |
| thesis-faithful | 0.10 | thesis_cnniiii | 65.88 | thesis_hqnn2 | 50.43 | 15.45 | no | none |
| thesis-faithful | 0.25 | thesis_cnniiii | 79.61 | thesis_hqnn2 | 62.45 | 17.16 | no | none |
| thesis-faithful | 0.50 | thesis_cnniiii | 82.40 | thesis_hqnn2 | 72.10 | 10.30 | no | none |
| thesis-faithful | 1.00 | thesis_cnniiii | 85.19 | thesis_hqnn2 | 78.33 | 6.86 | no | none |

Decision: low-data confirmation is complete for the flagged current-local multi-seed rows.
