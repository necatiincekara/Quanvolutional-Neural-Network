# Benchmark Summary

Auto-generated from `experiments/*.json` plus documented reference rows.

## current-local

| Model | Source | Runs | Best Val | Test | Params | Notes |
|---|---|---:|---:|---:|---:|---|
| classical_conv | repo-local-ablation | 3 | 86.26 ± 1.76 | 81.40 ± 1.06 | 88045 | - |
| non_trainable_quantum | repo-local-ablation | 3 | 85.77 ± 0.94 | 80.40 ± 0.69 | 88488 | - |
| param_linear | repo-local-ablation | 3 | 86.45 ± 0.61 | 81.12 ± 2.27 | 87798 | - |

## historical-reference

| Model | Source | Runs | Best Val | Test | Params | Notes |
|---|---|---:|---:|---:|---:|---|
| V4_historical_reference | historical-docs | 1 | 8.75 ± 0.00 | - | - | Historical non-trainable baseline reference. |

## modern-classical

| Model | Source | Runs | Best Val | Test | Params | Notes |
|---|---|---:|---:|---:|---:|---|
| resnet18_cifar_gray | modern-baseline | 3 | 92.98 ± 0.29 | 88.13 ± 0.82 | 11190252 | Reviewer-proof stronger classical baseline: torchvision ResNet18 adapted to 32x32 grayscale inputs with a CIFAR-style stem. |

## thesis-faithful

| Model | Source | Runs | Best Val | Test | Params | Notes |
|---|---|---:|---:|---:|---:|---|
| thesis_cnn3 | thesis-faithful | 3 | 85.38 ± 0.77 | 79.33 ± 1.26 | 769804 | CNN-III thesis reproduction candidate. |
| thesis_cnniiii | thesis-faithful | 3 | 92.11 ± 0.30 | 85.26 ± 0.97 | 1378124 | CNN-IIII best thesis classical model with augmentation. |
| thesis_hqnn2 | thesis-faithful | 3 | 83.72 ± 2.23 | 78.61 ± 0.69 | 248428 | 2-qubit non-entangled quantum preprocessing with 2 filters producing 4 channels. Spatial implementation chosen to match the reported parameter count (248,428). |

## trainable-quantum-case-study

| Model | Source | Runs | Best Val | Test | Params | Notes |
|---|---|---:|---:|---:|---:|---|
| V7_trainable_quantum_documented | docs/notebook | 1 | 67.35 ± 0.00 | 65.02 ± 0.00 | 87798 | Documented stabilized V7 result from docs/notebook output. |
| V7_trainable_quantum_rerun | colab-l4-user-log | 1 | 72.89 ± 0.00 | 72.53 ± 0.00 | 87798 | Fresh Colab L4 rerun reconstructed from the user-provided terminal log on April 7, 2026. Drive-backed checkpoint files are now synced locally, but the remote experiments/v7_* directory has not yet been recovered into the repo workspace. |
