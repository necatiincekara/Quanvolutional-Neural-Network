# Benchmark Matrix

**Date:** March 22, 2026

This document tracks:

- models with results already available
- thesis reference models
- publication-priority models that still need training or reruns

## 1. Current Result Inventory

| Model | Source | Type | Best Val | Test | Params | Status | Notes |
|---|---|---|---:|---:|---:|---|---|
| `classical_conv` | repo-local | classical matched baseline | 84.50 | 82.62 | 88,045 | trained | current best local test |
| `param_linear` | repo-local | 25-param matched replacement | 88.30 | 81.76 | 87,798 | trained | strong matched classical baseline |
| `non_trainable_quantum` | repo-local | Henderson-style cached quantum | 88.89 | 80.47 | 88,488 | trained | current local non-trainable quantum baseline |
| `V7 trainable quantum` | docs/notebook | trainable quantum | 67.35 | 65.02 | 87,798 | documented | engineering case study, not benchmark winner |
| `V4` | historical docs | old non-trainable | 8.75 | - | historical | historical | no longer a sufficient main reference |
| `CNN-III` | thesis | classical | 89.50 | 83.05 | 769,804 | thesis-only | pairwise reference for HQNN-II |
| `CNN-IIII` | thesis | classical | 84.21 | 83.69 | 1,378,124 | thesis-only | best thesis classical |
| `HQNN-I` | thesis | non-trainable quantum | - | 79.83 | 100,204 | thesis-only | historical thesis quantum |
| `HQNN-II` | thesis | non-trainable quantum | 86.88 | 82.40 | 248,428 | thesis-only | best thesis quantum |
| `HQNN-III` | thesis | non-trainable quantum | 87.46 | 80.47 | 241,180 | thesis-only | entangled thesis variant |

## 2. Publication Training Backlog

| Priority | Model | Goal | Seeds | Platform | Entry Point |
|---|---|---|---:|---|---|
| P0 | `thesis_hqnn2` | faithful best-thesis quantum reproduction | 3 | M4 or Colab | `train_thesis_models.py --model thesis_hqnn2` |
| P0 | `thesis_cnn3` | classical pairwise baseline for HQNN-II | 3 | M4 | `train_thesis_models.py --model thesis_cnn3` |
| P0 | `thesis_cnniiii` | faithful best-thesis classical reproduction | 3 | M4 | `train_thesis_models.py --model thesis_cnniiii` |
| P1 | `classical_conv` rerun | protocol-v1 deterministic multi-seed benchmark | 3 | M4 | `train_ablation_local.py --model classical_conv` |
| P1 | `param_linear` rerun | protocol-v1 deterministic multi-seed benchmark | 3 | M4 | `train_ablation_local.py --model param_linear` |
| P1 | `non_trainable_quantum` rerun | protocol-v1 deterministic multi-seed benchmark | 3 | M4 | `train_ablation_local.py --model non_trainable_quantum` |
| P1 | `V7 trainable quantum` rerun | confirmatory reproducible trainable-quantum case | 1 now | Colab L4/A100 | `train_v7.py` |
| P2 | `thesis_hqnn3` | entanglement ablation | 1-3 | M4/Colab | `train_thesis_models.py --model thesis_hqnn3` |
| P2 | `resnet18_cifar_gray` | stronger classical reviewer-proof baseline | 3 | M4/Colab | not implemented yet |
| P2 | low-data shortlist | 10/25/50/100% train fractions | 3 | M4 + selective GPU | not implemented yet |

## 3. Protocol V1 Rules

- Use explicit `--seed` and `--split-seed`.
- Save one canonical JSON per seed and split combination.
- Include `protocol_version`, `platform`, `params`, `runtime`, and dataset sizes.
- Cache non-trainable quantum features with a hash based on circuit spec, image size, dataset path, and seed.
- Keep thesis-faithful models separate from current Henderson-style local non-trainable quantum.

## 4. Commands

```bash
# Current local benchmarks
python train_ablation_local.py --model classical_conv --epochs 50 --seed 42 --split-seed 42
python train_ablation_local.py --model param_linear --epochs 50 --seed 42 --split-seed 42
python train_ablation_local.py --model non_trainable_quantum --epochs 50 --seed 42 --split-seed 42

# Thesis-faithful models
python train_thesis_models.py --model thesis_cnn3 --seed 42 --split-seed 42
python train_thesis_models.py --model thesis_cnniiii --seed 42 --split-seed 42
python train_thesis_models.py --model thesis_hqnn2 --seed 42 --split-seed 42

# Aggregate benchmark tables
python scripts/aggregate_benchmarks.py
```

## 5. Practical Next Order

1. Run `thesis_hqnn2`, `thesis_cnn3`, and `thesis_cnniiii` once each with seed 42.
2. Check whether reproduced metrics are reasonably close to thesis references.
3. Run 3-seed reruns for `classical_conv`, `param_linear`, and `non_trainable_quantum`.
4. Aggregate results into `docs/BENCHMARK_SUMMARY.md`.
5. Rewrite paper claims around the resulting table, not around V4-vs-V7 alone.
