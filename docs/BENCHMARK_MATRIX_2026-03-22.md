# Benchmark Matrix

**Date:** March 22, 2026

This document tracks:

- models with results already available
- thesis reference models
- publication-priority models that still need training or reruns

## 0. Current Compute Budget Snapshot

As of March 22, 2026:

- Google Colab budget available this month: `194` compute units
- Monthly renewal: `100` compute units
- Practical policy for this budget:
  - keep thesis-faithful classical and light non-trainable work on the M4 Mac
  - reserve Colab units mainly for trainable-quantum confirmation and only then for stronger extension studies
  - do **not** spend Colab units on work that is already feasible on the Mac

Recommended allocation:

| Compute bucket | Platform | Expected CU usage | Purpose |
|---|---|---:|---|
| `thesis_cnn3`, `thesis_cnniiii` | M4 Mac | 0 | thesis classical reproduction |
| `thesis_hqnn2` | M4 Mac by default | 0 | thesis-best quantum reproduction via cached preprocessing |
| local reruns: `classical_conv`, `param_linear`, `non_trainable_quantum` | M4 Mac | 0 | protocol-v1 multi-seed benchmark |
| `V7 trainable quantum` confirmatory rerun | Colab L4/A100 | ~40-60 | one reproducible trainable-quantum confirmation run |
| P2 extension work | Colab only if needed | remaining budget | low-data study, stronger baselines, or extra V7 seeds |

This means the immediate next training steps should stay on the Mac.

## 1. Current Result Inventory

| Model | Source | Type | Best Val | Test | Params | Status | Notes |
|---|---|---|---:|---:|---:|---|---|
| `classical_conv` | repo-local + publication_v1 rerun | classical matched baseline | 88.01 | 82.62 | 88,045 | reproduced three times | current 3-seed summary is `81.40 ± 1.06` test; seed 43 matches the older pre-protocol local claim (`82.62%`) |
| `param_linear` | repo-local + publication_v1 rerun | 25-param matched replacement | 87.13 | 83.69 | 87,798 | reproduced three times | current 3-seed summary is `81.12 ± 2.27` test; strongest single seed reaches `83.69%` |
| `non_trainable_quantum` | repo-local + publication_v1 rerun | Henderson-style cached quantum | 86.84 | 80.90 | 88,488 | reproduced three times | current 3-seed summary is `80.40 ± 0.69` test; runtime excludes the one-time quantum cache/precompute stage |
| `V7 trainable quantum` | docs/notebook | trainable quantum | 67.35 | 65.02 | 87,798 | documented | engineering case study, not benchmark winner |
| `V4` | historical docs | old non-trainable | 8.75 | - | historical | historical | no longer a sufficient main reference |
| `CNN-III` | thesis + local reproduction | classical | 85.96 | 80.26 | 769,804 | reproduced three times | current 3-seed summary is `79.33 ± 1.26` test; all local runs remain below the thesis reference (`83.05%`) |
| `CNN-IIII` | thesis + local reproduction | classical | 92.40 | 86.27 | 1,378,124 | reproduced three times | all three local seeds are above the thesis reference (`83.69%`); current 3-seed summary is `85.26 ± 0.97` test |
| `HQNN-I` | thesis | non-trainable quantum | - | 79.83 | 100,204 | thesis-only | historical thesis quantum |
| `HQNN-II` | thesis + local reproduction | non-trainable quantum | 85.67 | 79.40 | 248,428 | reproduced three times | current 3-seed summary is `78.61 ± 0.69` test; all local runs remain below the thesis reference (`82.40%`) |
| `HQNN-III` | thesis | non-trainable quantum | 87.46 | 80.47 | 241,180 | thesis-only | entangled thesis variant |

## 2. Publication Training Backlog

| Priority | Model | Goal | Seeds | Platform | Entry Point |
|---|---|---|---:|---|---|
| P0 | `thesis_hqnn2` | faithful best-thesis quantum reproduction | 3 | M4 default, Colab optional | `train_thesis_models.py --model thesis_hqnn2` |
| P0 | `thesis_cnn3` | classical pairwise baseline for HQNN-II | 3 | M4 | `train_thesis_models.py --model thesis_cnn3` |
| P0 | `thesis_cnniiii` | faithful best-thesis classical reproduction | 3 | M4 | `train_thesis_models.py --model thesis_cnniiii` |
| P1 | `classical_conv` rerun | protocol-v1 deterministic multi-seed benchmark | 3 | M4 | `train_ablation_local.py --model classical_conv` |
| P1 | `param_linear` rerun | protocol-v1 deterministic multi-seed benchmark | 3 | M4 | `train_ablation_local.py --model param_linear` |
| P1 | `non_trainable_quantum` rerun | protocol-v1 deterministic multi-seed benchmark | 3 | M4 | `train_ablation_local.py --model non_trainable_quantum` |
| P1 | `V7 trainable quantum` rerun | confirmatory reproducible trainable-quantum case | 1 now | Colab L4/A100 only | `train_v7.py` |
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

1. On the Mac, run `thesis_cnn3`, `thesis_hqnn2`, and `thesis_cnniiii` once each with seed 42.
2. Check whether reproduced metrics are reasonably close to thesis references.
3. On the Mac, run 3-seed reruns for `classical_conv`, `param_linear`, and `non_trainable_quantum`.
4. Aggregate results into `docs/BENCHMARK_SUMMARY.md`.
5. Spend Colab units only after the Mac shortlist is stable:
   - first on one `V7 trainable quantum` confirmatory rerun
   - then on optional extension experiments if the remaining unit budget is still healthy
6. Rewrite paper claims around the resulting table, not around V4-vs-V7 alone.
