# Benchmark Matrix

**Date:** March 22, 2026

This document tracks:

- models with results already available
- thesis reference models
- publication-priority models that still need training or reruns

## 0. Current Compute Budget Snapshot

As of April 23, 2026:

- Google Colab budget currently available: `245` compute units
- Monthly renewal: `100` compute units
- Practical policy for this budget:
  - keep thesis-faithful classical and light non-trainable work on the M4 Mac
  - the trainable-quantum confirmation is already complete, so reserve Colab mainly for only clearly paper-relevant extension studies
  - do **not** spend Colab units on work that is already feasible on the Mac

Recommended allocation:

| Compute bucket | Platform | Expected CU usage | Purpose |
|---|---|---:|---|
| `thesis_cnn3`, `thesis_cnniiii` | M4 Mac | 0 | thesis classical reproduction |
| `thesis_hqnn2` | M4 Mac by default | 0 | thesis-best quantum reproduction via cached preprocessing |
| local reruns: `classical_conv`, `param_linear`, `non_trainable_quantum` | M4 Mac | 0 | protocol-v1 multi-seed benchmark |
| `V7 trainable quantum` confirmatory rerun | Colab L4/A100 | completed | one reproducible trainable-quantum confirmation run |
| P2 extension work | Colab only if needed | remaining budget | low-data study, stronger baselines, or extra V7 seeds |

This means the immediate next training steps should stay on the Mac unless a new experiment is both paper-relevant and clearly not M4-feasible.

## 1. Current Result Inventory

| Model | Source | Type | Best Val | Test | Params | Status | Notes |
|---|---|---|---:|---:|---:|---|---|
| `classical_conv` | repo-local + publication_v1 rerun | classical matched baseline | 88.01 | 82.62 | 88,045 | reproduced three times | current 3-seed summary is `81.40 ± 1.06` test; seed 43 matches the older pre-protocol local claim (`82.62%`) |
| `param_linear` | repo-local + publication_v1 rerun | 25-param matched replacement | 87.13 | 83.69 | 87,798 | reproduced three times | current 3-seed summary is `81.12 ± 2.27` test; strongest single seed reaches `83.69%` |
| `non_trainable_quantum` | repo-local + publication_v1 rerun | Henderson-style cached quantum | 86.84 | 80.90 | 88,488 | reproduced three times | current 3-seed summary is `80.40 ± 0.69` test; runtime excludes the one-time quantum cache/precompute stage |
| `resnet18_cifar_gray` | modern-baseline + publication_v1 rerun | stronger modern classical baseline | 93.27 | 89.06 | 11,190,252 | reproduced three times | current 3-seed summary is `88.13 ± 0.82` test; reviewer-proof modern classical upper bound on the same fixed split |
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
| P2 | low-data shortlist | 10/25/50/100% train fractions | 3 | M4 + selective GPU | not implemented yet |
| P2 | transfer-learning classical upper bound | only if reviewer pressure justifies it | 1-3 | Colab only if justified | not implemented |

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

# Stronger modern classical baseline
python train_modern_baselines.py --model resnet18_cifar_gray --seed 42 --split-seed 42
```

## 5. Practical Next Order

1. Treat `resnet18_cifar_gray` as the current reviewer-proof modern classical upper bound and keep it analytically separate from thesis-faithful and matched-budget families.
2. Keep Colab in reserve; do not spend the current `245` CU budget on work that is already M4-feasible.
3. Tighten the paper and submission-facing documents around the now-complete benchmark picture.
4. If additional empirical work is still needed, prefer a low-data pilot on the Mac before any new Colab training.
5. Only consider extra Colab experiments if they are likely to change reviewer risk more than they consume compute budget.
