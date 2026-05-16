# Benchmark Matrix

**Date:** March 22, 2026

**Latest update:** May 16, 2026

This document tracks:

- models with results already available
- thesis reference models
- publication-priority models that still need training or reruns

## 0. Current Compute Budget Snapshot

As of May 16, 2026:

- Google Colab budget last user-reported before the low-data confirmation: `245` compute units
- Monthly renewal: `100` compute units, with a practical storage cap around `300` units
- Practical policy for this budget:
  - keep thesis-faithful classical and light non-trainable work on the M4 Mac
  - the trainable-quantum confirmations are complete, so do not spend Colab units only for V7 folder hygiene
  - the low-data current-local confirmation is complete: seeds 43 and 44 were run on Colab L4 for `classical_conv` and `non_trainable_quantum`
  - because unused surplus above the cap can be lost after renewal, spending roughly `45-50` units is acceptable if the experiment is paper-relevant and not merely duplicative
  - do **not** spend Colab units on work that is already feasible on the Mac unless doing so saves meaningful time for a high-impact paper audit or extension

Recommended allocation:

| Compute bucket | Platform | Expected CU usage | Purpose |
|---|---|---:|---|
| `thesis_cnn3`, `thesis_cnniiii` | M4 Mac | 0 | thesis classical reproduction |
| `thesis_hqnn2` | M4 Mac by default | 0 | thesis-best quantum reproduction via cached preprocessing |
| local reruns: `classical_conv`, `param_linear`, `non_trainable_quantum` | M4 Mac | 0 | protocol-v1 multi-seed benchmark |
| `V7 trainable quantum` confirmatory rerun | Colab L4/A100 | completed | one reproducible trainable-quantum confirmation run |
| low-data current-local confirmation | Colab L4 | completed | confirmed the seed-42 low-data signal for `classical_conv` vs `non_trainable_quantum` with seeds 43/44 |

This means there is no default next Colab training step. Further Colab use should require a new reviewer-facing question.

## 1. Current Result Inventory

| Model | Source | Type | Best Val | Test | Params | Status | Notes |
|---|---|---|---:|---:|---:|---|---|
| `classical_conv` | repo-local + publication_v1 rerun | classical matched baseline | 88.01 | 82.62 | 88,045 | reproduced three times | current 3-seed summary is `81.40 ± 1.06` test; seed 43 matches the older pre-protocol local claim (`82.62%`) |
| `param_linear` | repo-local + publication_v1 rerun | 25-param matched replacement | 87.13 | 83.69 | 87,798 | reproduced three times | current 3-seed summary is `81.12 ± 2.27` test; strongest single seed reaches `83.69%` |
| `non_trainable_quantum` | repo-local + publication_v1 rerun | Henderson-style cached quantum | 86.84 | 80.90 | 88,488 | reproduced three times | current 3-seed summary is `80.40 ± 0.69` test; runtime excludes the one-time quantum cache/precompute stage |
| low-data current-local confirmation | `experiments/low_data_summary.json` | low-data scaling axis | see summary | see summary | - | 3-seed confirmation complete | `non_trainable_quantum` exceeds `classical_conv` on mean test accuracy at 10/25/50/100% train fractions: gaps C-Q are `-2.29`, `-3.64`, `-1.14`, `-0.29`; this is a specific low-data competitiveness signal, not a generic advantage claim |
| low-data thesis-faithful pilot | `experiments/low_data_summary.json` | low-data scaling axis | see summary | see summary | - | seed-42 pilot complete | `thesis_cnniiii` remains ahead of `thesis_hqnn2` at every tested train fraction; no thesis-faithful Colab follow-up is justified by this pilot |
| `resnet18_cifar_gray` | modern-baseline + publication_v1 rerun | stronger modern classical baseline | 93.27 | 89.06 | 11,190,252 | reproduced three times | current 3-seed summary is `88.13 ± 0.82` test; reviewer-proof modern classical upper bound on the same fixed split |
| `V7 trainable quantum rerun` | colab-l4-user-log + local checkpoints | trainable quantum | 72.89 | 72.53 | 87,798 | rerun complete | April 6, 2026 resumed Colab rerun; still below strongest classical anchors; local checkpoints are synced, but remote `experiments/v7_*` metadata remains a provenance gap |
| `V7 trainable quantum clean` | colab-l4-notebook-output + Drive checkpoints | trainable quantum | 69.97 | 65.88 | 87,798 | clean rerun complete | April 27, 2026 clean non-resumed L4 run; JSON reconstructed from captured notebook output because runtime disconnected before Drive copy; confirms trainability but not benchmark leadership |
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
| P1 | `V7 trainable quantum` artifact sync | close provenance gap for completed trainable-quantum rerun | 0 new runs | local + Drive/Colab artifact recovery | `./scripts/codex-colab-handoff.sh` |
| P2 | `thesis_hqnn3` | entanglement ablation | 1-3 | M4/Colab | `train_thesis_models.py --model thesis_hqnn3` |
| P2 | low-data current-local confirmation | confirmed `classical_conv` vs `non_trainable_quantum` low-data signal | seeds 43/44 complete | Colab L4 completed | `scripts/run_low_data_grid.py --execute --models classical_conv non_trainable_quantum` |
| P2 | low-data thesis-faithful confirmation | only if a new artifact changes the current pilot picture | 0 by default | none | no follow-up recommended from seed-42 pilot |
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

# Low-data scaling pilot and aggregation
python scripts/run_low_data_grid.py --execute --protocol-version low_data_pilot_v1 --models classical_conv non_trainable_quantum thesis_cnniiii thesis_hqnn2 --fractions 0.10 0.25 0.50 1.00 --seeds 42 --split-seed 42
python scripts/aggregate_low_data.py

# Low-data current-local confirmation, completed on Colab L4 in May 2026
python scripts/run_low_data_grid.py --execute --protocol-version low_data_confirm_v1 --models classical_conv non_trainable_quantum --fractions 0.10 0.25 0.50 1.00 --seeds 43 44 --split-seed 42 --device auto

# Stronger modern classical baseline
python train_modern_baselines.py --model resnet18_cifar_gray --seed 42 --split-seed 42
```

## 5. Practical Next Order

1. Treat `resnet18_cifar_gray` as the current reviewer-proof modern classical upper bound and keep it analytically separate from thesis-faithful and matched-budget families.
2. Use Colab deliberately: do not rerun V7 for folder hygiene, but a `45-50` CU paper-impact experiment is reasonable before the next renewal because surplus above the cap may be lost.
3. Tighten the paper and submission-facing documents around the now-complete benchmark picture.
4. Keep the reconstructed V7 rows clearly labeled and recover missing remote `experiments/v7_*` metadata only if it is still available; the April 27 Drive `experiments/` subfolder is empty.
5. Treat the current-local low-data confirmation as implemented. It supports a specific low-data competitiveness signal for `non_trainable_quantum`, not a generic quantum-advantage claim.
6. Do not spend more Colab on low-data unless the paper review process asks for a narrower follow-up. The next work should be claim integration, figure/table preparation, and paper cleanup.
