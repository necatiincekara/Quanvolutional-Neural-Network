# Submission Benchmark And Claim Set

**Date:** May 16, 2026

This document freezes the benchmark picture that is currently safe to use for manuscript rewriting after the May 2026 low-data confirmation and paper cleanup pass.

Use this file together with:

- `docs/BENCHMARK_SUMMARY.md`
- `docs/PUBLICATION_STRATEGY_2026-03-22.md`
- `docs/EXPERIMENTS.md`
- `docs/LOW_DATA_SUMMARY.md`
- `paper/draft.md`

## 1. Main Benchmark Snapshot

### 1.1 Thesis-faithful family

| Model | Runs | Test | Best Val | Params | Interpretation |
|---|---:|---:|---:|---:|---|
| `thesis_cnniiii` | 3 | **85.26 ± 0.97** | **92.11 ± 0.30** | 1,378,124 | strongest thesis-faithful anchor; above thesis reference |
| `thesis_cnn3` | 3 | 79.33 ± 1.26 | 85.38 ± 0.77 | 769,804 | weaker classical thesis-faithful reference |
| `thesis_hqnn2` | 3 | 78.61 ± 0.69 | 83.72 ± 2.23 | 248,428 | best thesis-faithful quantum reproduction, but below thesis table and below `thesis_cnniiii` |

### 1.2 Current-local matched-budget family

| Model | Runs | Test | Best Val | Params | Interpretation |
|---|---:|---:|---:|---:|---|
| `classical_conv` | 3 | **81.40 ± 1.06** | 86.26 ± 1.76 | 88,045 | strongest current-local model by mean test accuracy |
| `param_linear` | 3 | 81.12 ± 2.27 | **86.45 ± 0.61** | 87,798 | matched classical replacement; nearly equal to `classical_conv` on mean test |
| `non_trainable_quantum` | 3 | 80.40 ± 0.69 | 85.77 ± 0.94 | 88,488 | stable Henderson-style non-trainable quantum baseline, but not the best current-local model |

### 1.3 Trainable quantum case-study

| Model | Runs | Test | Best Val | Params | Interpretation |
|---|---:|---:|---:|---:|---|
| `V7_trainable_quantum_rerun` | 1 resumed rerun | 72.53 | 72.89 | 87,798 | April 6 resumed trainable-quantum engineering case-study, still not current benchmark leader; Drive-backed checkpoint files are now synced locally |
| `V7_trainable_quantum_clean_20260427` | 1 clean non-resumed rerun | 65.88 | 69.97 | 87,798 | clean April 27-28 Colab L4 run reconstructed from captured notebook output after runtime disconnect; checkpoint files exist in Drive, but the JSON/experiment metadata was not copied before disconnect |
| `V7_trainable_quantum_documented` | 1 documented | 65.02 | 67.35 | 87,798 | older documented trainable result retained for historical comparison |

### 1.4 Modern-classical upper bound

| Model | Runs | Test | Best Val | Params | Interpretation |
|---|---:|---:|---:|---:|---|
| `resnet18_cifar_gray` | 3 | **88.13 ± 0.82** | **92.98 ± 0.29** | 11,190,252 | reviewer-proof stronger classical upper bound; useful for rebutting claims that the benchmark lacks a modern vision baseline |

### 1.5 Low-data scaling axis

Current-local rows are three-seed means over seeds 42, 43, and 44. Thesis-faithful rows are seed-42 pilot evidence only.

| Family | Fraction | Classical Test | Quantum Test | Gap C-Q | Interpretation |
|---|---:|---:|---:|---:|---|
| current-local | 0.10 | `classical_conv`: 48.42 ± 2.31 | `non_trainable_quantum`: 50.71 ± 2.93 | -2.29 | quantum ahead in this paired low-data comparison |
| current-local | 0.25 | `classical_conv`: 66.24 ± 1.78 | `non_trainable_quantum`: 69.88 ± 0.99 | -3.64 | quantum ahead in this paired low-data comparison |
| current-local | 0.50 | `classical_conv`: 75.61 ± 1.02 | `non_trainable_quantum`: 76.75 ± 0.50 | -1.14 | quantum ahead in this paired low-data comparison |
| current-local | 1.00 | `classical_conv`: 80.47 ± 0.57 | `non_trainable_quantum`: 80.76 ± 0.99 | -0.29 | quantum ahead narrowly in this paired comparison |
| thesis-faithful | 0.10 | `thesis_cnniiii`: 65.88 | `thesis_hqnn2`: 50.43 | 15.45 | seed-42 pilot remains classical-favored |
| thesis-faithful | 0.25 | `thesis_cnniiii`: 79.61 | `thesis_hqnn2`: 62.45 | 17.16 | seed-42 pilot remains classical-favored |
| thesis-faithful | 0.50 | `thesis_cnniiii`: 82.40 | `thesis_hqnn2`: 72.10 | 10.30 | seed-42 pilot remains classical-favored |
| thesis-faithful | 1.00 | `thesis_cnniiii`: 85.19 | `thesis_hqnn2`: 78.33 | 6.86 | seed-42 pilot remains classical-favored |

## 2. Safe Claim Hierarchy

These are the strongest claims currently supported by repository artifacts.

1. The repository now supports a **reproducible benchmark story**, not a quantum-win story.
2. In the current-local matched-budget family, **classical baselines outperform the current Henderson-style non-trainable quantum baseline on full-data mean test accuracy**.
3. In the thesis-faithful family, **the strongest reproduced model is classical (`thesis_cnniiii`)**.
4. A stronger modern classical upper bound now also exists: `resnet18_cifar_gray` reaches **88.13 ± 0.82%** test accuracy on the same fixed split.
5. The thesis-faithful quantum reproduction (`thesis_hqnn2`) is competitive with `thesis_cnn3`, but it does **not** surpass the strongest classical thesis-faithful model.
6. The May 2026 low-data confirmation supports a **specific current-local low-data competitiveness signal**: `non_trainable_quantum` exceeds `classical_conv` on three-seed mean test accuracy at the tested train fractions.
7. V7 remains valuable as a **hybrid-QML engineering and stabilization case-study**:
   - information bottleneck threshold
   - gradient stabilization
   - AMP / float16 incompatibility at the quantum boundary
   - Colab reruns without NaN failure, with observed single-run V7 test range from `65.88%` clean non-resumed to `72.53%` resumed

## 3. Claims To Avoid

Do not currently claim any of the following:

- generic "quantum advantage"
- that the trainable quantum path is the best model in the study
- that the Henderson-style non-trainable quantum baseline beats matched classical replacements under the full-data protocol
- that the May 2026 low-data signal generalizes to thesis HQNN-II, trainable V7, other datasets, quantum hardware, or generic quantum advantage
- that thesis HQNN-II has been surpassed by a quantum successor
- that simulator-side quantum methods provide a clear practical compute advantage

## 4. Recommended Main Paper Narrative

The paper should now be written around:

1. a **fair comparative benchmark** on a difficult 44-class Ottoman handwriting task,
2. **negative-result honesty** about where current quantum variants do not win,
3. a **narrow current-local low-data competitiveness signal** for a non-trainable quantum preprocessing baseline,
4. **engineering insight** from the V1--V7 trainable-quantum path,
5. the distinction between:
   - thesis-faithful reproductions,
   - current-local matched-budget ablations,
   - modern-classical upper bound,
   - low-data scaling,
   - documented trainable-quantum case-study.

## 5. Required Paper Edits

The current `paper/draft.md` now addresses the main rewrite requirements. Before submission, verify that:

1. the abstract and conclusion still avoid generic quantum-advantage language,
2. RQ1/RQ2/RQ3 remain aligned with the benchmark snapshot in this document,
3. V7 is framed as an engineering contribution rather than an accuracy leader,
4. current-local, thesis-faithful, modern-classical, low-data, and trainable-quantum families are not mixed into one misleading ranking,
5. low-data language stays limited to the current-local paired comparison unless new artifacts expand the claim.

## 6. Remaining High-Value Work

The most valuable next steps are now:

1. keep the reconstructed April 2026 V7 JSON rows clearly labeled and recover copied remote `experiments/v7_*` metadata only if it is still available; the April 27 Drive folder currently contains checkpoints but an empty `experiments/` subfolder,
2. keep `paper/draft.md`, this claim set, and Word exports synchronized after any result-language change,
3. add confidence intervals or formal significance tests if the target venue or reviewers are expected to push on variance,
4. test whether the current-local low-data signal transfers to a second dataset, robustness axis, or alternative matched-budget classical controls if aiming above a specialized/Q2 route,
5. use Colab only for a paper-impactful extension, not for V7 artifact hygiene or already-completed low-data rows.
