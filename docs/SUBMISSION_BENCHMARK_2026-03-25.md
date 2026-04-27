# Submission Benchmark And Claim Set

**Date:** March 25, 2026

This document freezes the benchmark picture that is currently safe to use for manuscript rewriting.

Use this file together with:

- `docs/BENCHMARK_SUMMARY.md`
- `docs/PUBLICATION_STRATEGY_2026-03-22.md`
- `docs/EXPERIMENTS.md`

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
| `V7_trainable_quantum_rerun` | 1 fresh rerun | 72.53 | 72.89 | 87,798 | improved trainable-quantum engineering case-study, still not current benchmark leader; Drive-backed checkpoint files are now synced locally |
| `V7_trainable_quantum_documented` | 1 documented | 65.02 | 67.35 | 87,798 | older documented trainable result retained for historical comparison |

### 1.4 Modern-classical upper bound

| Model | Runs | Test | Best Val | Params | Interpretation |
|---|---:|---:|---:|---:|---|
| `resnet18_cifar_gray` | 3 | **88.13 ± 0.82** | **92.98 ± 0.29** | 11,190,252 | reviewer-proof stronger classical upper bound; useful for rebutting claims that the benchmark lacks a modern vision baseline |

## 2. Safe Claim Hierarchy

These are the strongest claims currently supported by repository artifacts.

1. The repository now supports a **reproducible benchmark story**, not a quantum-win story.
2. In the current-local matched-budget family, **classical baselines outperform the current Henderson-style non-trainable quantum baseline on mean test accuracy**.
3. In the thesis-faithful family, **the strongest reproduced model is classical (`thesis_cnniiii`)**.
4. A stronger modern classical upper bound now also exists: `resnet18_cifar_gray` reaches **88.13 ± 0.82%** test accuracy on the same fixed split.
5. The thesis-faithful quantum reproduction (`thesis_hqnn2`) is competitive with `thesis_cnn3`, but it does **not** surpass the strongest classical thesis-faithful model.
6. V7 remains valuable as a **hybrid-QML engineering and stabilization case-study**:
   - information bottleneck threshold
   - gradient stabilization
   - AMP / float16 incompatibility at the quantum boundary
   - reproducible Colab rerun to `72.53%` test without NaN failure

## 3. Claims To Avoid

Do not currently claim any of the following:

- generic "quantum advantage"
- that the trainable quantum path is the best model in the study
- that the Henderson-style non-trainable quantum baseline beats matched classical replacements
- that thesis HQNN-II has been surpassed by a quantum successor
- that simulator-side quantum methods provide a clear practical compute advantage

## 4. Recommended Main Paper Narrative

The paper should now be written around:

1. a **fair comparative benchmark** on a difficult 44-class Ottoman handwriting task,
2. **negative-result honesty** about where current quantum variants do not win,
3. **engineering insight** from the V1--V7 trainable-quantum path,
4. the distinction between:
   - thesis-faithful reproductions,
   - current-local matched-budget ablations,
   - documented trainable-quantum case-study.

## 5. Required Paper Edits

Before submission, the draft should be rewritten so that:

1. the abstract no longer reads as if V7 is the main winning model,
2. the primary result table uses the benchmark snapshot in this document,
3. V7 is framed as an engineering contribution rather than an accuracy leader,
4. current-local and thesis-faithful families are not mixed into one misleading ranking,
5. the conclusion explicitly states that the strongest reproduced evidence currently favors classical baselines.

## 6. Remaining High-Value Work

The most valuable next steps are now:

1. sync the remote `experiments/v7_*` directory back into the repo workspace if it is still recoverable, so the rerun is represented by more than checkpoints plus the reconciled log-backed JSON row,
2. rewrite and tighten `paper/draft.md` around the benchmark hierarchy above,
3. refresh submission-facing summary documents and Word exports,
4. add significance intervals/tests if reviewers are expected to push on variance,
5. only then consider a low-data pilot or another narrowly justified extension if it materially improves reviewer resilience and is worth spending part of the remaining `245` Colab computing units.
