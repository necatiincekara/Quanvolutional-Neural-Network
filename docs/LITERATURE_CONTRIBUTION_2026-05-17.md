# Literature Contribution

**Date:** May 17, 2026

This note summarizes the literature-facing contribution of the study without overstating the quantum claim.

## Core Position

The paper should be positioned as a reproducible benchmark and hybrid-QML engineering study, not as a generic quantum-advantage paper.

The contribution is that the study makes the quantum OCR hypothesis more scientifically precise:

- where quantum variants do not win under full-data reproducible benchmarking,
- where one non-trainable quantum preprocessing baseline becomes narrowly competitive in a current-local low-data regime,
- and which engineering constraints determine whether trainable quanvolution learns or collapses.

## Contribution To Quanvolutional Neural Network Literature

Most quanvolutional studies are easy to misread if all quantum models and all classical baselines are collapsed into one leaderboard. This study contributes a cleaner evidence structure:

- thesis-faithful CNN/HQNN reproductions,
- current-local matched-budget ablations,
- a stronger modern-classical upper bound,
- a low-data scaling axis,
- and a trainable-quantum engineering case-study.

This family separation is itself a methodological contribution. It prevents unsupported comparisons such as treating `thesis_hqnn2`, `non_trainable_quantum`, and V7 as one interchangeable "quantum model" category.

## Contribution To QML Benchmarking

The full-data result is a disciplined negative result:

- `resnet18_cifar_gray` is strongest overall at `88.13 ± 0.82%` test.
- `thesis_cnniiii` is strongest thesis-faithful at `85.26 ± 0.97%`.
- `classical_conv` is strongest current-local full-data matched-budget at `81.40 ± 1.06%`.
- `non_trainable_quantum` is close but not the full-data current-local leader at `80.40 ± 0.69%`.
- V7 trainable quantum remains below the classical anchors and is treated as an engineering case-study.

This contributes to the benchmarking literature by showing how the claim changes when stronger classical baselines, multi-seed reporting, and artifact-backed logs are used. The paper's value is partly that it refuses to turn a weak or partial result into a broad quantum-win narrative.

## Contribution To Hybrid QML Engineering

The V1--V7 path contributes practical engineering evidence:

- feature maps below 8x8 caused information bottleneck failure on this 44-class task,
- gradient routing, residual structure, channel attention, and learnable scaling were needed before the trainable quantum path became viable,
- AMP float16 autocasting at the quantum boundary caused NaN corruption unless float32 was restored and GradScaler-aware stepping was used,
- Colab reruns show V7 can train without NaN collapse, but not that it is an accuracy leader.

These are reusable lessons for hybrid QML practitioners because they describe failure modes that are easy to miss if only final accuracy is reported.

## Contribution To Ottoman-Turkish OCR / Cultural Heritage Computing

The paper turns a thesis-era Ottoman-Turkish handwritten character recognition study into a stricter reproducible benchmark:

- fixed thesis-era train/test continuity,
- 44-class small-data OCR setting,
- multi-seed reruns for key models,
- stronger modern classical control,
- low-data scaling analysis,
- and a clear distinction between historical thesis claims and current artifact-backed evidence.

For cultural-heritage OCR, the contribution is not only the best accuracy row. It is the construction of a more reliable benchmark story around a small, historically meaningful script-recognition task.

## Low-Data Contribution

The May 2026 low-data confirmation adds the main positive quantum nuance:

| Fraction | `classical_conv` Test | `non_trainable_quantum` Test | Quantum Lead |
|---:|---:|---:|---:|
| 0.10 | 48.42 ± 2.31 | 50.71 ± 2.93 | +2.29 |
| 0.25 | 66.24 ± 1.78 | 69.88 ± 0.99 | +3.64 |
| 0.50 | 75.61 ± 1.02 | 76.75 ± 0.50 | +1.14 |
| 1.00 | 80.47 ± 0.57 | 80.76 ± 0.99 | +0.29 |

This is a scoped current-local signal. It should be described as low-data competitiveness for one non-trainable quantum preprocessing baseline, not as a result that generalizes to thesis HQNN-II, V7, quantum hardware, or other datasets.

## One-Sentence Literature Claim

This work contributes an artifact-backed, family-separated benchmark of quanvolutional models on a small Ottoman-Turkish OCR task, showing classical-favored full-data results, a narrow current-local low-data quantum competitiveness signal, and concrete hybrid-QML engineering failure modes that future studies can reproduce or challenge.
