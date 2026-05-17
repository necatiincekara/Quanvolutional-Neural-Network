# V8 Quantum Extension Decision

**Date:** May 17, 2026  
**Decision:** Do not launch a new V7 or V8 training run now. Treat V8 as a design-note and smoke-test candidate only.

## Evidence Base

- V7 is not competitive with the main classical anchors. The best artifact-backed V7 rerun is the April 6, 2026 resumed Colab L4 run with `72.89%` best validation and `72.53%` test accuracy.
- The strongest current full-data result is `resnet18_cifar_gray` at `88.13 ± 0.82%` test accuracy.
- The strongest thesis-faithful result is `thesis_cnniiii` at `85.26 ± 0.97%` test accuracy.
- The May 2026 low-data signal belongs to `non_trainable_quantum` versus `classical_conv`, not to V7.
- Current QML benchmarking literature supports strong classical comparators and cautious claims, not open-ended quantum reruns.

## Why No Immediate Training

New V7 training would mostly spend compute to re-confirm a non-leading trainable-quantum case-study. New V8 training is not justified until the architecture has a targeted rationale and an early smoke result showing better gradient health or a credible low-data/robustness contribution.

## Candidate V8 Directions

| Direction | Rationale | First Artifact | Stop Condition |
|---|---|---|---|
| Residual/deep quanvolution | Deep quanvolution literature emphasizes residual paths and gradient propagation. | `docs/V8_DESIGN_NOTE.md` plus a gradient-health smoke on a tiny subset. | Stop if gradients remain weaker than V7 or accuracy does not move above a matched classical replacement in smoke. |
| Equivariant/local-cost ansatz | 2026 equivariant QCNN work suggests symmetry and locality may improve trainability and measurement efficiency. | A circuit review note and a toy ansatz benchmark, not full OCR training. | Stop if the ansatz increases runtime without measurable gradient or validation benefit. |
| Low-data robustness extension | The positive current-local signal is in `non_trainable_quantum`, so robustness/low-data is the most evidence-aligned quantum axis. | Add second-seed/fraction or perturbation tests only for the current-local pair. | Stop if the signal disappears or remains statistically weak under additional controls. |

## Mandatory Gates Before V8 Training

Any V8 training proposal must pass all of these gates:

1. Preserve at least `8x8` pre-quantum spatial resolution or justify the information bottleneck explicitly.
2. Keep AMP away from the quantum boundary.
3. Include a matched classical replacement with comparable parameter and data flow.
4. Show early gradient health improvement over V7 on a small smoke run.
5. Define target metric, expected runtime, platform, and stop condition before using Colab.
6. Keep benchmark families separate: thesis-faithful, current-local, modern-classical, and trainable-quantum case-study.

## Colab Policy

No Colab CU should be spent for V7/V8 artifact hygiene. Colab is justified only if a small local or cheap remote smoke produces paper-impact evidence, such as a clear low-data robustness signal, improved trainability over V7, or reviewer-requested confirmation.

## Publication Recommendation

For the current submission, use V7 as an engineering case-study and emphasize:

- strong classical full-data hierarchy,
- a scoped current-local low-data quantum signal,
- trainable-quantum stabilization lessons,
- and clear limits on quantum advantage claims.

V8 belongs in future work unless the design-note plus smoke-test path produces artifact-backed evidence.
