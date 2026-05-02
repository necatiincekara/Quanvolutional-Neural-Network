# Next Phase Master Plan

**Date:** April 5, 2026

This document is the post-upgrade master plan for the study. It assumes the current benchmark truth remains unchanged and uses the upgraded Codex layer to reduce ambiguity in the next research phase.

> Update, April 30, 2026:
> The April 6 resumed Colab V7 rerun remains complete at `72.89%` best validation and `72.53%` test accuracy.
> The April 27 clean non-resumed Colab V7 rerun reached `69.97%` best validation and `65.88%` test; its JSON row is reconstructed from captured notebook output because the runtime disconnected before the artifact-copy cell copied JSON to Drive.
> The stronger modern classical baseline `resnet18_cifar_gray` has now completed three local seeds at `88.13 ± 0.82%` test accuracy.
> Remaining Colab budget is approximately `245` computing units.
> Default policy is still conservative for V7 reruns, but spending roughly `45-50` surplus Colab units can be justified if the experiment is paper-impactful before the next renewal cap.

## 1. Immediate Operating Sequence

These are the next exact actions, in order:

1. **Reconcile the completed `resnet18_cifar_gray` three-seed result into all benchmark-facing documents**
   - Goal: keep the new modern classical upper bound separated from thesis-faithful and matched-budget families while making the reviewer-proof benchmark picture explicit.
   - Platform: local + Drive
   - Paper impact: high

2. **Finish the remaining V7 artifact sync and reconciliation**
   - Goal: keep `experiments/*.json`, `docs/BENCHMARK_SUMMARY.md`, `docs/EXPERIMENTS.md`, and `paper/draft.md` aligned, and recover copied remote `experiments/v7_*` metadata only if it is still accessible.
   - Platform: local
   - Paper impact: high

3. **Tighten the manuscript around the now-complete benchmark picture**
   - Goal: move the paper fully into benchmark + engineering-lessons mode and remove any remaining V7-centric phrasing.
   - Platform: local
   - Paper impact: high

4. **Refresh advisor / submission / Word export artifacts**
   - Goal: keep shareable documents synchronized with the post-rerun benchmark truth.
   - Platform: local
   - Paper impact: medium

5. **Re-evaluate whether any extra training is still worth compute**
   - Goal: do not rerun V7 just for artifact hygiene, but use expiring Colab surplus for a high-impact experiment if it materially changes reviewer resilience or the paper decision boundary.
   - Platform: local planning only
   - Paper impact: medium

## 2. Track A — Near-Term Publishable Route

### Goal

Finish a credible Q2 / specialized-QML submission without implying unsupported quantum superiority.

### Exact Task Order

1. **Artifact sync and V7 result reconciliation**
   - Platform: local + existing Drive artifacts
   - Goal: keep the reconstructed April 2026 V7 JSON rows clearly labeled, preserve existing Drive checkpoints, and pull copied `experiments/v7_*` metadata only if it is still available
   - Paper impact: high
   - Stop condition: complete once benchmark-facing docs agree that the April 27 clean run has Drive checkpoints but no copied Drive JSON/experiment metadata

2. **Paper synchronization after the rerun**
   - Platform: local
   - Tools: `codex-paper-audit.sh`, `paper-sync`, `paper_writer`
   - Paper impact: high

3. **Submission-facing document refresh**
   - Platform: local
   - Tools: `codex-artifact-pack.sh`, `export_docx.sh`
   - Paper impact: medium

4. **Optional next empirical extension**
   - Candidate: a low-data shortlist pilot before any second dataset or extra V7 seed
   - Platform: M4 first for cheap pilots; Colab is acceptable for a selected paper-impact experiment using likely-expiring surplus units
   - Paper impact: medium to high
   - Stop condition: skip if the reviewer-risk reduction does not justify the implementation cost or Colab spend

## 3. Track B — Stronger Research Route

### Goal

Move from a fair benchmark paper toward a stronger methodology paper.

### Exact Task Order

1. Implement one stronger compact classical baseline
2. Run a low-data scaling study before considering any second dataset
3. Only if low-data results are weak or ambiguous, add a second dataset
4. Only after those results exist, reassess whether any narrow regime claim for quantum competitiveness is defendable

### Preferred Low-Data Shortlist

The default low-data shortlist should compare:

- `thesis_cnniiii`
- `thesis_hqnn2`
- `classical_conv`
- `non_trainable_quantum`

Fractions:

- 10%
- 25%
- 50%
- 100%

Execution rule:

- start with a pilot pass
- only expand to full multi-seed if the pilot suggests a meaningful regime difference

## 4. Compute Policy

- Keep Mac work for M4-feasible benchmark and documentation tasks.
- The V7 confirmatory reruns are already complete; treat them as the last default Colab training spend unless new evidence justifies more.
- Current remaining budget is approximately `245` computing units.
- Do not spend additional Colab units on work already completed under the publication benchmark protocol or on V7 folder hygiene.
- Prefer zero-CU work for artifact sync, reconciliation, manuscript tightening, and submission package cleanup.
- A `45-50` CU paper-impact experiment is acceptable before renewal if it addresses reviewer risk better than local-only work.
- Delay broader extension work until the rerun artifacts are reconciled and the manuscript is updated.

## 5. Current Stop Conditions

Stop expanding the trainable-quantum branch if:

- the April 2026 V7 reruns remain clearly below the strongest classical anchors, and
- the paper contribution is already defensible as a benchmark + engineering paper without more V7 seeds
- or the expected information gain does not justify spending Colab units beyond the surplus likely to be lost at renewal

Stop expanding the publication benchmark if:

- the next added baseline does not materially change reviewer resilience, or
- the new result does not improve the paper narrative enough to justify the engineering time

## 6. Current Default Conclusion

Until new evidence changes it, the repo should operate under this conclusion:

- strongest thesis-faithful evidence: `thesis_cnniiii`
- strongest current-local matched-budget evidence: `classical_conv`
- strongest modern classical upper bound: `resnet18_cifar_gray`
- strongest current trainable-quantum value: engineering insight, not benchmark leadership
