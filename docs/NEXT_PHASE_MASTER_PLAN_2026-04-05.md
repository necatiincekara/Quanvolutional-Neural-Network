# Next Phase Master Plan

**Date:** April 5, 2026
**Last updated:** May 16, 2026

This document is the post-upgrade master plan for the study. It assumes the current benchmark truth remains unchanged and uses the upgraded Codex layer to reduce ambiguity in the next research phase.

> Update, April 30, 2026:
> The April 6 resumed Colab V7 rerun remains complete at `72.89%` best validation and `72.53%` test accuracy.
> The April 27 clean non-resumed Colab V7 rerun reached `69.97%` best validation and `65.88%` test; its JSON row is reconstructed from captured notebook output because the runtime disconnected before the artifact-copy cell copied JSON to Drive.
> The stronger modern classical baseline `resnet18_cifar_gray` has now completed three local seeds at `88.13 ± 0.82%` test accuracy.
> The user-reported pre-low-data Colab budget was approximately `245` computing units; the current post-confirmation balance has not been remeasured in this repo.
> Default policy is still conservative for V7 reruns, but spending surplus Colab units can be justified if the experiment is paper-impactful before the next renewal cap.

> Update, May 16, 2026:
> The low-data scaling confirmation is complete for the current-local pair across seeds `42`, `43`, and `44`.
> `non_trainable_quantum` exceeded `classical_conv` at every measured fraction (`10%`, `25%`, `50%`, and `100%`), with the strongest low-data margins at `25%` and `10%`.
> The thesis-faithful low-data check remains a seed-42 pilot and is still classical-favored: `thesis_cnniiii` stays ahead of `thesis_hqnn2`.
> Treat this as a narrow current-local low-data competitiveness signal, not a general quantum advantage claim.

## 1. Immediate Operating Sequence

These are the next exact actions, in order:

1. **Finish submission-facing document synchronization**
   - Goal: keep `paper/draft.md`, `.docx` exports, `docs/SUBMISSION_BENCHMARK_2026-03-25.md`, `docs/PUBLICATION_STRATEGY_2026-03-22.md`, and this roadmap aligned with the May 2026 low-data result.
   - Platform: local
   - Paper impact: high

2. **Keep V7 provenance labels stable**
   - Goal: preserve the distinction between the April 6 resumed V7 run, the April 27 clean non-resumed V7 run, and the older documented V7 row; do not spend compute on V7 folder hygiene.
   - Platform: local
   - Paper impact: high

3. **Prepare the final advisor/submission packet**
   - Goal: package Markdown, Word, figures, benchmark tables, and experiment provenance so the project can be reviewed without stale historical framing.
   - Platform: local
   - Paper impact: high

4. **Re-evaluate only paper-impactful empirical extensions**
   - Goal: consider second-dataset robustness, additional statistical intervals, or a narrowly scoped matched-budget control only if it materially changes reviewer resilience or venue fit.
   - Platform: local planning first; Colab only if the extension requires it
   - Paper impact: medium

## 2. Track A — Near-Term Publishable Route

### Goal

Finish a credible Q2 / specialized-QML submission without implying unsupported quantum superiority.

### Exact Task Order

1. **Submission-facing claim sync**
   - Platform: local
   - Goal: keep the paper, benchmark summaries, publication strategy, roadmap, and shareable exports aligned with the full-data hierarchy and May 2026 low-data result
   - Paper impact: high
   - Stop condition: complete once the submission packet no longer describes low-data as future work and no longer implies a generic quantum advantage claim

2. **Final paper polish**
   - Platform: local
   - Tools: `codex-paper-audit.sh`, `paper-sync`, `paper_writer`
   - Paper impact: high

3. **Submission-facing document refresh**
   - Platform: local
   - Tools: `codex-artifact-pack.sh`, `export_docx.sh`
   - Paper impact: medium

4. **Optional next empirical extension**
   - Candidate: second-dataset robustness, confidence intervals/significance tests, or one additional matched-budget control if it answers a concrete reviewer risk
   - Platform: M4 first for cheap pilots; Colab is acceptable for a selected paper-impact experiment using surplus units
   - Paper impact: medium to high
   - Stop condition: skip if the reviewer-risk reduction does not justify the implementation cost or Colab spend

## 3. Track B — Stronger Research Route

### Goal

Move from a fair benchmark paper toward a stronger methodology paper.

### Exact Task Order

1. Keep the completed low-data axis scoped to the current-local model pair unless reviewers ask for a broader low-data grid
2. Add confidence intervals or simple significance tests for the full-data and low-data tables if the venue expects stronger statistical reporting
3. Add a second dataset or robustness split only if the goal shifts from a fair benchmark paper to a stronger methodology paper
4. Only after those results exist, reassess whether any broader regime claim for quantum competitiveness is defendable

### Completed Low-Data Axis

The completed current-local confirmation compared:

- `classical_conv`
- `non_trainable_quantum`

Fractions:

- 10%
- 25%
- 50%
- 100%

Result summary:

- `non_trainable_quantum` outperformed `classical_conv` at all measured fractions, with mean test accuracies of `50.71%`, `69.88%`, `76.75%`, and `80.76%`.
- `classical_conv` mean test accuracies were `48.42%`, `66.24%`, `75.61%`, and `80.47%`.
- The thesis-faithful low-data check remains a seed-42 pilot, where `thesis_cnniiii` stayed ahead of `thesis_hqnn2`.

Execution rule going forward:

- do not expand low-data by default
- add extra low-data seeds only if a reviewer asks whether the thesis-faithful pair shows the same pattern
- prioritize submission clarity over new training

## 4. Compute Policy

- Keep Mac work for M4-feasible benchmark and documentation tasks.
- The V7 confirmatory reruns are already complete; treat them as the last default Colab training spend unless new evidence justifies more.
- The last user-reported pre-low-data budget was approximately `245` computing units; remeasure before any new Colab run.
- Do not spend additional Colab units on work already completed under the publication benchmark protocol, low-data confirmation, or V7 folder hygiene.
- Prefer zero-CU work for artifact sync, reconciliation, manuscript tightening, and submission package cleanup.
- A paper-impact Colab experiment is acceptable before renewal if it addresses reviewer risk better than local-only work.
- Delay broader extension work until the submission package is coherent and the exact question for extra compute is explicit.

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
- specific current-local low-data signal: `non_trainable_quantum` is competitive with and modestly ahead of `classical_conv` across the measured low-data fractions
- strongest modern classical upper bound: `resnet18_cifar_gray`
- strongest current trainable-quantum value: engineering insight, not benchmark leadership
