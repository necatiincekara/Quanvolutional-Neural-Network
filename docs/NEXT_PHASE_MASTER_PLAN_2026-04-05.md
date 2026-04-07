# Next Phase Master Plan

**Date:** April 5, 2026

This document is the post-upgrade master plan for the study. It assumes the current benchmark truth remains unchanged and uses the upgraded Codex layer to reduce ambiguity in the next research phase.

> Update, April 7, 2026:
> The fresh Colab V7 confirmatory rerun has now been completed successfully at `72.89%` best validation and `72.53%` test accuracy.
> Remaining Colab budget is approximately `145` computing units.
> Default policy is now low-compute mode: do not start another Colab training run unless it changes the paper decision boundary.

## 1. Immediate Operating Sequence

These are the next exact actions, in order:

1. **Sync the remaining fresh V7 rerun artifacts from Drive/Colab into the local repo**
   - Goal: complete the artifact picture by recovering the remote `experiments/v7_*` folder, since the Drive-backed checkpoint files are now already synced locally.
   - Platform: local + Drive
   - Paper impact: high

2. **Reconcile the synced artifacts against the local benchmark tables**
   - Goal: keep `experiments/*.json`, `docs/BENCHMARK_SUMMARY.md`, `docs/EXPERIMENTS.md`, and `paper/draft.md` aligned.
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
   - Goal: default to no new Colab runs unless a new experiment would materially change reviewer resilience or the paper decision boundary.
   - Platform: local planning only
   - Paper impact: medium

## 2. Track A — Near-Term Publishable Route

### Goal

Finish a credible Q2 / specialized-QML submission without implying unsupported quantum superiority.

### Exact Task Order

1. **Artifact sync and V7 result reconciliation**
   - Platform: local + existing Drive artifacts
   - Goal: pull the actual `models/` and `experiments/v7_*` artifacts from Colab/Drive into the repo and replace any user-log-only placeholder records
   - Paper impact: high
   - Stop condition: complete once the rerun is backed by local artifact files rather than only reconstructed terminal output

2. **Paper synchronization after the rerun**
   - Platform: local
   - Tools: `codex-paper-audit.sh`, `paper-sync`, `paper_writer`
   - Paper impact: high

3. **Submission-facing document refresh**
   - Platform: local
   - Tools: `codex-artifact-pack.sh`, `export_docx.sh`
   - Paper impact: medium

4. **Optional stronger compact classical baseline**
   - Candidate: `resnet18_cifar_gray`
   - Platform: M4 first if feasible, Colab only if absolutely necessary
   - Paper impact: medium to high
   - Stop condition: skip if the reviewer-risk reduction does not justify the implementation cost or the remaining `145` CU budget

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
- The V7 confirmatory rerun is already complete; treat it as the last default Colab training spend unless new evidence justifies more.
- Current remaining budget is approximately `145` computing units.
- Do not spend additional Colab units on work already completed under the publication benchmark protocol.
- Prefer zero-CU work now: artifact sync, reconciliation, manuscript tightening, and submission package cleanup.
- Delay broader extension work until the rerun artifacts are synced and the manuscript is updated.

## 5. Current Stop Conditions

Stop expanding the trainable-quantum branch if:

- the fresh V7 rerun remains clearly below the strongest classical anchors, and
- the paper contribution is already defensible as a benchmark + engineering paper without more V7 seeds
- or the expected information gain does not justify spending the remaining `145` CU budget

Stop expanding the publication benchmark if:

- the next added baseline does not materially change reviewer resilience, or
- the new result does not improve the paper narrative enough to justify the engineering time

## 6. Current Default Conclusion

Until new evidence changes it, the repo should operate under this conclusion:

- strongest thesis-faithful evidence: `thesis_cnniiii`
- strongest current-local matched-budget evidence: `classical_conv`
- strongest current trainable-quantum value: engineering insight, not benchmark leadership
