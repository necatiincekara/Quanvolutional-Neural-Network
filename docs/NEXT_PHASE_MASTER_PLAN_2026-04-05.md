# Next Phase Master Plan

**Date:** April 5, 2026

This document is the post-upgrade master plan for the study. It assumes the current benchmark truth remains unchanged and uses the upgraded Codex layer to reduce ambiguity in the next research phase.

## 1. Immediate Operating Sequence

These are the next exact actions, in order:

1. Run `./scripts/codex-capability-audit.sh`
   - Goal: freeze the official Codex capability picture against current repo usage.
   - Platform: local
   - Paper impact: medium

2. Run `./scripts/codex-gap-audit.sh`
   - Goal: verify workflow coverage and confirm no critical Codex gap remains.
   - Platform: local
   - Paper impact: medium

3. Run `./scripts/codex-model-benchmark.sh`
   - Goal: compare `gpt-5.4`, `gpt-5.3-codex`, and `gpt-5.4-mini` on reconciliation, paper audit, and benchmark triage tasks.
   - Platform: local
   - Paper impact: low direct, high workflow value
   - Stop condition: if `gpt-5.3-codex` is not clearly better on at least 2/3 tasks, keep current defaults

4. Run `./scripts/codex-benchmark-triage.sh`
   - Goal: confirm the next scientifically meaningful run under the current evidence.
   - Platform: local
   - Expected answer: Colab V7 confirmatory rerun

5. Run `./scripts/codex-colab-handoff.sh`
   - Goal: prepare the exact Colab execution package for the V7 run.
   - Platform: local
   - Output: command choice, environment checklist, sync steps, checkpoint policy

## 2. Track A — Near-Term Publishable Route

### Goal

Finish a credible Q2 / specialized-QML submission without implying unsupported quantum superiority.

### Exact Task Order

1. **Colab V7 confirmatory rerun**
   - Platform: Colab L4 preferred; A100 acceptable if needed
   - Entry point: `train_v7.py`
   - Default circuit: `data_reuploading`
   - Initial command target: `python train_v7.py --epochs 10 --circuit data_reuploading`
   - Paper impact: high
   - Stop conditions:
     - stop immediately on NaN recurrence
     - if validation remains far below the documented trajectory after early epochs, switch to debugging rather than adding more epochs
     - if the rerun confirms V7 remains below the strongest classical baselines, keep V7 as an engineering case-study and do not spend Colab budget on extra seeds yet

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
   - Platform: M4 first if feasible, Colab only if necessary
   - Paper impact: medium to high
   - Stop condition: skip if the reviewer-risk reduction does not justify the implementation and runtime cost

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
- Spend Colab compute units first on the V7 confirmatory rerun.
- Do not spend Colab units on work already completed under the publication benchmark protocol.
- Delay broader extension work until the V7 rerun and manuscript synchronization are complete.

## 5. Current Stop Conditions

Stop expanding the trainable-quantum branch if:

- the fresh V7 rerun remains clearly below the strongest classical anchors, and
- the paper contribution is already defensible as a benchmark + engineering paper without more V7 seeds

Stop expanding the publication benchmark if:

- the next added baseline does not materially change reviewer resilience, or
- the new result does not improve the paper narrative enough to justify the engineering time

## 6. Current Default Conclusion

Until new evidence changes it, the repo should operate under this conclusion:

- strongest thesis-faithful evidence: `thesis_cnniiii`
- strongest current-local matched-budget evidence: `classical_conv`
- strongest current trainable-quantum value: engineering insight, not benchmark leadership
