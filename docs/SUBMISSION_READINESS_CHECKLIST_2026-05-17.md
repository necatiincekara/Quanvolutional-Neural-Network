# Submission Readiness Checklist

**Date:** May 17, 2026

This checklist translates the current artifact truth into submission-readiness gates. It is intentionally conservative: passing a gate means the claim is supported by repository artifacts, not that every possible reviewer objection is eliminated.

## Current Verdict

The project is ready for a specialized QML / applied OCR / cultural-heritage computing submission path after final venue formatting. It is not yet ready for a broad-scope Q1-style claim unless a second dataset, robustness axis, or stronger statistical design is added.

## Ready Gates

| Gate | Status | Evidence |
|---|---|---|
| Full-data benchmark artifacts exist | Ready | `experiments/benchmark_summary.json`, `docs/BENCHMARK_SUMMARY.md` |
| Low-data current-local confirmation exists | Ready | `experiments/low_data_summary.json`, `docs/LOW_DATA_SUMMARY.md`, `experiments/low_data_drive_manifest_20260502.json` |
| Statistical caution layer exists | Ready | `docs/STATISTICAL_EVIDENCE_2026-05-17.md`, `experiments/statistical_evidence_2026-05-17.json` |
| Literature contribution framing exists | Ready | `docs/LITERATURE_CONTRIBUTION_2026-05-17.md`, `paper/draft.md` Section 5.6 |
| Trainable V7 is framed as engineering evidence | Ready | `experiments/v7_trainable_quantum_rerun_20260406_l4.json`, `experiments/v7_trainable_quantum_clean_20260427_l4.json`, `docs/COLAB_V7_HANDOFF_2026-04-06.md` |
| Shareable Word exports exist | Ready | `paper/draft.docx`, submission/advisor/docx files under `docs/` |

## Claim Safety Gates

| Claim | Status | Safe Wording |
|---|---|---|
| Generic quantum advantage | Not supported | Do not claim it. |
| Full-data benchmark hierarchy | Supported | Strongest reproduced full-data evidence favors classical baselines. |
| Thesis-faithful comparison | Supported | `thesis_cnniiii` is the strongest thesis-faithful reproduction; `thesis_hqnn2` remains below it. |
| Current-local full-data comparison | Supported with nuance | `classical_conv` and `param_linear` slightly exceed `non_trainable_quantum`, but current-local full-data differences are small under `n=3`. |
| Low-data quantum signal | Supported with scope limits | `non_trainable_quantum` shows a narrow current-local low-data competitiveness signal against `classical_conv`. |
| V7 trainable quantum | Supported as engineering case-study | V7 can train after stabilization, but it is not an accuracy leader. |

## Submission Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Single dataset | Medium for specialized venues; high for broad venues | Position as benchmark/engineering study; add second dataset only for stronger venue route. |
| Small `n=3` statistics | Medium | Use descriptive confidence intervals and exploratory tests; avoid definitive p-value language. |
| Low-data seed43/44 raw JSONs are Drive-backed only | Low to medium | Keep manifest explicit; sync raw JSONs locally if final archival reproducibility is required. |
| April 27 V7 Drive `experiments/` subfolder is empty | Low | Keep JSON row marked as reconstructed from captured notebook output; do not rerun for folder hygiene. |
| Dataset access / licensing in final paper | Medium | Add dataset access instructions or archive link before external submission. |
| Venue scope mismatch | Medium | Choose specialized QML, applied ML, or cultural-heritage route; re-check current journal scope and quartile immediately before submission. |

## Final Pre-Submission Commands

Run these immediately before creating a submission archive:

```bash
git status --short --branch
git diff --check
venv/bin/python -m py_compile \
  scripts/aggregate_benchmarks.py \
  scripts/aggregate_low_data.py \
  scripts/statistical_evidence.py \
  scripts/plot_low_data_results.py
venv/bin/python scripts/aggregate_benchmarks.py \
  --json-out /tmp/qnn_benchmark_summary.json \
  --md-out /tmp/qnn_benchmark_summary.md
venv/bin/python scripts/statistical_evidence.py \
  --json-out /tmp/qnn_statistical_evidence.json \
  --md-out /tmp/qnn_statistical_evidence.md
./scripts/export_docx.sh paper/draft.md
```

Expected outcome:

- benchmark aggregate matches `experiments/benchmark_summary.json` and `docs/BENCHMARK_SUMMARY.md`;
- statistical evidence output matches `experiments/statistical_evidence_2026-05-17.json` and `docs/STATISTICAL_EVIDENCE_2026-05-17.md`;
- `paper/draft.docx` embeds `paper/figures/low_data_scaling.png`;
- working tree contains only intentional formatting/export changes.

## Do Not Do By Default

- Do not launch new V7 training.
- Do not rerun low-data rows only for artifact hygiene.
- Do not spend Colab CU until the target venue or reviewer risk justifies a specific additional experiment.
- Do not merge thesis-faithful, current-local, modern-classical, low-data, and V7 rows into one flat leaderboard.

## Next Human Decision

Choose the submission route:

1. **Specialized QML route:** strongest fit for benchmark discipline, negative-result honesty, and hybrid-QML engineering lessons.
2. **Applied OCR / cultural heritage route:** strongest fit if the Ottoman script recognition and dataset contribution are foregrounded.
3. **Broad Q1-style route:** requires more evidence, most likely a second dataset or robustness axis plus stronger statistical design.
