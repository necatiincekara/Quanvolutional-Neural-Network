# Submission Packet

**Date:** May 16, 2026

This manifest lists the current shareable paper/advisor package after the April 2026 V7 reruns, the May 2026 low-data confirmation, and the paper cleanup pass.

## Current Claim Set

- Full-data benchmark hierarchy remains classical-favored.
- `resnet18_cifar_gray` is the strongest modern-classical upper bound: `88.13 ± 0.82%` test.
- `thesis_cnniiii` is the strongest thesis-faithful reproduction: `85.26 ± 0.97%` test.
- `classical_conv` is the strongest current-local full-data matched-budget model: `81.40 ± 1.06%` test.
- `non_trainable_quantum` is not the full-data current-local leader, but the May 2026 low-data confirmation shows a narrow current-local competitiveness signal against `classical_conv` across 10/25/50/100% train fractions.
- V7 is a trainable-quantum engineering case-study, not a benchmark leader.
- Do not claim generic quantum advantage.

## Primary Files To Share

| Purpose | File |
|---|---|
| Paper draft | `paper/draft.md` |
| Word paper draft | `paper/draft.docx` |
| Low-data figure | `paper/figures/low_data_scaling.pdf` |
| Low-data figure preview | `paper/figures/low_data_scaling.png` |
| Statistical evidence report | `docs/STATISTICAL_EVIDENCE_2026-05-17.md` |
| Statistical evidence Word export | `docs/STATISTICAL_EVIDENCE_2026-05-17.docx` |
| Submission benchmark and claim set | `docs/SUBMISSION_BENCHMARK_2026-03-25.md` |
| Submission benchmark Word export | `docs/SUBMISSION_BENCHMARK_2026-03-25.docx` |
| Turkish advisor update | `docs/ADVISOR_UPDATE_2026-03-25_TR.md` |
| Turkish advisor update Word export | `docs/ADVISOR_UPDATE_2026-03-25_TR.docx` |
| Publication strategy | `docs/PUBLICATION_STRATEGY_2026-03-22.md` |
| Publication strategy Word export | `docs/PUBLICATION_STRATEGY_2026-03-22.docx` |
| Next-phase roadmap | `docs/NEXT_PHASE_MASTER_PLAN_2026-04-05.md` |
| Next-phase roadmap Word export | `docs/NEXT_PHASE_MASTER_PLAN_2026-04-05.docx` |

## Evidence Files

| Evidence | File |
|---|---|
| Full-data benchmark aggregate JSON | `experiments/benchmark_summary.json` |
| Full-data benchmark aggregate Markdown | `docs/BENCHMARK_SUMMARY.md` |
| Low-data aggregate JSON | `experiments/low_data_summary.json` |
| Low-data aggregate Markdown | `docs/LOW_DATA_SUMMARY.md` |
| Statistical evidence JSON | `experiments/statistical_evidence_2026-05-17.json` |
| Low-data Drive provenance manifest | `experiments/low_data_drive_manifest_20260502.json` |
| Experiment log | `docs/EXPERIMENTS.md` |
| V7 operational/provenance handoff | `docs/COLAB_V7_HANDOFF_2026-04-06.md` |

## Known Provenance Notes

- Current-local low-data seed-43 and seed-44 raw JSON rows are Drive-backed and recorded in `experiments/low_data_drive_manifest_20260502.json`; they are not tracked as local raw JSON files.
- `scripts/aggregate_low_data.py` now protects the canonical low-data summary from being overwritten by local seed-42 pilot rows while those confirmation JSONs remain remote-only.
- `scripts/statistical_evidence.py` reports 95% confidence intervals and exploratory Welch comparisons; because most multi-seed groups have only `n=3`, these p-values are descriptive reviewer aids rather than definitive inferential claims.
- The April 27 clean V7 Drive folder contains checkpoints, but its `experiments/` subfolder is empty because the Colab runtime disconnected before the artifact-copy cell ran.
- The last user-reported Colab budget before the May 2026 low-data confirmation was approximately `245` CU; remeasure before planning any new Colab run.

## Next Decision Points

1. Decide the submission route: specialized QML insight paper versus applied OCR / cultural heritage paper.
2. Add confidence intervals or formal significance tests only if the target venue/reviewer profile needs them.
3. Add a second dataset or robustness axis only for a stronger methodology route.
4. Do not rerun V7 or low-data rows for artifact hygiene.
