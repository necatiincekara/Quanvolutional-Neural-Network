# Submission Readiness Checklist

**Last reconciled:** August 9, 2026<br>
**Verdict:** **Minor work required; advisor-review ready, not yet external-submission ready.**

The repository-side evidence, Drive reconciliation, class-aware metrics, and public-data provenance are ready for advisor review. External submission still requires first-author metadata, both authors' declaration/CRediT approval, one venue/template decision, and approval of the final submission file.

## Evidence Gates

| Gate | Status | Evidence / limitation |
|---|---|---|
| Three-seed full-data benchmark | Ready | `experiments/benchmark_summary.json` |
| Full-data class-aware metrics | Ready | 21 checkpoints; `experiments/classification_metrics_20260728.json` |
| Low-data six-seed accuracy analysis | Qualified ready | Byte-original rows; paired analysis; one-image parser drift |
| Low-data class-aware metrics | Qualified ready | 48 checkpoints; `experiments/low_data_classification_metrics_20260809.json`; exploratory only |
| Drive low-data provenance | Ready | Complete 56 JSON / 56 checkpoint grid; `experiments/drive_artifact_reconciliation_20260809.json` |
| Trainable V7 evidence | Qualified ready | Engineering case-study only; single runs and legacy protocol limitations |
| Paired/multiplicity-aware statistics | Ready | `experiments/statistical_evidence_2026-05-17.json` |
| Artifact byte manifest | Ready | `experiments/submission_artifact_manifest_20260728.json` |
| Exact local environment | Ready | `requirements-publication-lock.txt` |
| Exact April 2026 Colab environment | Not recoverable | Disclosed limitation; no rerun required for provenance hygiene |
| Dataset source/license/identity | Ready | Public Kaggle source, license label “GPL 2,” 3,894/3,894 local images content-identical |

## Claim Safety Gates

| Claim | Status | Safe wording |
|---|---|---|
| Generic quantum advantage | Not supported | Do not claim it. |
| Full-data hierarchy | Supported | Strongest reproduced evidence favors classical baselines. |
| Class-aware hierarchy | Supported | ResNet-18 and `thesis_cnniiii` remain strongest; fixed quantum has a descriptive current-local macro-F1 pattern. |
| Low-data signal | Exploratory only | Quantum mean is higher at all fractions, but all paired CIs cross zero and Holm-adjusted p-values are 1.0. |
| V7 component causality | Not supported | The bundled V7 package is associated with restored trainability; no component attribution. |
| AMP mechanism | Partially supported | Float16 boundary and non-finite failure were observed; low-level adjoint mechanism was not isolated. |
| Hypothesis revision from thesis | Supported if chronology stays explicit | The 2024 hypothesis was narrowed by later stronger controls; low-data is hypothesis-generating. |

## Human And Venue Gates

| Gate | Status | Required action |
|---|---|---|
| Authors/order | Provisionally known | Necati Incekara, Erdem Bilgili; both authors must approve order and final manuscript |
| Erdem Bilgili affiliation | Ready for confirmation | Faculty of Engineering, Piri Reis University, Istanbul, Türkiye |
| First-author metadata | **Open** | Affiliation, e-mail, ORCID and corresponding-author choice |
| Funding | Ready | “This research received no external funding.” |
| Competing interests | **Open** | Both authors confirm “none” or provide exact disclosure |
| CRediT contributions | **Open** | Both authors approve the proposed role allocation |
| Ethics wording | Qualified open | Secondary analysis of public isolated-character data; adapt to chosen venue's form |
| Venue/template | **Open** | Author selects one venue; advisor need not make this operational choice |
| All-author approval | **Open** | Obtain written approval of the final PDF/source before submission |
| Versioned public release | Submission-stage | Create immutable release/archive only after the paper and share scope are final |

## Scientific Decision

- Do not delay advisor review for a new quantum architecture or a V7 hyperparameter sweep.
- The optional low-cost addition is the existing `param_linear` control on the six-seed low-data grid. It is scientifically useful but not a current submission blocker.
- If a target venue explicitly demands trainable-quantum novelty, use a pre-specified V7-Lite design with a matched classical replacement; do not perform open-ended search.

## Final Verification Commands

```bash
git diff --check
venv/bin/python -m py_compile \
  src/dataset.py src/enhanced_training.py src/trainable_quantum_model.py train_v7.py \
  scripts/aggregate_low_data.py scripts/statistical_evidence.py \
  scripts/evaluate_classification_metrics.py \
  scripts/evaluate_low_data_classification_metrics.py \
  scripts/reconcile_drive_artifacts.py scripts/build_submission_manifest.py
venv/bin/python scripts/reconcile_drive_artifacts.py
venv/bin/python scripts/aggregate_benchmarks.py
venv/bin/python scripts/aggregate_low_data.py
venv/bin/python scripts/statistical_evidence.py
venv/bin/python scripts/evaluate_classification_metrics.py
venv/bin/python scripts/evaluate_low_data_classification_metrics.py
venv/bin/python scripts/build_submission_manifest.py
./scripts/export_docx.sh paper/draft.md
```

Expected outcome:

- full-data aggregate remains unchanged;
- low-data aggregate contains six current-local seeds and reports train-size ranges;
- every paired low-data confidence interval crosses zero;
- all 21 full-data and 48 current-local low-data checkpoint inferences reproduce source top-1 rows;
- manifests record canonical evidence and the dataset snapshot;
- exported Word documents contain no clipped figures or tables.

## Submission-Day Stop Conditions

Do not submit if any of the following remains true:

- first-author affiliation/contact or corresponding author is unresolved;
- any author has not approved order, CRediT roles, COI declaration, and the final manuscript;
- the chosen venue's current template and declarations are not applied;
- the manuscript says “quantum advantage,” “confirmed low-data advantage,” or claims a causal V7 component effect;
- reconstructed V7 JSON is described as byte-original;
- the same full manuscript is under review at another archival venue.
