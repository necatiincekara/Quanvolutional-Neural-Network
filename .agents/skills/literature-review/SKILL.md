---
name: literature-review
description: Conduct current-date, primary-source literature reviews for this Ottoman QML OCR study before related-work edits, V8 or Colab roadmap changes, or publication-positioning claims.
---

# Literature Review

Use this skill when the task needs current literature context for quantum machine learning, quanvolutional/QCNN architectures, Ottoman OCR/HTR, publication positioning, or V8/new Colab decisions.

## Workflow

1. Reconcile artifact truth before reading or writing claims:
   - `experiments/*.json`
   - `experiments/low_data_summary.json`
   - `docs/EXPERIMENTS.md`
   - `docs/BENCHMARK_SUMMARY.md`
   - `paper/draft.md`
2. Run a current web scan when the topic may have changed since the last repo update. Prefer primary sources:
   - arXiv pages
   - DOI landing pages
   - publisher pages
   - institutional repository records
3. Exclude weak sources from scientific claims:
   - blogs may be used only for discovery context
   - Reddit is not a source for this study
   - stale internal docs are not primary evidence
4. Classify every source into one of these buckets:
   - `directly relevant`: changes how the paper should frame benchmarks, OCR context, or QML architecture
   - `background`: useful citation context, but not decisive for this repo
   - `do-not-overclaim`: source is often cited for a tempting claim that this repo cannot support
   - `actionable for this repo`: source implies a concrete experiment, wording change, or stop/go gate
5. For each source, record:
   - citation and URL or DOI
   - what it supports
   - what it does not support
   - repo action
6. Before recommending V8, a new quantum training run, or Colab spend, require:
   - current artifact-backed baseline hierarchy
   - a matched classical comparator
   - separated model families
   - expected paper impact
   - platform and cost estimate
   - stop condition

## Claim Safety

- Do not claim generic quantum advantage.
- Treat V7 as a trainable-quantum engineering case-study unless new artifacts prove otherwise.
- Treat the low-data result as a scoped current-local signal for `non_trainable_quantum` versus `classical_conv`, not as evidence that every quantum model helps.
- Do not conflate `thesis_hqnn2`, `non_trainable_quantum`, and V7.
- Do not use modern-classical upper-bound results as thesis-faithful comparisons.

## Output Standard

A good literature-review output includes:

- `directly relevant` source table
- `background` source table
- `do-not-overclaim` notes
- `actionable for this repo` decisions
- publication wording implications
- V8 or Colab stop/go recommendation when applicable
