# Publication Strategy

**Date:** March 22, 2026

This document records the realistic publication status of the study and the strategy required to turn it into a publishable paper.

Operational benchmark tracking now lives alongside this strategy in:

- `docs/BENCHMARK_MATRIX_2026-03-22.md`
- `docs/BENCHMARK_SUMMARY.md`

## 1. Short Verdict

As the repository stands today, this work is **not yet strong enough for a broad-scope Q1 AI or computer vision journal**.

It is still **worth publishing**, but not in its current narrative form. The publishable value is in the benchmark discipline, negative-result honesty, hybrid-QML engineering lessons, and thesis-to-reproducible-study evolution, not in an unsupported raw "quantum wins" story.

It is **potentially publishable** in one of two ways:

1. as a **specialized QML paper** centered on rigorous failure analysis, fair benchmarking, and engineering lessons for hybrid quantum-classical training, or
2. as an **applied OCR / cultural-heritage computing paper** if the Ottoman character recognition contribution is strengthened and the quantum claim is made more modest and honest.

The current repository evidence does **not** support a strong "quantum advantage" claim.

## 2. Why The Current Paper Is Not Q1-Ready

### 2.1 The strongest current result is not quantum

Current repository evidence after the unified `publication_v1` benchmark pass and the fresh April 2026 Colab V7 rerun:

| Model | Test Accuracy | Status |
|---|---:|---|
| `thesis_cnniiii` | **85.26 ± 0.97** | strongest thesis-faithful reproduction |
| `resnet18_cifar_gray` | **88.13 ± 0.82** | stronger modern classical upper bound on the same fixed split |
| `classical_conv` | **81.40 ± 1.06** | strongest current-local matched-budget model |
| `param_linear` | 81.12 ± 2.27 | strong matched classical replacement |
| `non_trainable_quantum` | 80.40 ± 0.69 | current-local Henderson-style non-trainable quantum |
| `thesis_cnn3` | 79.33 ± 1.26 | thesis-faithful classical pairwise reference |
| `thesis_hqnn2` | 78.61 ± 0.69 | best thesis-faithful quantum reproduction |
| `V7 trainable quantum rerun` | 72.53% | fresh Colab L4 rerun, still below strongest classical anchors |
| `V7 trainable quantum (older documented)` | 65.02% | historical documented stabilized result |

This means the present paper cannot credibly argue that the trainable or non-trainable quantum path outperforms matched classical baselines.

### 2.2 The current draft over-focuses on V7 versus an old weak baseline

The paper draft currently frames V7 primarily against V4 and historical failures. That is scientifically incomplete because newer local ablations are stronger than both V4 and V7 on test accuracy.

### 2.3 The benchmark scope is too narrow for a broad Q1 claim

The study currently relies on a single small task:

- 44-class Ottoman-Turkish handwritten character recognition
- 3,894 total samples
- one main dataset split

That is interesting and publishable in a niche or domain-aware venue, but it is not enough by itself for a strong general claim about quanvolutional models.

### 2.4 Statistical rigor is still below top-tier expectations

Still missing or incomplete elements include:

- confidence intervals or significance tests
- a clearly defined evaluation framework for any "practical quantum advantage" claim
- the actual synced Colab artifact files for the fresh V7 rerun inside the local repo workspace
- a broader validation axis such as low-data scaling or a second dataset

### 2.5 There is no hardware or computational advantage argument yet

The experiments are simulator-centric. That is acceptable for research, but it weakens any claim that the quantum component is practically useful unless the paper demonstrates one of the following:

- better accuracy in a well-defined regime
- better parameter efficiency
- better low-data behavior
- better robustness
- a concrete resource tradeoff that remains favorable

At the moment, that case is not established.

## 3. What Could Still Be Publishable

### Route A: QML Insight Paper

This is the strongest scientific route if the paper is reframed honestly.

Core angle:

- hybrid QML failure analysis on a real 44-class task
- fair classical-vs-quantum benchmarking
- documented engineering lessons:
  - information bottleneck threshold
  - gradient collapse
  - AMP / float16 incompatibility at the quantum boundary
  - trainable vs non-trainable vs matched classical replacements

This route does **not** require proving quantum superiority. It requires proving that the study contributes trustworthy evidence and useful design rules for QML practitioners.

### Route B: Applied OCR / Cultural Heritage Paper

This route is more application-driven.

Core angle:

- Ottoman handwritten character recognition benchmark
- thesis-to-research evolution with stronger baselines and reproducible experiments
- quantum methods as a tested hypothesis rather than a guaranteed improvement

For this route to work, the paper must emphasize:

- the OCR task and dataset value
- the benchmark contribution
- what quantum methods do and do not help with

## 4. Realistic Venue Logic

Do **not** treat venue choice as fixed before the evidence is fixed.

### Broad Q1 AI / ML application journals

These typically expect strong novelty, convincing benchmarking, and clear practical value. For example:

- Expert Systems with Applications emphasizes original papers on the design, development, testing, implementation, and management of intelligent systems, and explicitly discourages superficial novelty claims or renaming existing ideas.
- Machine Learning with Applications emphasizes novelty, technical soundness, practical value, and effectiveness for important problems.

For this study, a broad Q1 application venue becomes realistic only if the benchmark and evaluation package are substantially strengthened.

### Specialized QML venues

A specialized QML venue may be more realistic if the paper is reframed around trustworthy evidence rather than raw accuracy claims.

Notably, Quantum Machine Intelligence has published work that explicitly states it is **not** making strong performance claims against classical ML and instead argues for limited, well-defined investigations that provide useful qualitative insights. That is much closer to the current strength of this repository.

### Cultural heritage computing venues

The cultural-heritage route is plausible if the contribution is framed as ICT for heritage study and preservation rather than as a frontier-QML victory claim. ACM's Journal on Computing and Cultural Heritage describes its scope broadly as ICT in support of cultural heritage management, presentation, and study.

### Quartile caution

Journal quartiles can vary by indexing service and subject category. Re-check Scopus / SCImago / JCR status immediately before submission.

## 5. Recommended Publication Target By Current Evidence

### Current best target: Q2 or specialized QML route

If we submit soon with only moderate additional work, the most realistic route is:

- a specialized QML journal or
- an applied ML / cultural-heritage venue where the main value is the benchmark and the honest comparison

### Possible Q1 route

A Q1 route is still possible, but only after the paper changes from:

- "we built a quantum OCR model"

to:

- "we provide a rigorous and reproducible benchmark of trainable and non-trainable quanvolution on a difficult 44-class handwriting problem, identify when and why the quantum component fails, and define the narrow regimes where it remains competitive or useful"

That requires more evidence than is currently present.

## 6. Concrete Gap List Before Submission

The following are the highest-priority missing pieces.

### 6.1 Synchronize the scientific claim

The paper, README, thesis comparisons, and experiment logs must all reflect the same current truth:

- the strongest thesis-faithful reproduced model is classical (`thesis_cnniiii = 85.26 ± 0.97`)
- the strongest current-local matched-budget model is classical (`classical_conv = 81.40 ± 1.06`)
- current-local Henderson-style non-trainable quantum is `80.40 ± 0.69`
- thesis-faithful HQNN-II reproduction is `78.61 ± 0.69`
- V7 trainable quantum is not the best current model

### 6.2 Reproduce the thesis-best quantum baseline faithfully

Completed at the repository level: a faithful reproduction path for thesis HQNN-II now exists and has been evaluated as a separate model.

Why this matters:

- the current Henderson-style non-trainable quantum ablation is **not** the same architecture as thesis HQNN-II
- current evidence shows that the thesis-faithful HQNN-II reproduction remains below the strongest classical thesis-faithful anchor

### 6.3 Add stronger classical baselines

Completed addition:

- one modern compact CNN baseline now exists: `resnet18_cifar_gray = 88.13 ± 0.82` test over three seeds

Still optional:

- one transfer-learning or stronger vision baseline if reviewer pressure justifies it
- matched-parameter comparisons where relevant

The goal is not to make quantum look better. The goal is to make reviewer objections weaker.

### 6.4 Run multi-seed experiments

Minimum recommendation:

- 3 seeds for all key models
- report mean ± std
- keep splits and augmentation logic consistent

### 6.5 Define the quantum claim narrowly

The paper should stop implicitly chasing generic "quantum advantage".

Choose **one** defensible claim:

1. **Engineering claim:** gradient stabilization and precision-boundary lessons make trainable quanvolution reproducible.
2. **Benchmark claim:** on a hard low-data heritage OCR task, quantum and classical variants can be compared on equal footing and the current evidence favors classical baselines.
3. **Regime claim:** quantum may remain competitive only in a specific regime such as low-data or parameter-efficiency, if future experiments support it.

Do not claim all three unless the evidence truly supports all three.

### 6.6 Add one broader validation axis

At least one of the following should be added:

- a second dataset
- a low-data scaling study
- parameter-budget scaling
- robustness / noise / corruption evaluation

Without this, the paper remains too single-benchmark to be persuasive for stronger venues.

## 7. Practical Two-Track Strategy

### Track 1: Fast publishable paper

Goal: produce a credible Q2 / specialized-journal manuscript without waiting for a perfect quantum win.

Steps:

1. synchronize claims across paper and docs
2. use the now-completed 3-seed comparisons for:
   - thesis HQNN-II
   - thesis CNN-III
   - thesis CNN-IIII
   - current Henderson-style non-trainable quantum
   - classical_conv
   - param_linear
3. do not schedule another V7 seed by default; first sync the finished rerun artifacts and update the paper from that result
4. rewrite the paper around trustworthy comparative evidence

Operational note:
the trainable-quantum confirmation has now been obtained once on Colab L4, and the stronger modern classical upper bound has been completed locally. Remaining Colab budget is about `245` computing units, so further remote runs should still be treated as exceptional rather than routine.

This is the shortest realistic publication path.

### Track 2: Stronger Q1 attempt

Goal: upgrade the paper into a stronger benchmark / methodology contribution.

Additional steps:

1. add stronger classical baselines
2. add one external or second benchmark
3. add low-data or parameter-efficiency analysis
4. formalize a fair comparison framework inspired by practical quantum advantage literature
5. demonstrate at least one regime where the quantum model is genuinely competitive under rules defined in advance

This route is slower, but it is the only honest way toward a broad Q1 attempt.

## 8. Recommended Writing Position Today

If we start rewriting now, the paper should be positioned as:

> A rigorous comparative study of trainable and non-trainable quanvolutional models for Ottoman handwritten character recognition, with emphasis on failure analysis, benchmarking discipline, and reproducible engineering lessons for hybrid QML.

That is much stronger and more defensible than:

> A quantum model that outperforms classical OCR.

## 9. Immediate Action List

1. Sync the actual Colab `models/` and `experiments/v7_*` artifacts into the repo workspace.
2. Reconcile the fresh V7 rerun row against `docs/EXPERIMENTS.md`, `docs/BENCHMARK_SUMMARY.md`, and `paper/draft.md`.
3. Finalize the benchmark-first paper wording and submission packet from the now-complete benchmark hierarchy.
4. Delay any additional Colab training unless it clearly changes reviewer risk enough to justify the remaining `245` computing units.
5. Decide whether the paper will follow the QML-insight route or the OCR/heritage route.

## 10. Source Notes

This strategy was informed by:

- the current repository evidence
- the present paper draft and roadmap
- current journal scope pages
- recent QML literature emphasizing either fair classical comparison or scope-limited but rigorous hybrid-QML analysis

Relevant sources:

- Expert Systems with Applications aims and scope: `https://www.sciencedirect.com/journal/expert-systems-with-applications`
- Machine Learning with Applications aims and scope: `https://www.sciencedirect.com/journal/machine-learning-with-applications`
- SCImago category example showing broad AI Q1 venue context: `https://www.scimagojr.com/journalrank.php?category=1702&country=GB&ord=desc&order=cd`
- SCImago category example for Quantum Machine Intelligence: `https://www.scimagojr.com/journalrank.php?category=2614&country=DE`
- SCImago category example for Journal on Computing and Cultural Heritage: `https://www.scimagojr.com/diamond/journalrank.php?category=1206&country=OECD&type=j`
- Quantum Machine Intelligence example article discussing limited but well-defined QML investigations: `https://link.springer.com/article/10.1007/s42484-024-00200-0`
- Practical quantum advantage framework: `https://www.nature.com/articles/s42005-024-01552-6`
