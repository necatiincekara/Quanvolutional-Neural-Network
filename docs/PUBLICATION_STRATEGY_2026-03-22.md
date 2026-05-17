# Publication Strategy

**Date:** March 22, 2026
**Last updated:** May 16, 2026

This document records the realistic publication status of the study and the strategy required to turn it into a publishable paper.

Operational benchmark tracking now lives alongside this strategy in:

- `docs/BENCHMARK_MATRIX_2026-03-22.md`
- `docs/BENCHMARK_SUMMARY.md`

## 1. Short Verdict

As the repository stands today, this work is **not yet strong enough for a broad-scope Q1 AI or computer vision journal**.

It is still **worth publishing**, and the current `paper/draft.md` has now been refocused around benchmark discipline, negative-result honesty, a narrow low-data competitiveness signal, hybrid-QML engineering lessons, and thesis-to-reproducible-study evolution. The paper should not be positioned as an unsupported raw "quantum wins" story.

It is **potentially publishable** in one of two ways:

1. as a **specialized QML paper** centered on rigorous failure analysis, fair benchmarking, and engineering lessons for hybrid quantum-classical training, or
2. as an **applied OCR / cultural-heritage computing paper** if the Ottoman character recognition contribution is strengthened and the quantum claim is made more modest and honest.

The current repository evidence does **not** support a strong "quantum advantage" claim.

## 2. Why The Current Paper Is Not Q1-Ready

### 2.1 The strongest current result is not quantum

Current repository evidence after the unified `publication_v1` benchmark pass, the April 2026 Colab V7 reruns, and the May 2026 low-data confirmation:

| Model | Test Accuracy | Status |
|---|---:|---|
| `resnet18_cifar_gray` | **88.13 ± 0.82** | stronger modern classical upper bound on the same fixed split |
| `thesis_cnniiii` | **85.26 ± 0.97** | strongest thesis-faithful reproduction |
| `classical_conv` | **81.40 ± 1.06** | strongest current-local matched-budget model |
| `param_linear` | 81.12 ± 2.27 | strong matched classical replacement |
| `non_trainable_quantum` | 80.40 ± 0.69 | current-local Henderson-style non-trainable quantum |
| `thesis_cnn3` | 79.33 ± 1.26 | thesis-faithful classical pairwise reference |
| `thesis_hqnn2` | 78.61 ± 0.69 | best thesis-faithful quantum reproduction |
| `V7 trainable quantum rerun` | 72.53% | April 6 resumed Colab L4 rerun, still below strongest classical anchors |
| `V7 trainable quantum clean rerun` | 65.88% | April 27-28 clean non-resumed Colab L4 run reconstructed from captured notebook output after runtime disconnect |
| `V7 trainable quantum (older documented)` | 65.02% | historical documented stabilized result |

This means the present paper cannot credibly argue that the trainable or non-trainable quantum path outperforms matched classical baselines.

Low-data scaling adds a narrower qualification:

| Family | Low-data result | Status |
|---|---|---|
| current-local | `non_trainable_quantum` exceeds `classical_conv` on three-seed mean test accuracy at 10/25/50/100% train fractions | specific competitiveness signal |
| thesis-faithful | `thesis_cnniiii` remains ahead of `thesis_hqnn2` in the seed-42 pilot at all train fractions | classical-favored |

### 2.2 Draft framing status

Earlier versions of the draft over-focused on V7 versus V4 and historical failures. The current `paper/draft.md` has been refocused around RQ1/RQ2/RQ3: full-data benchmark hierarchy, low-data scaling, and trainable-quantum engineering constraints. The remaining writing risk is now submission polish and venue fit, not a stale V7-winner narrative.

### 2.3 The benchmark scope is too narrow for a broad Q1 claim

The study currently relies on a single small task:

- 44-class Ottoman-Turkish handwritten character recognition
- 3,894 total samples
- one main dataset split

That is interesting and publishable in a niche or domain-aware venue, but it is not enough by itself for a strong general claim about quanvolutional models.

### 2.4 Statistical rigor is still below top-tier expectations

Still missing or incomplete elements include:

- formal inferential power beyond the May 17, 2026 confidence-interval and exploratory Welch report; most multi-seed groups still have only `n=3`
- a clearly defined evaluation framework for any "practical quantum advantage" claim
- directly copied Colab JSON/experiment metadata for the April 2026 V7 reruns; the repository has reconstructed JSON rows and Drive-backed checkpoints, but the April 27 Drive `experiments/` subfolder is empty
- a second dataset or robustness axis; the May 2026 low-data scaling axis is now confirmed for the current-local pair and remains thesis-faithful seed-42-only

### 2.5 There is no hardware or computational advantage argument yet

The experiments are simulator-centric. That is acceptable for research, but it weakens any claim that the quantum component is practically useful unless the paper demonstrates one of the following:

- better accuracy in a well-defined regime
- better parameter efficiency
- better low-data behavior beyond the narrow May 2026 current-local signal
- better robustness
- a concrete resource tradeoff that remains favorable

At the moment, only a narrow current-local low-data competitiveness signal is established. A broad practical-usefulness or hardware-advantage case is not established.

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

The following are the highest-priority remaining pieces.

### 6.1 Synchronize the scientific claim

The paper, README, thesis comparisons, and experiment logs should continue to reflect the same current truth:

- the strongest thesis-faithful reproduced model is classical (`thesis_cnniiii = 85.26 ± 0.97`)
- the strongest current-local matched-budget model is classical (`classical_conv = 81.40 ± 1.06`)
- current-local Henderson-style non-trainable quantum is `80.40 ± 0.69`
- thesis-faithful HQNN-II reproduction is `78.61 ± 0.69`
- V7 trainable quantum is not the best current model
- current-local low-data confirmation supports a specific `non_trainable_quantum` competitiveness signal against `classical_conv`, not a generic quantum-advantage claim

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

Completed for the core full-data benchmark:

- 3 seeds for all key models
- report mean ± std
- keep splits and augmentation logic consistent

Completed for the current-local low-data confirmation:

- seeds 42, 43, and 44 for `classical_conv` and `non_trainable_quantum`
- fixed split seed 42 and fraction seed 42

Still optional:

- stronger statistical reporting if the target venue requires more than the May 17, 2026 descriptive confidence-interval and exploratory Welch report
- extra thesis-faithful low-data seeds only if reviewer feedback specifically asks for that axis

### 6.5 Define the quantum claim narrowly

The paper should stop implicitly chasing generic "quantum advantage".

Choose **one** defensible claim:

1. **Engineering claim:** gradient stabilization and precision-boundary lessons make trainable quanvolution reproducible.
2. **Benchmark claim:** on a hard low-data heritage OCR task, quantum and classical variants can be compared on equal footing and the current evidence favors classical baselines.
3. **Regime claim:** quantum can be narrowly competitive in a specific current-local low-data regime, while thesis-faithful and strongest full-data comparisons remain classical-favored.

Do not claim all three unless the evidence truly supports all three.

### 6.6 Add one broader validation axis

At least one of the following should be added:

- a second dataset
- a second-dataset, robustness, or parameter-budget extension to the May 2026 low-data scaling study
- parameter-budget scaling
- robustness / noise / corruption evaluation

Without this, the paper remains too single-benchmark to be persuasive for stronger broad-scope venues, although it is still viable as a specialized benchmark/engineering paper.

## 7. Practical Two-Track Strategy

### Track 1: Fast publishable paper

Goal: produce a credible Q2 / specialized-journal manuscript without waiting for a perfect quantum win.

Steps:

1. keep claims synchronized across paper and docs after each result-language edit
2. use the now-completed 3-seed comparisons for:
   - thesis HQNN-II
   - thesis CNN-III
   - thesis CNN-IIII
   - current Henderson-style non-trainable quantum
   - classical_conv
   - param_linear
3. use the May 2026 low-data confirmation only as a narrow current-local regime claim
4. do not schedule another V7 seed by default; keep the finished rerun artifacts and reconstructed JSON rows clearly labeled
5. keep `paper/draft.md` and `paper/draft.docx` refreshed from the current Markdown

Operational note:
the trainable-quantum confirmations have been obtained on Colab L4, the stronger modern classical upper bound has been completed locally, and the current-local low-data confirmation has been completed. The last user-reported Colab budget before this low-data confirmation was about `245` computing units; future remote spending should address reviewer risk or a stronger venue target, not V7 artifact hygiene or already-completed low-data rows.

This is the shortest realistic publication path.

### Track 2: Stronger Q1 attempt

Goal: upgrade the paper into a stronger benchmark / methodology contribution.

Additional steps:

1. add stronger classical baselines
2. add one external or second benchmark
3. extend the low-data result to a second dataset, robustness axis, or parameter-efficiency analysis
4. formalize a fair comparison framework inspired by practical quantum advantage literature
5. demonstrate at least one regime where the quantum model is genuinely competitive under rules defined in advance

This route is slower, but it is the only honest way toward a broad Q1 attempt.

## 8. Recommended Writing Position Today

The current paper should be positioned as:

> A rigorous comparative study of trainable and non-trainable quanvolutional models for Ottoman handwritten character recognition, with emphasis on failure analysis, benchmarking discipline, and reproducible engineering lessons for hybrid QML.

That is much stronger and more defensible than:

> A quantum model that outperforms classical OCR.

## 9. Immediate Action List

1. Keep reconstructed April 2026 V7 rows clearly labeled, and sync copied Colab `experiments/v7_*` metadata only if it is still available.
2. Keep the May 2026 low-data claim scoped to the current-local `non_trainable_quantum` versus `classical_conv` comparison unless new artifacts expand it.
3. Maintain synchronized Markdown and Word exports for `paper/draft.md`, `docs/SUBMISSION_BENCHMARK_2026-03-25.md`, and this strategy document.
4. Decide whether the paper will follow the specialized QML-insight route or the OCR/heritage route.
5. Use Colab only for a paper-impactful extension, such as second-dataset/robustness transfer or reviewer-requested statistics.

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
