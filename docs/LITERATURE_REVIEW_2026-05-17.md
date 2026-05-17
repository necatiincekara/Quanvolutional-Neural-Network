# May 2026 Literature Review

**Date:** May 17, 2026  
**Scope:** QML benchmarking, quanvolutional/QCNN architecture directions, and Ottoman OCR/HTR positioning for the current hybrid quantum-classical OCR study.

## Artifact Grounding

This review uses the current repository evidence as the starting point:

- Full-data hierarchy remains classical-favored. `resnet18_cifar_gray` is the strongest modern-classical upper bound at `88.13 ± 0.82%` test accuracy.
- `thesis_cnniiii` is the strongest thesis-faithful reproduction at `85.26 ± 0.97%` test accuracy.
- In the current-local full-data family, `classical_conv` is `81.40 ± 1.06%`, while `non_trainable_quantum` is `80.40 ± 0.69%`.
- In the May 2026 low-data confirmation, `non_trainable_quantum` shows a narrow current-local competitiveness signal against `classical_conv` across 10%, 25%, 50%, and 100% training fractions.
- V7 is a trainable-quantum engineering case-study, not a benchmark leader. Its best artifact-backed rerun is the April 6, 2026 resumed Colab L4 run at `72.89%` best validation and `72.53%` test.

## Source Protocol

Sources were prioritized in this order: arXiv, DOI landing pages, publisher pages, and institutional repository pages. Blogs and social media are excluded from scientific claims. The review separates what each source supports from what it does not support, so the paper does not turn literature context into unsupported performance claims.

## Directly Relevant Sources

| Area | Source | Relevance | Supports | Does Not Support | Repo Action |
|---|---|---|---|---|---|
| QML benchmarking | Bowles, Ahmed, and Schuld, "Better than classical? The subtle art of benchmarking quantum machine learning models", arXiv:2403.07059, 2024. <https://arxiv.org/abs/2403.07059> | Directly relevant to fair simulated-QML benchmarking. | Strong classical comparators, careful experimental design, and non-leaderboard analysis. | A generic claim that QML should beat classical models on this OCR task. | Keep the paper's cautious benchmark framing and avoid generic quantum advantage language. |
| Practical quantum advantage | Hibat-Allah et al., "A framework for demonstrating practical quantum advantage", Communications Physics, 2024. <https://www.nature.com/articles/s42005-024-01552-6> | Relevant to how advantage claims should be staged and measured. | Predefined comparison rules, application-relevant metrics, and classical race conditions. | That this repo has demonstrated practical quantum advantage. | Present low-data evidence as a scoped competitiveness signal, not as PQA. |
| QML data and baselines | Huang et al., "Power of data in quantum machine learning", Nature Communications, 2021. <https://www.nature.com/articles/s41467-021-22539-9> | Relevant to low-data and advantage claims. | Advantage discussions require strong classical ML and classical approximations of quantum models. | That quantum models should be favored without a matched classical baseline. | Keep `classical_conv`, `thesis_cnniiii`, and `resnet18_cifar_gray` visible in the claim hierarchy. |
| Trainable quanvolution | Kashif and Shafique, "Deep quanvolutional neural networks with enhanced trainability and gradient propagation", Scientific Reports, 2025. <https://www.nature.com/articles/s41598-025-06035-4> | Directly relevant to V7/V8 trainability. | Residual quanvolution and gradient-flow concerns align with this repo's V7 stabilization story. | That simply making V7 deeper will fix performance. | If V8 is considered, start with a residual/deep-quanvolution design note and gradient smoke, not a full run. |
| Hybrid QCNN variants | Long et al., "Hybrid quantum-classical-quantum convolutional neural networks", Scientific Reports, 2025. <https://www.nature.com/articles/s41598-025-13417-1> | Background for hybrid quantum-classical-quantum image classifiers. | Hybrid Q-C-Q designs are an active research direction for small image tasks. | That this architecture transfers directly to 44-class Ottoman handwriting. | Treat as background and avoid using it as proof of expected V8 gains. |
| QCNN image classification | Daka and Bhattacharyya, "A novel quantum convolutional neural network framework for quantum-enhanced classification of pixelated colour images", Scientific Reports, 2026. <https://www.nature.com/articles/s41598-026-45140-w> | Relevant but not directly matched to this dataset. | Low-resolution quantum convolution/pooling pipelines are active in 2026 literature. | That the method solves this repo's 44-class grayscale OCR setting. | Consider only as background for architecture vocabulary and constraints. |
| Equivariant QCNNs | Chinzei et al., "Resource-efficient equivariant quantum convolutional neural networks", Quantum Machine Intelligence, 2026. <https://link.springer.com/article/10.1007/s42484-026-00397-2> | Relevant to possible V8 ansatz design. | Symmetry, locality, and resource-efficient gradient estimation may improve trainability. | A quick implementation path or guaranteed OCR improvement. | Use as a research direction for an equivariant/local-cost ansatz smoke only. |
| Ottoman handwritten and printed OCR | Demir and Özkaya, "An Object Detection-Based Character Recognition Method for Ottoman Handwritten Documents", IJDAR, 2025. <https://doi.org/10.1007/s10032-025-00529-7> | Highly relevant applied OCR context. | Ottoman handwritten and printed character recognition is an active document-analysis problem with object-detection baselines. | Direct comparability to isolated 44-class character classification without dataset alignment. | Cite in related work and use it to justify applied relevance. |
| Printed Ottoman OCR | Demir and Özkaya, "Ottoman Character Recognition on Printed Documents Using Deep Learning", Journal of Engineering Sciences and Design, 2024. <https://doi.org/10.21923/jesd.1383926> | Relevant applied OCR baseline context. | YOLO-style object detection has been used for printed Ottoman character recognition. | Direct comparability to this handwritten isolated-character benchmark. | Cite as printed-document context, not as a direct accuracy comparator. |
| Printed Ottoman OCR | Dölek and Kurt, "Ottoman Optical Character Recognition with deep neural networks", Journal of the Faculty of Engineering and Architecture of Gazi University, 2023. <https://avesis.istanbul.edu.tr/yayin/e3c143a7-8682-45f6-aacc-e6351e7c2b0b/ottoman-optical-character-recognition-with-deep-neural-networks-derin-sinir-aglariyla-osmanlica-optik-karakter-tanima> | Relevant to Ottoman OCR history. | CNN/RNN OCR pipelines have been applied to printed Ottoman naskh-font documents. | Direct evidence about handwritten isolated-character QML. | Cite as Ottoman OCR background and keep modality differences explicit. |
| Ottoman transcription | Sabanci DAS, "Automatic transcription of Ottoman documents using deep learning", 2024. <https://research.sabanciuniv.edu/id/eprint/50453/> | Applied heritage-document context. | Ottoman transcription and OCR/HTR remain active cultural-heritage tasks. | Benchmark comparability to this dataset. | Use as motivation/background, not metric comparison. |

## Background Sources

- Earlier quanvolutional and QCNN work remains useful for context, but May 2026 positioning should emphasize fair benchmarking and trainability rather than suggesting automatic quantum advantage.
- Modern Ottoman OCR work is increasingly object-detection and document-level. This study is narrower: isolated handwritten character classification with controlled classical and quantum-hybrid comparisons.
- The low-data axis is useful because practical quantum-advantage discussions often care about data-limited regimes, but this repo's result is scoped to `non_trainable_quantum` versus `classical_conv`.

## Do-Not-Overclaim Notes

- Do not write that the study demonstrates quantum advantage.
- Do not write that V7 is close to the strongest classical baselines. It is not.
- Do not generalize the low-data signal to all quantum models. `thesis_hqnn2` remains below `thesis_cnniiii` in the currently available low-data evidence.
- Do not compare object-detection mAP from Ottoman document OCR papers directly to isolated-character classification accuracy in this repo.
- Do not treat 2026 QCNN architecture papers as evidence that a new V8 run will improve the submission without a smoke test.

## Actionable For This Repo

1. Keep the paper's central contribution as a fair benchmark and engineering study, with a scoped low-data current-local quantum signal.
2. Add or strengthen related-work citations for Demir and Özkaya 2025, Demir and Özkaya 2024, and Dölek and Kurt 2023 when revising `paper/draft.md`.
3. Keep Bowles/Ahmed/Schuld 2024 and the practical quantum-advantage framework near the benchmark methodology discussion.
4. Use the deep quanvolution and equivariant QCNN papers only to motivate a V8 design note and stop/go gates.
5. Do not start new V7 or V8 training as part of this literature pass.

## Publication Implication

The specialized QML/applied OCR route is currently stronger than a broad Q1 route. A broader route would likely need either a second dataset, a robustness axis, or a stronger methodological contribution beyond the current isolated-character benchmark. The current paper can defend a careful, negative-or-mixed QML benchmark story with a narrow low-data signal, provided the wording stays explicit about model families and evidence limits.
