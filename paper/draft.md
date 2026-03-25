# Fair Benchmarking and Engineering Lessons for Quanvolutional Neural Networks on Ottoman-Turkish Handwritten Character Recognition

> Editorial status note, March 25, 2026:
> This draft still reflects the older V7-centered manuscript state and is not submission-ready in its current form.
> The current benchmark picture is now:
> - `thesis_cnniiii`: `85.26 ± 0.97` test
> - `classical_conv`: `81.40 ± 1.06` test
> - `param_linear`: `81.12 ± 2.27` test
> - `non_trainable_quantum`: `80.40 ± 0.69` test
> - `thesis_hqnn2`: `78.61 ± 0.69` test
> - documented `V7 trainable quantum`: `65.02%` test
> Therefore the paper must now be rewritten as a fair benchmark + engineering-lessons manuscript, not as a V7 accuracy-win manuscript.
> Before using this draft for submission, reconcile claims against `docs/SUBMISSION_BENCHMARK_2026-03-25.md`, `docs/PUBLICATION_STRATEGY_2026-03-22.md`, `docs/EXPERIMENTS.md`, and `experiments/*.json`.

## Authors

Necati Incekara

---

## Abstract

We present a reproducible benchmark extension of a master's-thesis study on quanvolutional neural networks for Ottoman-Turkish handwritten character recognition, a difficult 44-class low-data task with 3,894 grayscale samples. The study combines two complementary views of the problem: thesis-faithful model reproductions and current-local matched-budget ablations. Across multi-seed evaluation, the strongest reproduced evidence currently favors classical baselines rather than quantum variants. In the thesis-faithful family, the strongest reproduced model is `thesis_cnniiii` with **85.26 ± 0.97%** test accuracy, while the best reproduced thesis-faithful quantum baseline, `thesis_hqnn2`, reaches **78.61 ± 0.69%**. In the current-local matched-budget family, `classical_conv` reaches **81.40 ± 1.06%**, `param_linear` reaches **81.12 ± 2.27%**, and the Henderson-style non-trainable quantum baseline reaches **80.40 ± 0.69%**. We therefore do not claim a quantum advantage on this task. Instead, the scientific value of the work lies in three contributions: (1) a fair comparative benchmark that separates thesis-faithful and matched-budget families, (2) a trainable-quantum engineering case-study showing how information bottlenecks, gradient collapse, and optimizer/precision mistakes affect hybrid training, and (3) the documentation of a practical incompatibility between PyTorch AMP float16 autocasting and variational quantum circuit backpropagation at the quantum boundary. The stabilized V7 trainable model remains useful as an engineering result, reaching **65.02%** documented test accuracy while exposing concrete design rules for hybrid quantum-classical pipelines.

**Keywords:** Quantum Machine Learning, Quanvolutional Neural Networks, Hybrid Quantum-Classical Computing, Ottoman Script Recognition, Variational Quantum Circuits, Gradient Stabilization, Barren Plateaus

---

## 1. Introduction

Quantum machine learning (QML) has emerged as a promising paradigm for leveraging quantum computational properties in pattern recognition tasks [1,2]. Among various QML architectures, quanvolutional neural networks---which replace or augment classical convolutional filters with parameterized quantum circuits (PQCs)---have attracted significant research attention since their introduction by Henderson et al. [3]. However, the practical deployment of quanvolutional layers faces substantial engineering and theoretical challenges that are often underreported in the literature.

A central concern in the QML community is the preponderance of positive results: papers demonstrating quantum enhancement while omitting systematic failure analysis [see Bowles et al., 2024]. This "positive bias" undermines reproducibility and obscures the true practical limits of near-term quantum machine learning. In this work, we deliberately adopt a failure-first methodology, documenting every architectural failure in detail alongside our eventual successes.

We apply quanvolutional neural networks to Ottoman-Turkish handwritten character recognition, a historically significant and technically challenging classification problem. The Ottoman script comprises 44 distinct character classes with significant morphological variation and limited training data (3,894 samples), making it an ideal testbed for evaluating quantum feature extractors under resource-constrained conditions.

### 1.1 Contributions

Our contributions are fourfold:

1. **Reproducible Benchmark Structure.** We organize the study into two analytically separate families: thesis-faithful reproductions and current-local matched-budget ablations. This prevents misleading direct comparisons between differently sized or differently motivated models and yields an artifact-backed benchmark picture for the repository.

2. **Negative-Result Benchmark Evidence.** We show that the strongest reproduced evidence on this 44-class task currently favors classical baselines. The best thesis-faithful model is classical (`thesis_cnniiii`), and the best current-local matched-budget model is also classical (`classical_conv`). This makes the work more honest and more informative than a selective "quantum win" narrative.

3. **Information Bottleneck and Gradient Stabilization Analysis.** Through the V1--V7 trainable-quantum path, we identify an empirical spatial threshold below which quantum feature extraction collapses, and we show that learnable scaling, residual routing, and channel attention are sufficient to transform a non-learning hybrid architecture into a trainable one.

4. **AMP--PennyLane Incompatibility Documentation.** We identify and document a concrete failure mode in which PyTorch AMP float16 autocasting corrupts variational quantum circuit backpropagation unless float32 is restored at the quantum boundary and optimizer stepping remains GradScaler-aware.

### 1.2 Relation To The Thesis

This paper should be read as a continuation and formalization of the master's-thesis study, not as a repudiation of it. The thesis established the dataset framing, the OCR problem formulation, and the initial CNN/HQNN comparison space. The present paper extends that foundation in four ways:

1. it re-runs key thesis models under a unified and reproducible protocol,
2. it separates thesis-faithful reproductions from newer matched-budget ablations,
3. it adds multi-seed reporting instead of relying on single highlighted runs, and
4. it incorporates the later V1--V7 engineering path for trainable quantum layers.

Under this framing, the thesis remains scientifically valuable as the origin of the task definition and the first model family, while the paper provides the stricter reproducibility, benchmarking, and engineering analysis needed for publication.

### 1.3 Paper Organization

Section 2 reviews related work. Section 3 describes the dataset, architectures, and training pipelines. Section 4 presents the benchmark results, separating thesis-faithful reproductions, current-local matched-budget ablations, and the trainable-quantum engineering case-study. Section 5 discusses the scientific implications of the benchmark picture. Section 6 concludes with realistic publication takeaways and future work.

---

## 2. Related Work

### 2.1 Quanvolutional Neural Networks

Henderson et al. [3] introduced quanvolutional neural networks, demonstrating that random non-trainable quantum circuits can be used as fixed feature extractors before a classical classifier. That formulation remains important because it motivates the non-trainable preprocessing baselines used both in our thesis-faithful reproductions and in our current-local Henderson-style comparisons. Later work such as Hur et al. [4] explored trainable quantum convolutional variants for classical data classification, opening the question of whether learned quantum parameters provide a practical benefit over random or classical replacements.

Our study differs from most one-family quanvolution papers in that it does not treat all quantum models as interchangeable. Instead, it distinguishes between: (1) thesis-faithful reproductions of earlier CNN/HQNN designs, (2) current-local matched-budget ablations designed for fair comparison, and (3) a trainable-quantum engineering case-study. This separation is necessary because different quanvolutional setups answer different scientific questions.

### 2.2 Benchmarking, Reproducibility, and Quantum Claims

Recent work has raised critical questions about quantum-advantage claims in the classical simulation regime. Bowles et al. [11] showed that several claimed quantum improvements disappear when stronger or more carefully tuned classical baselines are used. This criticism is especially relevant for small-data image classification tasks, where modest implementation differences can dominate the apparent contribution of a quantum block.

Our paper adopts this more conservative benchmark philosophy. Instead of asking only whether a quantum model can beat a weak historical baseline, we ask a stricter question: what remains true after multi-seed reruns, artifact-backed reporting, family separation, and comparison against matched classical alternatives? This places the present work closer to a reproducibility and failure-analysis contribution than to a pure performance-claim paper.

### 2.3 Quantum Circuit Training and Hybrid Engineering

The barren plateau phenomenon [7] poses a fundamental challenge for training parameterized quantum circuits: gradient magnitudes can decrease sharply with system size and circuit structure, making optimization difficult. For shallow 4-qubit circuits as used in this work, barren plateaus are less severe but optimization pathologies still remain [8]. Data re-uploading [10] has been proposed as a circuit design strategy to increase expressibility without increasing qubit count, which we adopt as the primary trainable circuit family in the V7 line.

The engineering side of hybrid classical-quantum training is much less standardized than the theoretical side. Mixed precision training, optimizer scheduling, gradient scaling, and skip-routing are routine concerns in classical deep learning but remain underreported in hybrid pipelines. Our work contributes in this space by treating the V1--V7 line as an explicit engineering study: which design choices make quanvolution trainable, which ones cause collapse, and which precision assumptions silently break the entire run.

### 2.4 Ottoman Script Recognition and Thesis Context

Ottoman-Turkish handwriting recognition is an active research area given the historical significance of Ottoman documents [9]. Classical deep learning approaches have achieved moderate success on limited datasets. The script's 44 character classes with significant intra-class variation and limited training data (~88 samples/class average) present challenges representative of real-world QML deployment scenarios with data-scarce settings.

This paper builds directly on a master's-thesis study conducted on the same OCR problem. The thesis established the task framing and the initial CNN/HQNN comparison space; the present work extends it by introducing reproducible reruns, multi-seed reporting, stricter model-family separation, and a trainable-quantum engineering analysis. In that sense, the thesis is the foundation of the paper rather than a discarded precursor.

*[References [9] and [11] to be completed with full citations]*

---

## 3. Methodology

### 3.1 Dataset

The Ottoman-Turkish Handwritten Character Dataset consists of 32x32 grayscale character images across 44 classes. The original directory structure contains 3,428 training files and 466 test files. Under the current parser, one malformed training filename is skipped, so the effective loaded benchmark uses 3,427 training samples and 466 test samples. The original thesis train/test separation is preserved rather than re-sampled from scratch, allowing direct continuity between thesis-era results and the present paper.

| Property | Value |
|----------|-------|
| Classes | 44 Ottoman-Turkish characters |
| Total files | 3,894 |
| Effective loaded train set | 3,427 images |
| Test set | 466 images |
| Image dimensions | 32x32 grayscale |
| Avg. samples/class | ~88 (training) |
| Random baseline | 2.27% (1/44) |

The class distribution is substantially imbalanced, so the benchmark should be interpreted as a challenging small-data heritage OCR problem rather than as a balanced large-scale vision benchmark. This is one reason we report multi-seed performance and keep benchmark families analytically separate.

#### 3.1.1 Publication Protocol

To support reproducibility, the publication benchmark uses the following rules:

- the thesis train/test split remains fixed,
- validation is derived deterministically from the training split,
- runs are tracked with explicit train seed and split seed,
- key benchmark rows are reported with three seeds,
- artifact-backed JSON logs are treated as the primary result source.

#### 3.1.2 Benchmark Families

The methodology separates three families of evidence:

1. **Thesis-faithful reproductions:** reruns of the thesis-era CNN/HQNN models under the current publication protocol.
2. **Current-local matched-budget ablations:** smaller local models designed to compare classical replacements against non-trainable quantum preprocessing under similar parameter budgets.
3. **Trainable-quantum engineering case-study:** the V1--V7 development path used to study gradient flow, bottlenecks, and precision failures.

This separation is methodologically important. A thesis-faithful HQNN model and a current-local Henderson-style non-trainable quantum baseline are both "quantum" models, but they do not answer the same question and should not be merged into a single ranking row.

### 3.2 Quantum Circuit Design

We employ small parameterized quantum circuits as the core computational units of the quanvolutional blocks. Different benchmark families use different circuit roles:

- thesis-faithful HQNN models use fixed non-trainable quantum preprocessing,
- current-local Henderson-style baselines use cached non-trainable quantum filters,
- the V7 engineering line uses a 4-qubit trainable circuit with data re-uploading.

The primary trainable circuit uses the data re-uploading strategy [10]:

#### 3.2.1 Data Re-uploading Circuit (Primary)

```
For each layer l in {1, 2}:
    AngleEmbedding(x, wires=[0,1,2,3])        # encode 4 input values
    For each qubit q in {0,1,2,3}:
        Rot(theta_l,q,0, theta_l,q,1, theta_l,q,2, wires=q)   # trainable
    CNOT(0->1), CNOT(1->2), CNOT(2->3), CNOT(3->0)            # entanglement ring
Measure: <Z_0>, <Z_1>, <Z_2>, <Z_3>
```

**Trainable parameters:** 24 (2 layers × 4 qubits × 3 Euler angles per Rot gate) + 1 learnable gradient scaling parameter = **25 total quantum parameters**.

#### 3.2.2 Strongly Entangling Circuit (Ablation)

Uses PennyLane's `StronglyEntanglingLayers` template with 3 layers, providing 36 trainable parameters with guaranteed expressibility bounds.

#### 3.2.3 Hardware-Efficient Circuit (Ablation)

RY and RZ rotations with CZ entangling gates: 16 trainable parameters. Designed for compatibility with near-term NISQ devices.

### 3.3 Hybrid Architecture: V7 (EnhancedQuanvNet)

The V7 architecture is described in detail because it is the main trainable-quantum engineering case-study in the paper. It is not the strongest benchmark model overall, but it is the most informative architecture for analyzing how trainable quanvolution behaves under realistic optimization constraints.

#### 3.3.1 Classical Preprocessing

```
Input: 32x32x1
  -> Conv2d(1, 8, stride=2) + GroupNorm(8) + GELU    -> 16x16x8
  -> ResidualBlock(8, 8)  [Conv-GN-GELU-Conv-GN + skip]  -> 16x16x8
  -> Conv2d(8, 4, stride=2) + GroupNorm(4) + GELU    -> 8x8x4
Output: 8x8x4  [64 spatial values per channel]
```

GroupNorm is used throughout instead of BatchNorm for stability with small effective batch sizes in the quantum regime.

#### 3.3.2 Quantum Processing (TrainableQuanvLayer)

The quanvolutional layer applies 2x2 patch extraction with stride 2 to the 8x8 feature map, yielding 16 patches per channel. Each patch's 4 values are encoded into the 4-qubit PQC via AngleEmbedding.

**Key precision boundary:** Before quantum processing, inputs are explicitly cast to float32 regardless of the AMP autocast context:
```python
patches = patches.float()   # prevent AMP float16 from entering PQC
quantum_output = self.qlayer(patches) * self.gradient_scale
```

Output dimensions: 4×4 spatial × 4 expectation values × 4 input channels = 4×4×16.

**Quantum computational cost:** 16 circuit evaluations per image (93.75% reduction from V1's 256).

#### 3.3.3 Gradient Stabilization

Three mechanisms prevent gradient vanishing at the quantum-classical interface:

1. **Learnable Gradient Scaling (α):** Initialized to 0.1, scales quantum outputs: `y_q = α · f_quantum(x)`. Prevents early training instability from large quantum output magnitudes.

2. **Residual Skip Connection (β):** A 1×1 convolution adapter (`Conv2d(4, 16, 1×1)`) with learnable weight β (init=0.1) provides a gradient highway bypassing the quantum layer: `y = y_q + β · W_skip(x_classical)`.

3. **Channel Attention (SE-Block):** Squeeze-and-excitation attention recalibrates channel-wise responses post-quantum, allowing the network to weight informative quantum measurements.

#### 3.3.4 Classical Post-processing

```
4x4x16
  -> Conv2d(16, 32, 3×3) + GroupNorm(8) + GELU
  -> Conv2d(32, 64, 3×3) + GroupNorm(16) + GELU + Dropout(0.3)
  -> AdaptiveAvgPool2d(1)  -> 64-dim vector
  -> Linear(64, 44)
Output: 44-class logits
```

**Total parameters:** 87,798 (25 quantum + 87,773 classical).

### 3.4 Training Pipeline

#### 3.4.1 Dual Optimizer Strategy

Quantum and classical parameters are optimized separately to account for their different loss landscape geometries:

| Optimizer | Parameters | LR | Weight Decay | Grad Clip |
|-----------|-----------|-----|-------------|-----------|
| Adam | Quantum (25) | 0.0005 | 1e-5 | max_norm=0.5 |
| AdamW | Classical (87,773) | 0.002 | 1e-4 | max_norm=1.0 |

Schedulers: `CosineAnnealingWarmRestarts(T_0=5)` for quantum; `CosineAnnealingLR(T_max=10)` for classical.

#### 3.4.2 Regularization

- **Label smoothing:** ε=0.1 in cross-entropy loss
- **Mixup augmentation:** Applied with 50% probability, α=0.2
- **Dropout:** 0.3 in post-processing layers
- **GroupNorm:** Replaces BatchNorm throughout

#### 3.4.3 AMP Integration

Standard AMP (float16) is used for classical computations. The quantum boundary requires explicit handling:

```python
# GradScaler-aware optimizer stepping (prevents NaN weight corruption)
scaler.scale(loss).backward()
scaler.unscale_(quantum_optimizer)
scaler.unscale_(classical_optimizer)
torch.nn.utils.clip_grad_norm_(quantum_params, max_norm=0.5)
torch.nn.utils.clip_grad_norm_(classical_params, max_norm=1.0)
scaler.step(quantum_optimizer)
scaler.step(classical_optimizer)
scaler.update()
```

Using `optimizer.step()` directly (bypassing `scaler.step()`) caused irreversible model corruption in V7 Run 1 by applying NaN gradients to all parameters.

### 3.5 Computational Infrastructure

- **Primary trainable-quantum training:** NVIDIA L4 GPU (Google Colab Pro, ~2-3 compute units/hour)
- **Initial exploratory runs:** NVIDIA A100-SXM4-80GB (Google Colab Pro, ~7-8 units/hour)
- **Reproducibility and benchmark reruns:** Apple M4 Mac Mini (CPU, `default.qubit`) for all M4-feasible thesis-faithful and matched-budget models
- **Quantum simulator:** PennyLane `lightning.gpu` with adjoint differentiation
- **Framework:** PyTorch 2.x, PennyLane 0.44, NumPy ≥ 2.0
- **Batch size / epochs:** configuration depends on benchmark family; the documented V7 case-study shown in this paper uses batch size 128 and 9 epochs

---

## 4. Experimental Results

### 4.1 Benchmark Snapshot

The benchmark picture must be read in three layers rather than as a single flat leaderboard:

1. **Thesis-faithful family:** reproductions of the thesis-era classical and quantum models.
2. **Current-local matched-budget family:** smaller parameter-matched ablations designed for fair local comparison.
3. **Trainable-quantum case-study:** the V1--V7 engineering path, which is valuable scientifically even though it is not the strongest benchmark family.

### 4.2 Thesis-Faithful Family

| Model | Runs | Best Val | Test | Params | Interpretation |
|-------|---:|---:|---:|---:|---|
| `thesis_cnniiii` | 3 | **92.11 ± 0.30** | **85.26 ± 0.97** | 1,378,124 | strongest thesis-faithful reproduction; above thesis table reference |
| `thesis_cnn3` | 3 | 85.38 ± 0.77 | 79.33 ± 1.26 | 769,804 | weaker classical thesis-faithful anchor |
| `thesis_hqnn2` | 3 | 83.72 ± 2.23 | 78.61 ± 0.69 | 248,428 | best thesis-faithful quantum reproduction, but below `thesis_cnniiii` and below thesis table reference |

This family is the most relevant when asking whether the thesis-era quantum claim remains competitive under reproducible reruns. The answer is mixed: `thesis_hqnn2` remains competitive with `thesis_cnn3`, but the strongest thesis-faithful reproduced evidence clearly favors the classical `thesis_cnniiii` model.

### 4.3 Current-Local Matched-Budget Family

| Model | Runs | Best Val | Test | Params | Interpretation |
|-------|---:|---:|---:|---:|---|
| `classical_conv` | 3 | 86.26 ± 1.76 | **81.40 ± 1.06** | 88,045 | strongest matched-budget local model by mean test accuracy |
| `param_linear` | 3 | **86.45 ± 0.61** | 81.12 ± 2.27 | 87,798 | matched classical replacement, nearly tied with `classical_conv` on mean test |
| `non_trainable_quantum` | 3 | 85.77 ± 0.94 | 80.40 ± 0.69 | 88,488 | stable Henderson-style non-trainable quantum baseline, but not the strongest local model |

This family asks a narrower question: if the parameter budget is kept close to the current V7-style local models, do quantum layers beat classical replacements? Under the present `publication_v1` protocol, the answer is no. The Henderson-style non-trainable quantum baseline is stable, but it does not exceed the strongest classical matched-budget alternatives.

### 4.4 Trainable-Quantum Engineering Case-Study

The V1--V7 line remains scientifically useful, but it should be interpreted as an engineering case-study rather than as the main benchmark winner. Its value lies in showing what failed, what became trainable, and why.

#### 4.4.1 Architectural Evolution Summary

| Version | Feature Map | Q-Calls/img | Epoch Time | Best Val Acc. | Outcome |
|---------|------------|-------------|------------|--------------|---------|
| V1 | 32×32 | 256 | >8h | 2.3% | Computationally infeasible |
| V2 | 32×32 | 256 (GPU) | ~8h | 3.3% | LR scheduler bug |
| V3 | 16×16 | 64 | ~5.5h | 6.4% | First learning signal |
| V4 | 8×8 | 16 | ~1.5h | 8.75% | Stable non-trainable baseline |
| V5 | 4×4 | 4 | ~51s/batch | 2.04% | Information bottleneck |
| V6 | 6×6 | 9 | ~117s/batch | 0.00% | Gradient collapse |
| V7-Run1 | 8×8 | 16 | ~5.4min/batch | NaN | AMP float16 bug |
| **V7-Run2** | **8×8** | **16** | **~2.3h/epoch** | **67.35%** | **stabilized trainable-quantum case-study** |

#### 4.4.2 V7 Full Training Curve

Complete epoch-by-epoch documented results for V7 (data_reuploading circuit, L4 GPU):

| Epoch | Train Loss | Train Acc. | Val Acc. | Q-Grad Mean | C-Grad Mean | gradient_scale α |
|-------|-----------|-----------|---------|-------------|-------------|-----------------|
| 1 | 3.5688 | 6.30% | 14.87% | 7.78e-04 | 2.14e-02 | 0.1000 |
| 2 | 3.1780 | 18.52% | 34.40% | 5.67e-02 | 7.90e-02 | 0.0962 |
| 3 | 2.8565 | 29.06% | 39.65% | 3.01e-01 | 2.06e-01 | 0.0940 |
| 4 | 2.7266 | 33.54% | 40.23% | 3.05e-01 | 1.26e-01 | 0.0922 |
| 5 | 2.6620 | 35.42% | 44.31% | 6.93e-02 | 1.21e-01 | 0.0895 |
| 6 | 2.4402 | 41.46% | 48.10% | 6.90e-02 | 2.13e-01 | 0.0895 |
| 7 | 2.3886 | 43.74% | 54.23% | 5.15e-02 | 2.63e-01 | 0.0882 |
| 8 | ~2.20 | ~46% | 58.31% | 4.50e-02 | 2.74e-01 | 0.0883 |
| 9 | 2.0757 | 55.98% | **67.35%** | 2.88e-01 | 1.70e-01 | 0.0878 |

**Documented test accuracy:** `65.02%`

The V7 result is not competitive with the strongest reproduced classical baselines, but it remains informative because it demonstrates that trainable hybrid quanvolution can be stabilized and meaningfully optimized after multiple failed versions.

### 4.5 Information Bottleneck Analysis

Systematic reduction of feature-map dimensions reveals a critical spatial threshold:

| Feature Map | Spatial Values/Channel | Q-Calls/img | Val Acc. | Gradient Status |
|------------|----------------------|-------------|---------|----------------|
| 8×8 | 64 | 16 | 8.75% (V4) / 67.35% (V7) | Healthy |
| 6×6 | 36 | 9 | 0.00% | Complete collapse |
| 4×4 | 16 | 4 | 2.04% | Below random baseline |

**Finding:** For 44-class Ottoman character recognition with 4-qubit circuits, feature maps must maintain ≥8×8 spatial resolution before the quantum layer. Below this threshold, the quantum output standard deviation approaches zero regardless of circuit expressivity or training duration---the quantum component cannot learn.

This establishes an empirical lower bound on classical preprocessing aggressiveness in hybrid architectures: faster quantum evaluation (fewer spatial patches) comes at the cost of information bottleneck failure below the threshold.

### 4.6 Gradient Flow Analysis

#### V6 Failure Case (6×6 Feature Maps)

- Quantum output standard deviation: <1e-6 (effectively constant)
- No gradient signal through quantum layer
- Model converged to constant predictions (0% accuracy)
- Root cause: Insufficient spatial information combined with fixed 12-parameter quantum circuit and no gradient stabilization

#### V7 Gradient Health

Quantum gradient magnitude across all 9 epochs ranged from 7.78e-04 to 3.01e-01. The growth pattern (from negligible in epoch 1 to substantial in epochs 3-4 and 9) suggests a characteristic "cold start" dynamics in variational quantum circuits within hybrid architectures---the quantum circuit requires initial warm-up epochs before meaningful gradient signal propagates. This phenomenon warrants further investigation.

### 4.7 AMP--PennyLane Incompatibility

**Mechanism:** Under PyTorch AMP autocast, the forward pass automatically downcasts tensor operations to float16. The PennyLane quantum circuit processes these float16 inputs during unitary evolution in a 2^4=16 dimensional Hilbert space. Adjoint differentiation computes quantum gradients via matrix inversion of the unitary; float16 precision is insufficient for this operation, causing overflow.

**Cascade failure sequence:**
1. float16 input enters `qlayer()`
2. Adjoint differentiation overflows → NaN quantum gradients
3. `optimizer.step()` called directly (bypassing GradScaler protection) → NaN gradients applied to weights
4. All 87,798 parameters corrupted with NaN within epoch 1
5. All subsequent computations produce NaN

**Timeline:** The model appeared to train normally for 13 batches (loss: 3.786→3.604, accuracy: 0→10.94%) before NaN cascade propagated.

**Fix:** Two lines of code:
```python
patches = patches.float()         # at quantum boundary in forward()
scaler.step(quantum_optimizer)    # in training loop, not optimizer.step()
```

**Diagnostic note:** GradScaler inflates gradient magnitudes by its scale factor (~65,536). Debug outputs of "quantum grad mean=2.09e+02" in Run 1 appeared alarming but represented unscaled magnitude ~0.003 (healthy). Moving gradient logging after `scaler.unscale_()` is essential for accurate diagnostics in hybrid AMP pipelines.

### 4.8 Quantum Computational Efficiency

| Transition | Technique | Q-Call Reduction | Epoch Speedup |
|-----------|-----------|-----------------|---------------|
| V1→V2 | Vectorization + GPU | 0% | ~0% |
| V2→V3 | Classical preprocessing (32→16) | 75% | ~31% |
| V3→V4 | Aggressive preprocessing (16→8) | 75% | ~73% |
| **V1→V4/V7** | **Combined** | **93.75%** | **>80%** |

### 4.9 Comparative Interpretation

| Variant | Quantum Params | Val Acc. | Test Acc. | Notes |
|---------|---------------|---------|---------|-------|
| V4 (non-trainable, old arch.) | 0 trainable | 8.75% | — | Original baseline |
| V7 (trainable, data_reuploading) | 25 | 67.35% | 65.02% | stabilized engineering case-study, not current benchmark leader |
| `classical_conv` | 0 | 86.26 ± 1.76 | **81.40 ± 1.06** | strongest current-local matched-budget baseline |
| `non_trainable_quantum` | 0 trainable | 85.77 ± 0.94 | 80.40 ± 0.69 | Henderson-style cached quantum baseline |
| `param_linear` | 0 | **86.45 ± 0.61** | 81.12 ± 2.27 | matched classical replacement for the quantum block |
| `thesis_hqnn2` | 0 trainable | 83.72 ± 2.23 | 78.61 ± 0.69 | best thesis-faithful quantum reproduction |
| `thesis_cnniiii` | 0 | **92.11 ± 0.30** | **85.26 ± 0.97** | strongest thesis-faithful reproduced model |

---

## 5. Discussion

### 5.1 What The Benchmark Now Shows

The most important scientific update in the repository is no longer the V4-to-V7 jump alone. The stronger conclusion is that, once the benchmark is organized fairly and run with multi-seed reporting, the strongest reproduced evidence currently favors classical baselines. This is true in both benchmark families considered here:

- in the thesis-faithful family, `thesis_cnniiii` clearly exceeds `thesis_hqnn2`,
- in the current-local matched-budget family, `classical_conv` and `param_linear` both slightly exceed the Henderson-style non-trainable quantum baseline on mean test accuracy.

This does not make the quantum experiments uninformative. It simply constrains the valid claim: the current evidence supports an honest comparative benchmark and engineering analysis, not a quantum-win narrative.

### 5.2 The Quantum Preprocessing Trade-off

Our experiments still reveal a fundamental design tension in hybrid quantum-classical architectures: reducing quantum computational overhead through classical preprocessing simultaneously reduces the spatial information available to the quantum feature extractor. The 8×8 threshold identified in the V4--V7 path suggests a minimum information density requirement for 4-qubit circuits on this 44-class task.

This finding has practical value beyond Ottoman script recognition. Hybrid QML designers must balance:

1. computational feasibility, which favors fewer quantum evaluations, and
2. information sufficiency, which favors larger pre-quantum feature maps.

### 5.3 Engineering Challenges In Hybrid Pipelines

The AMP incompatibility documents an underreported class of hybrid-QML failures: modern classical optimization defaults are not always safe when a variational quantum circuit sits inside the forward and backward path. In this study, float16 autocasting at the quantum boundary was sufficient to destroy the trainable V7 run until explicit float32 restoration and GradScaler-aware optimizer stepping were enforced.

Our recommendation for hybrid QML practitioners is therefore concrete:

1. maintain explicit float32 casting at quantum boundaries,
2. treat optimizer stepping and gradient logging carefully under AMP,
3. verify gradient health empirically rather than assuming that shallow circuits will train cleanly.

### 5.4 Why Thesis-Faithful Reproduction Still Matters

The repository now contains two distinct quantum baselines: the current-local Henderson-style cached baseline and the thesis-faithful HQNN-II reproduction. Keeping them separate is essential. They differ in architecture, parameter count, and motivation, so combining them into a single "quantum" row would be scientifically misleading.

The thesis-faithful path matters because it answers a narrower historical question: does the thesis-era best quantum baseline remain credible under modern reproducible reruns? Our current answer is that it remains competitive with one weaker thesis-era classical reference (`thesis_cnn3`), but it does not surpass the strongest reproduced thesis-faithful classical model.

This interpretation should not be misread as "the thesis was wrong." A more accurate reading is that the thesis identified a promising comparison space, while the paper extends it with stricter protocols, clearer family separation, and more demanding evidence standards. Under those stricter rules, some thesis-era headline numbers are not recovered exactly, but the thesis remains the scientific foundation of the present study.

### 5.5 Limitations And Scope

1. **No quantum advantage claimed.** The current benchmark picture does not support a claim that quantum variants outperform the strongest reproduced classical baselines on this dataset.

2. **Simulator-based evaluation.** All quantum experiments rely on classical simulation. This is acceptable for methodology research, but it prevents any hardware-side performance claim.

3. **Single dataset.** Results remain specific to Ottoman handwritten character recognition. A second dataset, low-data scaling study, or robustness axis would materially strengthen the paper.

4. **Trainable-quantum artifact gap.** The V7 result is currently documented from prior runs, but a fresh artifact-backed rerun would still improve the paper if that case-study remains central.

---

## 6. Conclusion

We presented a reproducible benchmark and engineering study of quanvolutional neural networks for Ottoman-Turkish handwritten character recognition. The strongest current evidence in the repository favors classical baselines rather than quantum variants: the best thesis-faithful reproduced model is `thesis_cnniiii` with **85.26 ± 0.97%** test accuracy, and the strongest current-local matched-budget model is `classical_conv` with **81.40 ± 1.06%**. The thesis-faithful quantum reproduction `thesis_hqnn2` reaches **78.61 ± 0.69%**, while the documented stabilized trainable V7 case-study reaches **65.02%**.

The paper therefore should not claim quantum superiority. Its value lies elsewhere:

1. **Fair benchmark evidence:** classical baselines remain stronger under the current reproduced evidence.
2. **Engineering lessons:** trainable hybrid quanvolution can be stabilized, but only when information flow, gradient routing, and precision boundaries are handled carefully.
3. **Negative-result clarity:** honest separation of thesis-faithful and matched-budget benchmark families prevents misleading conclusions.

Future work should focus on strengthening the benchmark rather than chasing an unsupported generic quantum-advantage claim: (1) add at least one stronger modern classical baseline, (2) add one broader validation axis such as low-data scaling or a second dataset, and (3) obtain a fresh artifact-backed rerun for the trainable V7 path if the engineering case-study is to remain prominent in the final submission.

---

## References

[1] Biamonte, J., et al. "Quantum machine learning." *Nature* 549.7671 (2017): 195-202.

[2] Schuld, M., & Petruccione, F. *Machine Learning with Quantum Computers.* Springer (2021).

[3] Henderson, M., et al. "Quanvolutional neural networks: powering image recognition with quantum circuits." *Quantum Machine Intelligence* 2.1 (2020): 1-9.

[4] Hur, T., Kim, L., & Park, D. K. "Quantum convolutional neural network for classical data classification." *Quantum Machine Intelligence* 4.1 (2022): 1-18.

[5] Li, W., & Deng, D. L. "Recent advances for quantum classifiers." *Science China Physics, Mechanics & Astronomy* 65.2 (2022): 220301.

[6] Senokosov, A., et al. "Quantum machine learning for image classification." *arXiv:2304.09224* (2023).

[7] McClean, J. R., et al. "Barren plateaus in quantum neural network training landscapes." *Nature Communications* 9.1 (2018): 4812.

[8] Cerezo, M., et al. "Cost function dependent barren plateaus in shallow parametrized quantum circuits." *Nature Communications* 12.1 (2021): 1791.

[9] *[Ottoman script recognition reference to be added — search: "Ottoman handwriting recognition deep learning"]*

[10] Perez-Salinas, A., et al. "Data re-uploading for a universal quantum classifier." *Quantum* 4 (2020): 226.

[11] Bowles, J., et al. "Better classical surrogate models for quantum classifiers." *arXiv:2403.07998* (2024).

---

## Appendix A: Experimental Configuration

*Complete hyperparameter settings, debug output, and per-version configurations documented in `docs/EXPERIMENTS.md`.*

## Appendix B: Reproducibility

- **Code:** https://github.com/necatiincekara/Quanvolutional-Neural-Network
- **Training notebook:** `train_v7_colab.ipynb`
- **Framework versions:** PyTorch 2.x, PennyLane 0.44, NumPy ≥ 2.0, CUDA 12.1
- **Hardware:** NVIDIA L4 / A100-SXM4-80GB (Google Colab Pro); Apple M4 Mac Mini (development)
- **Random seeds:** *[to be specified before submission]*
- **Dataset:** *[access instructions to be added]*
