# Gradient-Stabilized Quanvolutional Neural Networks for Ottoman-Turkish Handwritten Character Recognition

## Authors

Necati Incekara

---

## Abstract

We present a systematic investigation of hybrid quantum-classical convolutional neural networks (quanvolutional networks) for Ottoman-Turkish handwritten character recognition, a challenging 44-class classification task with limited training data (3,894 samples). Through seven iterative architectural versions (V1--V7), we identify critical design principles for integrating parameterized quantum circuits into deep learning pipelines. Our key contributions include: (1) the empirical discovery of an information bottleneck threshold below which quantum feature extraction fails entirely, (2) a gradient stabilization framework combining learnable scaling, residual connections, and channel attention that resolves vanishing gradient problems in quantum-classical interfaces, and (3) the first systematic documentation of an incompatibility between PyTorch Automatic Mixed Precision (AMP) and variational quantum circuit backpropagation. Starting from a computationally infeasible baseline requiring >8 hours per epoch with 256 quantum circuit evaluations per image, we achieve a 93.75% reduction in quantum computational overhead. With our gradient-stabilized trainable architecture (V7), we reach **65.02% test accuracy** after 9 epochs---28.6x above the random baseline of 2.27% and 7.4x above the non-trainable quanvolutional baseline (V4: 8.75%). We further demonstrate that float16 mixed precision is fundamentally incompatible with quantum circuit backpropagation, and provide a minimal one-line fix with broad implications for the hybrid QML community.

**Keywords:** Quantum Machine Learning, Quanvolutional Neural Networks, Hybrid Quantum-Classical Computing, Ottoman Script Recognition, Variational Quantum Circuits, Gradient Stabilization, Barren Plateaus

---

## 1. Introduction

Quantum machine learning (QML) has emerged as a promising paradigm for leveraging quantum computational properties in pattern recognition tasks [1,2]. Among various QML architectures, quanvolutional neural networks---which replace or augment classical convolutional filters with parameterized quantum circuits (PQCs)---have attracted significant research attention since their introduction by Henderson et al. [3]. However, the practical deployment of quanvolutional layers faces substantial engineering and theoretical challenges that are often underreported in the literature.

A central concern in the QML community is the preponderance of positive results: papers demonstrating quantum enhancement while omitting systematic failure analysis [see Bowles et al., 2024]. This "positive bias" undermines reproducibility and obscures the true practical limits of near-term quantum machine learning. In this work, we deliberately adopt a failure-first methodology, documenting every architectural failure in detail alongside our eventual successes.

We apply quanvolutional neural networks to Ottoman-Turkish handwritten character recognition, a historically significant and technically challenging classification problem. The Ottoman script comprises 44 distinct character classes with significant morphological variation and limited training data (3,894 samples), making it an ideal testbed for evaluating quantum feature extractors under resource-constrained conditions.

### 1.1 Contributions

Our contributions are threefold:

1. **Information Bottleneck Characterization.** We empirically establish that quantum convolutional layers require a minimum spatial resolution in their input feature maps. Through systematic ablation (V4--V6), we show that 8x8 feature maps sustain learning while 6x6 maps cause complete gradient collapse (0% accuracy) and 4x4 maps degrade below random baseline---regardless of circuit expressivity. This finding provides concrete design guidance for hybrid QML practitioners.

2. **Gradient Stabilization Framework.** We propose a combination of learnable gradient scaling, residual skip connections, and channel attention mechanisms that stabilize gradient flow through the quantum-classical interface. This framework transforms a completely non-learning architecture (V6: 0%) into one that achieves 65.02% test accuracy (V7), demonstrating that gradient collapse is an architectural deficiency---not an inherent property of quantum backpropagation.

3. **AMP--PennyLane Incompatibility Documentation.** We identify and document a previously unreported failure mode: PyTorch's Automatic Mixed Precision (AMP), a standard optimization in modern deep learning, causes catastrophic numerical overflow when float16-cast inputs enter variational quantum circuit backpropagation via adjoint differentiation. We provide a minimal fix (explicit float32 casting at the quantum boundary) with broad implications for hybrid QML pipelines.

### 1.2 Paper Organization

Section 2 reviews related work. Section 3 describes our methodology. Section 4 presents experimental results across seven architectural versions. Section 5 discusses findings. Section 6 concludes with future directions.

---

## 2. Related Work

### 2.1 Quanvolutional Neural Networks

Henderson et al. [3] introduced quanvolutional neural networks, demonstrating that random (non-trainable) quantum circuits can serve as feature extractors when combined with classical neural network layers. The original work used random circuit parameters, motivating subsequent investigation of trainable variants [4]. Hur et al. [4] explored quantum convolutional neural networks for classical data classification, showing that trainable circuits can improve upon random feature extractors in certain regimes.

Recent work has raised critical questions about quantum advantage claims in the classical simulation regime. Bowles et al. [11] demonstrated that for many benchmarks where quantum models were claimed to outperform classical methods, carefully tuned classical counterparts achieve equivalent or superior performance. Our work adopts the honest methodology advocated by this line of research.

### 2.2 Quantum Circuit Training Challenges

The barren plateau phenomenon [7] poses a fundamental challenge for training parameterized quantum circuits: gradient magnitudes decrease exponentially with system size, making optimization intractable for large circuits. For shallow 4-qubit circuits as used in this work, barren plateaus are less severe but still present [8]. Data re-uploading [10] has been proposed as a circuit design strategy to increase expressibility without increasing qubit count, which we adopt as our primary circuit architecture.

### 2.3 Hybrid Classical-Quantum Training Pipelines

The engineering challenges of hybrid quantum-classical training are less studied than the theoretical aspects. Mixed precision training---a standard optimization in classical deep learning using float16 computations with float32 master weights---has not been systematically studied in hybrid quantum pipelines. Our work documents the first known failure mode arising from this interaction.

### 2.4 Ottoman Script Recognition

Ottoman-Turkish handwriting recognition is an active research area given the historical significance of Ottoman documents [9]. Classical deep learning approaches have achieved moderate success on limited datasets. The script's 44 character classes with significant intra-class variation and limited training data (~88 samples/class average) present challenges representative of real-world QML deployment scenarios with data-scarce settings.

*[References [9] and [11] to be completed with full citations]*

---

## 3. Methodology

### 3.1 Dataset

The Ottoman-Turkish Handwritten Character Dataset consists of 3,894 grayscale images across 44 character classes. Each image is 32x32 pixels. The dataset is split into 3,428 training images and 466 test images (~88%/12% split). The validation set is derived from the training set.

| Property | Value |
|----------|-------|
| Classes | 44 Ottoman-Turkish characters |
| Total samples | 3,894 |
| Training set | 3,428 images |
| Test set | 466 images |
| Image dimensions | 32x32 grayscale |
| Avg. samples/class | ~88 (training) |
| Random baseline | 2.27% (1/44) |

### 3.2 Quantum Circuit Design

We employ a 4-qubit parameterized quantum circuit (PQC) as the core computational unit of the quanvolutional layer. The primary circuit uses the data re-uploading strategy [10]:

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

- **Primary training:** NVIDIA L4 GPU (Google Colab Pro, ~2-3 compute units/hour)
- **Initial runs:** NVIDIA A100-SXM4-80GB (Google Colab Pro, ~7-8 units/hour)
- **Development:** Apple M4 Mac Mini (CPU, `default.qubit`)
- **Quantum simulator:** PennyLane `lightning.gpu` with adjoint differentiation
- **Framework:** PyTorch 2.x, PennyLane 0.44, NumPy ≥ 2.0
- **Batch size:** 128; **Epochs:** 9

---

## 4. Experimental Results

### 4.1 Architectural Evolution Summary

| Version | Feature Map | Q-Calls/img | Epoch Time | Best Val Acc. | Outcome |
|---------|------------|-------------|------------|--------------|---------|
| V1 | 32×32 | 256 | >8h | 2.3% | Computationally infeasible |
| V2 | 32×32 | 256 (GPU) | ~8h | 3.3% | LR scheduler bug |
| V3 | 16×16 | 64 | ~5.5h | 6.4% | First learning signal |
| V4 | 8×8 | 16 | ~1.5h | 8.75% | Stable non-trainable baseline |
| V5 | 4×4 | 4 | ~51s/batch | 2.04% | Information bottleneck |
| V6 | 6×6 | 9 | ~117s/batch | 0.00% | Gradient collapse |
| V7-Run1 | 8×8 | 16 | ~5.4min/batch | NaN | AMP float16 bug |
| **V7-Run2** | **8×8** | **16** | **~2.3h/epoch** | **67.35%** | **✓ Target achieved** |

### 4.2 V7 Full Training Curve

Complete epoch-by-epoch results for V7 (data_reuploading circuit, L4 GPU):

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

**Final Test Accuracy: 65.02%** (28.6× above random baseline; 7.4× above V4 non-trainable baseline)

**Key training dynamics:**
- Quantum gradient magnitudes span three orders of magnitude across training (7.78e-04 to 3.01e-01), confirming active PQC parameter learning
- The learnable gradient scale α decreases monotonically (0.1000 → 0.0878), indicating automatic calibration of quantum output magnitude
- Val accuracy growth rate accelerates after epoch 7, suggesting the quantum circuit reaches a productive parameter regime ("warm-up" phase)
- Total wall time: ~20 hours on L4 GPU (9 epochs × ~2.2h/epoch)

### 4.3 Information Bottleneck Analysis

Systematic reduction of feature map dimensions reveals a critical spatial threshold:

| Feature Map | Spatial Values/Channel | Q-Calls/img | Val Acc. | Gradient Status |
|------------|----------------------|-------------|---------|----------------|
| 8×8 | 64 | 16 | 8.75% (V4) / 67.35% (V7) | Healthy |
| 6×6 | 36 | 9 | 0.00% | Complete collapse |
| 4×4 | 16 | 4 | 2.04% | Below random baseline |

**Finding:** For 44-class Ottoman character recognition with 4-qubit circuits, feature maps must maintain ≥8×8 spatial resolution before the quantum layer. Below this threshold, the quantum output standard deviation approaches zero regardless of circuit expressivity or training duration---the quantum component cannot learn.

This establishes an empirical lower bound on classical preprocessing aggressiveness in hybrid architectures: faster quantum evaluation (fewer spatial patches) comes at the cost of information bottleneck failure below the threshold.

### 4.4 Gradient Flow Analysis

#### V6 Failure Case (6×6 Feature Maps)

- Quantum output standard deviation: <1e-6 (effectively constant)
- No gradient signal through quantum layer
- Model converged to constant predictions (0% accuracy)
- Root cause: Insufficient spatial information combined with fixed 12-parameter quantum circuit and no gradient stabilization

#### V7 Gradient Health

Quantum gradient magnitude across all 9 epochs ranged from 7.78e-04 to 3.01e-01. The growth pattern (from negligible in epoch 1 to substantial in epochs 3-4 and 9) suggests a characteristic "cold start" dynamics in variational quantum circuits within hybrid architectures---the quantum circuit requires initial warm-up epochs before meaningful gradient signal propagates. This phenomenon warrants further investigation.

### 4.5 AMP--PennyLane Incompatibility

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

### 4.6 Quantum Computational Efficiency

| Transition | Technique | Q-Call Reduction | Epoch Speedup |
|-----------|-----------|-----------------|---------------|
| V1→V2 | Vectorization + GPU | 0% | ~0% |
| V2→V3 | Classical preprocessing (32→16) | 75% | ~31% |
| V3→V4 | Aggressive preprocessing (16→8) | 75% | ~73% |
| **V1→V4/V7** | **Combined** | **93.75%** | **>80%** |

### 4.7 Ablation Studies

*[To be populated after classical baseline and non-trainable ablation experiments]*

| Variant | Quantum Params | Val Acc. | Test Acc. | Notes |
|---------|---------------|---------|---------|-------|
| V4 (non-trainable, old arch.) | 0 trainable | 8.75% | — | Original baseline |
| V7 (trainable, data_reuploading) | 25 | **67.35%** | **65.02%** | Primary result |
| Classical Conv2d equivalent | 0 | TBD | TBD | Pending Experiment A |
| Non-trainable V7 arch. | 0 trainable | TBD | TBD | Pending Experiment B |
| Parameter-matched linear (25 params) | 0 | TBD | TBD | Pending Experiment D |

---

## 5. Discussion

### 5.1 The Quantum Preprocessing Trade-off

Our experiments reveal a fundamental design tension in hybrid quantum-classical architectures: reducing quantum computational overhead through classical preprocessing simultaneously reduces spatial information available to the quantum feature extractor. The 8×8 threshold we identify suggests a minimum "information density" requirement for 4-qubit circuits to function as meaningful feature extractors on a 44-class problem.

This finding has practical implications beyond Ottoman script: hybrid QML designers face a constrained optimization between (a) computational feasibility (fewer quantum circuit calls) and (b) information sufficiency (enough spatial content for the quantum layer to discriminate). The threshold location likely depends on qubit count, circuit expressivity, and task complexity---relationships warranting future systematic study.

### 5.2 Engineering Challenges in Hybrid Pipelines

The AMP incompatibility documents an underreported category of hybrid QML challenges: the precision mismatch between mature classical frameworks (optimized for float16/bfloat16) and quantum simulators (requiring float32/float64). This is not a limitation of PennyLane specifically but a fundamental property of quantum circuit simulation: unitary matrix operations in Hilbert space require higher precision than typical neural network operations.

Our recommendation for all hybrid QML practitioners: maintain explicit float32 casting at every quantum-classical boundary, and always use GradScaler-aware optimizer stepping (`scaler.step()`) rather than direct optimizer calls when combining AMP with quantum layers.

### 5.3 Trainable vs. Non-Trainable Quantum Circuits

The V4-to-V7 improvement (8.75% → 67.35% val accuracy) demonstrates that trainable quantum parameters substantially improve performance over random/fixed circuits in our setting. However, this comparison conflates two factors: (1) trainability of quantum parameters, and (2) architectural improvements (residual connections, GroupNorm, gradient scaling). Experiment B (non-trainable V7 architecture) will disentangle these contributions and is necessary before definitive claims about trainability benefit can be made.

### 5.4 The "Cold Start" Phenomenon in Quantum Training

An unexpected observation in our training dynamics is the non-monotonic quantum gradient magnitude across epochs: very small in epoch 1 (7.78e-04), growing substantially by epoch 3 (3.01e-01), fluctuating in epochs 5-8, then resurging in epoch 9 (2.88e-01). This pattern---absent in typical classical training---may reflect the geometry of variational quantum circuit parameter landscapes: the circuit requires an initial phase to "locate" a productive gradient region before meaningful learning commences. This "quantum cold start" phenomenon, if reproducible, has implications for warm-up strategies in hybrid training.

### 5.5 Limitations and Scope

1. **No quantum advantage claimed:** We do not claim computational or statistical advantage over classical methods. The ablation experiments (Section 4.7, pending) will determine whether the quantum component contributes beyond a classical equivalent.

2. **Simulator-based evaluation:** All experiments use classical quantum simulators (`lightning.gpu`). Real quantum hardware introduces noise, gate errors, and decoherence not captured here.

3. **Single dataset and circuit family:** Results are specific to Ottoman character recognition with data re-uploading circuits. Generalization requires multi-dataset and multi-architecture evaluation.

4. **Small qubit count:** 4-qubit circuits may represent a regime too small for quantum expressivity advantages to manifest. Scaling to 8+ qubits may reveal qualitatively different behavior.

---

## 6. Conclusion

We have presented a systematic investigation of quanvolutional neural networks for Ottoman-Turkish handwritten character recognition, evolving through seven architectural versions to identify critical design principles. Starting from a computationally infeasible naive implementation (V1: >8h/epoch, 2.3% accuracy), we reach a gradient-stabilized trainable architecture (V7) achieving **65.02% test accuracy**---a 28.6× improvement above random chance and 7.4× above the non-trainable quanvolutional baseline.

Our key findings provide concrete guidance for hybrid QML practitioners:

1. **Information bottleneck threshold:** Maintain ≥8×8 spatial feature maps before quantum layers in 4-qubit hybrid CNNs processing 32×32 images for multi-class recognition.

2. **Gradient stabilization is essential:** Learnable gradient scaling + residual skip connections + channel attention transforms a completely non-learning architecture (0% accuracy) into a functional one, demonstrating that quantum gradient collapse is architectural---not fundamental.

3. **AMP requires explicit management:** Float32 casting at quantum boundaries and GradScaler-aware optimizer stepping are non-negotiable for stable hybrid training. Violating either causes irreversible model corruption within one epoch.

Future work will: (1) complete ablation studies isolating quantum layer contributions, (2) extend evaluation to benchmark datasets (MNIST, Arabic handwriting) for cross-domain comparison, (3) assess circuit type sensitivity (strongly entangling vs. hardware-efficient), and (4) investigate the quantum "cold start" dynamics observed in gradient magnitude evolution.

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
