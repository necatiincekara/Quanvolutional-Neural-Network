# Experimental Log & Results

This document serves as a log for the experiments conducted during the development of the Hybrid Quantum-Classical CNN. It tracks the evolution of the model, detailing the configuration, performance metrics, and accuracy results at each major stage. This log is intended for academic and research purposes.

> Status note, March 22, 2026:
> This file is still useful as the detailed experiment history, but it is no longer the safest single-source summary of the study.
> Later local ablations show that current classical baselines outperform the current trainable and Henderson-style non-trainable quantum variants on test accuracy.
> For the up-to-date publication assessment and claim hierarchy, use `docs/PUBLICATION_STRATEGY_2026-03-22.md` together with `experiments/*.json`.

---

## Experiment 01: V1 - Baseline Naive Implementation

*   **Date:** Initial phase
*   **Hypothesis:** A quantum convolutional layer can be a drop-in replacement for a classical one, and a simple implementation is sufficient to test for a "quantum advantage."
*   **Model Configuration:**
    *   **General:**
        *   **Batch Size:** 64
        *   **Learning Rate:** 0.001 (nominal)
        *   **Optimizer:** Adam
        *   **Scheduler:** None
    *   **Quantum Circuit (`quanv_circuit`):**
        *   **Qubits (`N_QUBITS`):** 4
        *   **Quantum Device:** `default.qubit` (CPU)
        *   **Diff Method:** `parameter-shift`
        *   **Structure:** Basic angle encoding with minimal trainable weights.
    *   **Hybrid Architecture:**
        *   **Input:** 32x32 grayscale image.
        *   **Classical Pre-processing:** None.
        *   **Quantum Layer (`QuanvLayer`):**
            *   Applies the quantum circuit directly to 2x2 patches of the 32x32 input.
            *   **Implementation:** Nested Python `for` loops (non-vectorized).
            *   **Total Quantum Executions:** 256 per image.
        *   **Classical Post-processing:**
            *   `nn.Flatten()`
            *   `nn.Linear(in_features=1024, out_features=44)`
*   **Performance Metrics:**
    *   **Epoch Time:** > 8 hours (estimated).
*   **Accuracy Metrics:**
    *   **Validation Accuracy:** ~2.3% (random guess).
*   **Conclusion:** **Failure.** Computationally infeasible. The model shows no signs of learning.

---

## Experiment 02: V2 - Vectorization and GPU Acceleration

*   **Date:** First optimization phase
*   **Hypothesis:** Vectorizing the quantum layer for GPU parallelism and using a GPU-accelerated simulator will solve the performance bottleneck.
*   **Model Configuration:**
    *   **General:**
        *   **Batch Size:** 64
        *   **Learning Rate:** 0.001
        *   **Optimizer:** Adam
        *   **Scheduler:** `LambdaLR` (Incorrectly stepped per-epoch).
    *   **Quantum Circuit (`quanv_circuit`):**
        *   **Qubits (`N_QUBITS`):** 4
        *   **Quantum Device:** `lightning.gpu` (L4 GPU)
        *   **Diff Method:** `adjoint`
        *   **Structure:** Unchanged from V1.
    *   **Hybrid Architecture:**
        *   **Input:** 32x32 grayscale image.
        *   **Classical Pre-processing:** None.
        *   **Quantum Layer (`QuanvLayer`):**
            *   Vectorized implementation using PyTorch `unfold` and `reshape`.
            *   Processes 256 patches in a single batch operation.
        *   **Classical Post-processing:**
            *   `nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3)`
            *   `nn.BatchNorm2d(16)`
            *   `nn.Flatten()`
            *   `nn.Linear(in_features=..., out_features=44)`
*   **Performance Metrics:**
    *   **Epoch Time:** ~8 hours.
    *   **Batch Time:** ~573 seconds.
*   **Accuracy Metrics:**
    *   **Validation Accuracy:** ~3.3%. Still no significant learning due to the scheduler bug.
*   **Conclusion:** **Partial Success.** Performance became measurable, but still too slow. The learning failure pointed to deeper issues.

---

## Experiment 03: V3 - Architectural Optimization

*   **Date:** Second optimization phase
*   **Hypothesis:** Reducing the quantum workload via classical pre-processing is the key to performance.
*   **Model Configuration:**
    *   **General:**
        *   **Batch Size:** 64
        *   **Learning Rate:** 0.001 (Tuned down during experiments)
        *   **Optimizer:** Adam
        *   **Scheduler:** `LambdaLR` (Correctly stepped per-batch).
    *   **Quantum Circuit (`quanv_circuit`):**
        *   **Qubits (`N_QUBITS`):** 4
        *   **Quantum Device:** `lightning.gpu`
        *   **Diff Method:** `adjoint`
        *   **Structure:** `AngleEmbedding` -> `Rot` gates -> `CNOT` chain.
    *   **Hybrid Architecture:**
        *   **Input:** 32x32 grayscale image.
        *   **Classical Pre-processing:**
            *   `nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)`
            *   `nn.MaxPool2d(kernel_size=2, stride=2)` # Output: 4x16x16
        *   **Quantum Layer (`QuanvLayer`):**
            *   Operates on the 16x16 feature map.
            *   **Total Quantum Executions:** 64 per image (4x reduction).
        *   **Classical Post-processing:**
            *   Deeper CNN with multiple `Conv2d` and `GroupNorm` layers.
*   **Performance Metrics:**
    *   **Epoch Time:** ~5.5 hours.
*   **Accuracy Metrics:**
    *   **Validation Accuracy:** 6.41% (first epoch).
*   **Conclusion:** **Success.** The model learned, and training time improved. Still too slow for rapid iteration.

---

## Experiment 04: V4 - Aggressive Optimization

*   **Date:** Final optimization phase
*   **Hypothesis:** Further reducing quantum patches and optimizing GPU usage will yield a practical model.
*   **Model Configuration:**
    *   **General:**
        *   **Batch Size:** 128
        *   **Learning Rate:** 0.0001
        *   **Optimizer:** Adam
        *   **Scheduler:** `LambdaLR` with warmup, stepped per-batch.
    *   **Quantum Circuit (`quanv_circuit`):**
        *   **Qubits (`N_QUBITS`):** 4
        *   **Quantum Device:** `lightning.gpu`
        *   **Diff Method:** `adjoint`
        *   **Structure:** Unchanged from V3.
    *   **Hybrid Architecture:**
        *   **Input:** 32x32 grayscale image.
        *   **Classical Pre-processing:**
            *   `nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1)` # Output: 4x16x16
            *   `nn.MaxPool2d(kernel_size=2, stride=2)` # Output: 4x8x8
        *   **Quantum Layer (`QuanvLayer`):**
            *   Operates on the 8x8 feature map.
            *   **Total Quantum Executions:** 16 per image (16x reduction vs V1).
        *   **Classical Post-processing:**
            *   `nn.Conv2d(in_channels=16, ...)`
            *   `nn.GroupNorm(...)`
            *   `nn.MaxPool2d(2)`
            *   `nn.Flatten()` -> `nn.Linear(...)` -> `nn.Dropout(0.5)` -> `nn.Linear(...)`
*   **Performance Metrics:**
    *   **Epoch Time:** ~1.5 hours.
*   **Accuracy Metrics:**
    *   Shows a consistent downward trend in loss.
*   **Conclusion:** A stable, performant, and learnable baseline suitable for research.

---

## Experiment 05: V4.1 - Hyperparameter Tuning

*   **Date:** Latest run
*   **Hypothesis:** The stable V4 architecture can be further improved with hyperparameter tuning.
*   **Model Configuration:**
    *   Identical to **Experiment 04 (V4)** in all aspects of architecture and quantum configuration. This experiment only varied training parameters.
*   **Performance Metrics:**
    *   **Batch Time:** ~205 seconds (consistent with V4).
*   **Accuracy Metrics (Run interrupted):**
    *   **Epoch 1 Validation Accuracy:** 8.75%
    *   **Epoch 2 Validation Accuracy:** 8.16%
*   **Conclusion:** Inconclusive but promising. The high initial accuracy suggests the potential of the V4 architecture, but the dip indicates a need for further LR/scheduler tuning.

---

## Experiment 06: V5 - Information Bottleneck Test

*   **Date:** Latest run
*   **Hypothesis:** Reducing the feature map to 4x4 before the quantum layer will maximize performance.
*   **Model Configuration:**
    *   **General:** Based on V4 settings (Batch Size: 128, LR: 0.0001, etc.).
    *   **Quantum Circuit (`quanv_circuit`):** Unchanged from V4.
    *   **Hybrid Architecture:**
        *   **Input:** 32x32 grayscale image.
        *   **Classical Pre-processing:**
            *   `nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1)` # Output: 4x16x16
            *   `nn.ReLU()`
            *   `nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1)` # Output: 4x8x8
            *   `nn.MaxPool2d(kernel_size=2, stride=2)` # Output: 4x4x4
        *   **Quantum Layer (`QuanvLayer`):**
            *   Operates on the tiny 4x4 feature map.
            *   **Total Quantum Executions:** 4 per image (64x reduction vs V1).
        *   **Classical Post-processing:** Same structure as V4, adapted for the smaller input size.
*   **Performance Metrics:**
    *   **Batch Time:** ~51 seconds (massive speedup).
*   **Accuracy Metrics (Run interrupted):**
    *   **Epoch 1 Validation Accuracy:** 2.04%
*   **Conclusion:** **Failure.** Performance hypothesis confirmed, but the extreme spatial reduction created an information bottleneck, destroying the model's ability to learn. This establishes a lower bound on the pre-processing strategy.

---

## Experiment 07: V6 - Balanced Architecture (6x6 Feature Map)

*   **Date:** Completed
*   **Hypothesis:** A 6x6 feature map will provide a balance between the information preservation of V4 (8x8) and the speed of V5 (4x4), resulting in a model that both learns effectively and trains at a practical speed.
*   **Model Configuration:**
    *   **General:** Inherited from V5 (Batch Size: 128, LR: 0.0001, etc.).
    *   **Quantum Circuit (`quanv_circuit`):** Unchanged from V4.
    *   **Hybrid Architecture:**
        *   **Input:** 32x32 grayscale image.
        *   **Classical Pre-processing:**
            *   `nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1)` # Output: 4x16x16
            *   `nn.ReLU()`
            *   `nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1)` # Output: 4x8x8
            *   `nn.ReLU()`
            *   `nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=0)` # Output: 4x6x6
        *   **Quantum Layer (`QuanvLayer`):**
            *   Operates on the 6x6 feature map.
            *   **Total Quantum Executions:** 9 per image (28.4x reduction vs V1).
        *   **Classical Post-processing:** Same structure as V4, adapted for the 6x6 input size.
*   **Performance Metrics:**
    *   **Batch Time:** ~117 seconds per iteration (≈43% faster than V4)
*   **Accuracy Metrics:**
    *   **Epoch 1 Training Loss:** 3.8467
    *   **Epoch 1 Validation Loss:** 3.8173
    *   **Epoch 1 Validation Accuracy:** 0.00%
*   **Conclusion:** **Failure.** Although the training speed improved significantly, the model failed to learn (0% accuracy). Post-mortem analysis suggests the aggressive spatial reduction without proper information preservation mechanisms (residual connections, adaptive pooling) created a severe information bottleneck. The quantum layer likely receives insufficient spatial information to extract meaningful features.
*   **Lessons Learned:**
    *   6x6 is theoretically viable but requires architectural modifications (residual connections, better preprocessing)
    *   V4 (8x8) remains the optimal balance between speed and accuracy
    *   Future experiments should focus on enhancing quantum circuit expressivity rather than further spatial reduction

---

## Experiment 08: V7 - Gradient-Stabilized Trainable Quantum Model

*   **Date:** March 2, 2026
*   **Hypothesis:** Making quantum parameters fully trainable with gradient stabilization techniques (residual connections, gradient scaling, channel attention) will overcome V6's gradient collapse while exceeding V4's 8.75% accuracy.
*   **Model Configuration:**
    *   **General:**
        *   **Batch Size:** 128
        *   **Learning Rate:** Quantum: 0.0005, Classical: 0.002 (separate optimizers)
        *   **Optimizer:** Adam (quantum) + AdamW (classical)
        *   **Scheduler:** CosineAnnealingWarmRestarts (quantum), CosineAnnealingLR (classical)
        *   **Loss:** LabelSmoothingCrossEntropy (smoothing=0.1) + Mixup (alpha=0.2)
    *   **Quantum Circuit (`data_reuploading_circuit`):**
        *   **Qubits (`N_QUBITS`):** 4
        *   **Quantum Device:** `lightning.gpu` (A100 80GB)
        *   **Diff Method:** `adjoint`
        *   **Structure:** 2-layer data re-uploading with AngleEmbedding, Rot gates, CNOT ring topology
        *   **Trainable Parameters:** 25 (vs 12 fixed in V1-V6)
    *   **Hybrid Architecture (EnhancedQuanvNet):**
        *   **Input:** 32x32 grayscale image.
        *   **Classical Pre-processing:**
            *   `Conv2d(1, 8, stride=2)` + GroupNorm + GELU → 16x16
            *   `ResidualBlock(8, 8)` (identity skip connection)
            *   `Conv2d(8, 4, stride=2)` + GroupNorm + GELU → 8x8
        *   **Quantum Layer (`TrainableQuanvLayer`):**
            *   Operates on 8x8 feature map (V4 optimal size).
            *   2x2 patches with stride 2 → 16 quantum executions per image.
            *   **Gradient scaling:** Learnable `gradient_scale` parameter (init=0.1)
            *   **Output:** 16 channels (4 input channels × 4 qubits)
        *   **Skip Connection:**
            *   `Conv2d(4→16, 1x1)` adapter for classical features
            *   Learnable `skip_weight` parameter (init=0.1)
            *   `quantum_out + skip_weight * adapted_classical`
        *   **Channel Attention:** SE-block style (squeeze-excitation)
        *   **Classical Post-processing:**
            *   `Conv2d(16, 32, kernel=3)` + GroupNorm + GELU
            *   `Conv2d(32, 64, kernel=3)` + GroupNorm + GELU + Dropout(0.3)
            *   AdaptiveAvgPool2d(1) → Linear(64, 44)
        *   **Total Parameters:** 87,798 (25 quantum + 87,773 classical)
*   **Run 1 Results (NaN Failure):**
    *   **Epoch 1 Batch 1:** gradient_scale=0.1, quantum grad mean=2.09e+02 (SCALED), classical grad mean=1.59e+03 (SCALED)
    *   **Epoch 1 Final:** Train Loss: **NaN**, Train Acc: 5.91%, Val Acc: 3.50%
    *   **Epoch 2-3:** All NaN — model completely corrupted
    *   **Test Accuracy:** 4.29% (random noise, meaningless)
    *   **Duration:** 6 hours 55 minutes (3 epochs)
*   **Root Cause Analysis (NaN):**
    1.  **AMP Float16 Overflow:** PyTorch Automatic Mixed Precision cast quantum layer inputs to float16. PennyLane quantum circuits require float32 precision — float16 caused numerical overflow in gradient computation, producing NaN.
    2.  **Incorrect GradScaler Usage:** `optimizer.step()` was called directly instead of `scaler.step()`. When AMP's GradScaler detects inf/NaN gradients, `scaler.step()` skips the weight update. Direct `optimizer.step()` applied NaN gradients to model weights, irreversibly corrupting all parameters.
    3.  **Misleading Gradient Diagnostics:** Debug output logged SCALED gradients (inflated by GradScaler's scale factor of ~65536) but compared against UNSCALED thresholds. Reported "quantum grad mean=2.09e+02" appeared alarming but actual unscaled value was ~0.003 (healthy).
    4.  **Learning Rate Propagation Bug:** `QuantumAwareOptimizer` defaults were lowered to (0.0005, 0.002) but `EnhancedTrainer.__init__` passed old values (0.001, 0.005), overriding the fix.
*   **Fixes Applied:**
    *   Added `patches = patches.float()` before quantum circuit — forces float32 regardless of AMP context
    *   Replaced `optimizer.step()` with `scaler.step(quantum_optimizer)` + `scaler.step(classical_optimizer)`
    *   Moved gradient logging after `scaler.unscale_()` — now reports true gradient magnitudes
    *   Fixed LR propagation: EnhancedTrainer passes quantum_lr=0.0005, classical_lr=0.002
    *   Updated gradient health thresholds for unscaled values (vanishing: <1e-7, exploding: >1.0)
*   **Full Training Results (9 epochs, L4 GPU):**

    | Epoch | Train Loss | Train Acc | Val Acc | Q-Grad Mean | C-Grad Mean | alpha |
    |-------|-----------|-----------|---------|-------------|-------------|-------|
    | 1 | 3.5688 | 6.30% | 14.87% | 7.78e-04 | 2.14e-02 | 0.1000 |
    | 2 | 3.1780 | 18.52% | 34.40% | 5.67e-02 | 7.90e-02 | 0.0962 |
    | 3 | 2.8565 | 29.06% | 39.65% | 3.01e-01 | 2.06e-01 | 0.0940 |
    | 4 | 2.7266 | 33.54% | 40.23% | 3.05e-01 | 1.26e-01 | 0.0922 |
    | 5 | 2.6620 | 35.42% | 44.31% | 6.93e-02 | 1.21e-01 | 0.0895 |
    | 6 | 2.4402 | 41.46% | 48.10% | 6.90e-02 | 2.13e-01 | 0.0895 |
    | 7 | 2.3886 | 43.74% | 54.23% | 5.15e-02 | 2.63e-01 | 0.0882 |
    | 8 | ~2.20 | ~46% | 58.31% | 4.50e-02 | 2.74e-01 | 0.0883 |
    | 9 | 2.0757 | 55.98% | **67.35%** | 2.88e-01 | 1.70e-01 | 0.0878 |

    *   **Final Test Accuracy: 65.02%**
    *   **Total training time:** ~20 hours (9 epochs × ~2.2h on L4 GPU)
    *   **Improvement over V4:** 7.4× (val: 8.75% → 67.35%)
    *   **Above random baseline:** 28.6× (2.27% → 65.02%)
*   **Conclusion:** **Full Success.** V7 achieves 65.02% test accuracy with trainable quantum circuit. The gradient stabilization framework (learnable alpha scaling, residual skip, SE-attention) successfully maintains healthy gradient flow across all 9 epochs. Notable: quantum gradient magnitude spans 3 orders of magnitude across training (7.78e-04 → 3.01e-01), suggesting a "cold start" warm-up phase in variational quantum circuit training.
*   **Key Insight for Publication:** This represents the most complete systematic study of trainable quanvolutional architectures on historical script recognition. Combined with the AMP incompatibility finding and information bottleneck characterization, this provides genuine novel contributions to the hybrid QML engineering literature.

---

## Future Experiments & Recommendations

### Experiment 08: Enhanced Quantum Circuit (Proposed)

*   **Goal:** Improve accuracy by increasing quantum circuit expressivity without sacrificing speed
*   **Key Changes:**
    *   Implement 2-layer quantum circuit with data re-uploading
    *   Add trainable quantum weights (16-32 parameters instead of 12)
    *   Use separate optimizers for quantum (LR: 0.0001) and classical (LR: 0.0005) parameters
    *   Add gradient clipping for quantum parameters (max_norm=0.5)
*   **Expected Impact:**
    *   Target: 12-15% validation accuracy (up from 8-9%)
    *   Training time: Similar to V4 (~1.5 hours/epoch)
*   **Implementation:** See `improved_quantum_circuit.py` and `trainable_quantum_model.py`

### Experiment 09: V4 with Residual Connections (Proposed)

*   **Goal:** Add skip connections to V4 architecture to preserve information flow
*   **Key Changes:**
    *   Add residual connection between pre-quantum and post-quantum features
    *   Implement multi-scale feature extraction
    *   Add attention mechanisms before quantum layer
*   **Expected Impact:**
    *   Target: 15-18% validation accuracy
    *   Better gradient flow through quantum layer
*   **References:** See recommendations in `QUANTUM_ML_RECOMMENDATIONS.md`

### Experiment 10: Ensemble & Augmentation (Proposed)

*   **Goal:** Push accuracy to 20%+ through ensemble methods and data augmentation
*   **Key Changes:**
    *   Multiple quantum circuits with voting mechanism
    *   Aggressive data augmentation (rotation, translation, mixup)
    *   Label smoothing and regularization techniques
*   **Expected Impact:**
    *   Target: 20-25% validation accuracy
    *   More robust model with better generalization
*   **Implementation Guide:** See `IMPLEMENTATION_GUIDE.md` for roadmap to 90% accuracy 
