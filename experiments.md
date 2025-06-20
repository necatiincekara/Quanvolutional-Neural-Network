# Experimental Log & Results

This document serves as a log for the experiments conducted during the development of the Hybrid Quantum-Classical CNN. It tracks the evolution of the model, detailing the configuration, performance metrics, and accuracy results at each major stage. This log is intended for academic and research purposes.

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

*   **Date:** Pending
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
*   **Conclusion:** **Failure.** Although the training speed improved significantly, the model failed to learn (0 % accuracy). Diagnostics required to determine whether the quantum layer is "dead" (no signal) or gradients are vanishing due to learning-rate scheduling. 