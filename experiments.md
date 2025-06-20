# Experimental Log & Results

This document serves as a log for the experiments conducted during the development of the Hybrid Quantum-Classical CNN. It tracks the evolution of the model, detailing the configuration, performance metrics, and accuracy results at each major stage. This log is intended for academic and research purposes.

---

## Experiment 01: Baseline Naive Implementation

*   **Date:** Initial phase
*   **Hypothesis:** A quantum convolutional layer can be a drop-in replacement for a classical one, and a simple implementation is sufficient to test for a "quantum advantage."
*   **Model Configuration:**
    *   **Quantum Layer:** `QuanvLayer` processing patches from the full 32x32 image.
    *   **Implementation:** Nested Python `for` loops for patching.
    *   **Simulator:** PennyLane's `default.qubit` (CPU-based).
    *   **Classical Part:** A single fully-connected layer for classification.
*   **Performance Metrics:**
    *   **Epoch Time:** > 8 hours (estimated).
    *   **Batch Time:** Not practically measurable due to extreme slowness.
*   **Accuracy Metrics:**
    *   **Validation Accuracy:** ~2.3% (equivalent to random guessing for 44 classes).
    *   **Loss:** Stagnated around ~3.78.
*   **Conclusion:** **Failure.** The implementation is computationally infeasible for a dataset of this size. The model shows no signs of learning. The initial hypothesis is rejected due to practical constraints.

---

## Experiment 02: Vectorization and GPU Acceleration

*   **Date:** First optimization phase
*   **Hypothesis:** The performance bottleneck can be solved by vectorizing the quantum layer to leverage GPU parallelism and by using a dedicated GPU-accelerated quantum simulator.
*   **Model Configuration:**
    *   **Quantum Layer:** `QuanvLayer` refactored to use PyTorch's `unfold` and `reshape` for batch processing.
    *   **Simulator:** `lightning.gpu` on an L4 GPU.
    *   **Classical Part:** `Conv2d` + `BatchNorm` + `Linear` layers.
    *   **Training:** Per-epoch LR scheduler implemented.
*   **Performance Metrics:**
    *   **Epoch Time:** ~8 hours (observed on L4 GPU).
    *   **Batch Time:** ~573 seconds.
*   **Accuracy Metrics:**
    *   **Validation Accuracy:** ~3.3%. Still no significant learning.
    *   **Loss:** Stagnated. The `LR scheduler` was later found to be implemented incorrectly (stepping per-epoch instead of per-batch), which was the primary cause of the learning failure.
*   **Conclusion:** **Partial Success.** Performance was made measurable, but still far too slow for practical use. The model's failure to learn highlighted deeper issues in the training loop and architecture.

---

## Experiment 03: Architectural Optimization to Reduce Quantum Workload

*   **Date:** Second optimization phase
*   **Hypothesis:** The primary bottleneck is the sheer number of quantum circuit evaluations. Reducing this workload with a classical pre-processing layer will yield the most significant performance gains.
*   **Model Configuration:**
    *   **Pre-processing:** A `Conv2d(stride=1) + MaxPool2d` block was added before the quantum layer, reducing the feature map from 32x32 to 16x16.
    *   **Quantum Layer:** Now processes 4x fewer patches.
    *   **Classical Part:** Deepened with additional `Conv2d` layers.
    *   **Simulator:** `lightning.gpu` on an L4 GPU.
*   **Performance Metrics:**
    *   **Epoch Time:** ~5.5 hours.
    *   **Batch Time:** ~413 seconds.
*   **Accuracy Metrics:**
    *   **Epoch 1 Training Loss:** 3.7361
    *   **Epoch 1 Validation Loss:** 3.6908
    *   **Epoch 1 Validation Accuracy:** 6.41%
*   **Conclusion:** **Success.** This was the breakthrough. The model demonstrated clear signs of learning, and the training time was significantly reduced. However, the epoch time was still too long for rapid experimentation. The `CPU/CUDA device mismatch` error also appeared and was fixed at this stage.

---

## Experiment 04: Aggressive Optimization for Practical Training

*   **Date:** Final optimization phase
*   **Hypothesis:** Further reducing the quantum patch count, increasing batch size for better GPU utilization, and stabilizing the training process will result in a practical and effective model.
*   **Model Configuration:**
    *   **Pre-processing:** `Conv2d` stride increased to 2, reducing the feature map to 8x8 (a total **16x** reduction in quantum patches from the start).
    *   **Batch Size:** Increased from 64 to 128.
    *   **Normalization:** `BatchNorm` replaced with `GroupNorm` for better stability.
    *   **Training:** Corrected per-batch LR scheduler with a warm-up phase. Enabled PennyLane's kernel disk cache.
*   **Performance Metrics (Initial Run):**
    *   **Epoch Time:** ~1.5 hours (projected).
    *   **Batch Time:** ~208 seconds.
*   **Accuracy Metrics (Initial Run):**
    *   **Loss:** Initially stagnated due to an incorrect LR scheduler implementation (now fixed). After the fix, the loss shows a consistent downward trend.
*   **Conclusion:** The project has reached a stable, performant, and learnable baseline. The current configuration is suitable for further hyperparameter tuning and research, having balanced the trade-offs between quantum feature extraction and classical processing efficiency. 

---

## Experiment 05: Hyperparameter Tuning on V4 Architecture

*   **Date:** Latest run
*   **Hypothesis:** The stable V4 architecture serves as a baseline for hyperparameter tuning. This experiment tests a specific configuration with the goal of improving initial learning dynamics and peak accuracy.
*   **Model Configuration:**
    *   The configuration is assumed to be identical to Experiment 04 (V4), as no code changes were specified. This includes the `Conv2d` with `stride=2`, `GroupNorm`, and a batch size of 128, running on a GPU.
*   **Performance Metrics:**
    *   **Batch Time:** ~205 seconds. This metric is consistent with the performance observed in Experiment 04, confirming that the computational workload remains the same.
*   **Accuracy Metrics (Run interrupted):**
    *   **Epoch 1 Training Loss:** 3.7841
    *   **Epoch 1 Validation Loss:** 3.6937
    *   **Epoch 1 Validation Accuracy:** 8.75%
    *   **Epoch 2 Training Loss:** 3.7254
    *   **Epoch 2 Validation Loss:** 3.6533
    *   **Epoch 2 Validation Accuracy:** 8.16%
*   **Conclusion:** The experiment started with the highest initial validation accuracy to date (8.75%), suggesting a potentially effective set of hyperparameters. However, the accuracy slightly decreased in the second epoch, which could indicate that the learning rate is suboptimal or that the model is experiencing early instability. The run was terminated before a clear trend could be established, making the results inconclusive but valuable for future tuning attempts. Additionally, the log produced a `FutureWarning` for `torch.cuda.amp.autocast`, indicating a minor, non-critical dependency update is needed in `train.py`. 