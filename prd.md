# Product Requirements Document: Hybrid Quanvolutional Neural Network

## 1. Overview & Problem Statement

**Product:** A hybrid quantum-classical neural network for classifying handwritten Ottoman-Turkish characters.
**Problem:** Standard classical CNNs are effective for character recognition but can struggle with scripts that have high structural similarity and variance, like handwritten Ottoman. This project investigates if a Quantum Machine Learning (QML) approach, specifically a Quanvolutional layer, can provide a "quantum advantage" by learning complex correlations in the feature space that are inaccessible to classical models.
**Goal:** Develop a functional, performant, and reasonably accurate hybrid model that serves as a baseline for future QML research in this domain.

---

## 2. Target Audience

*   **Primary:** QML Researchers, Computer Science students, and AI developers interested in practical applications of quantum computing.
*   **Secondary:** Digital Humanities scholars working with historical manuscripts.

---

## 3. The Journey: From Naive Implementation to Optimized Model

This project was not a linear path. It was an iterative process of identifying bottlenecks and solving them. Understanding this journey is key to understanding the final architecture.

> **Note:** A detailed log of the quantitative results for each experiment described below can be found in [`experiments.md`](./experiments.md).

### **V1: The Naive Implementation (Initial State)**

*   **Architecture:** A single Jupyter/Colab notebook containing a simple Quanvolutional layer followed by a classical dense layer.
*   **Hypothesis:** A quantum circuit can act as a direct replacement for a classical convolutional layer.
*   **Result: FAILURE**
    *   **Performance:** Training was impossibly slow (>8 hours per epoch). The Quanvolutional layer, implemented with nested Python `for` loops, was the bottleneck.
    *   **Accuracy:** The model failed to learn, with accuracy stuck at the random-guess level (~2.3%).

### **V2: The Modular & Vectorized Model**

*   **Hypothesis:** Proper software engineering (modularity) and leveraging GPU-native tensor operations (vectorization) will solve the performance issues.
*   **Changes:**
    1.  **Code Modularity:** The project was split into `config`, `dataset`, `model`, and `train` modules.
    2.  **Vectorization:** The `QuanvLayer`'s `for` loops were replaced with PyTorch's `unfold` and `reshape` operations, allowing all image patches to be processed in a single batch.
    3.  **GPU Simulator:** Switched from `default.qubit` to the high-performance `lightning.gpu` simulator.
    4.  **Checkpointing:** Added `--resume` functionality to save and load training state.
*   **Result: PARTIAL SUCCESS**
    *   **Performance:** Training time per epoch was significantly reduced.
    *   **Accuracy:** The model still failed to learn. The loss function stagnated.

### **V3: The Learning Model (Solving the Accuracy Problem)**

*   **Hypothesis:** The model is not learning due to architectural flaws or unstable training dynamics, not a fundamental flaw in the hybrid concept.
*   **Changes:**
    1.  **Diagnostic Phase:** A systematic check of the quantum layer's output variance and the model's gradients confirmed that a learning signal *was* flowing, but it was either too weak or unstable.
    2.  **LR Scheduler Fix:** Discovered the learning rate scheduler was being stepped incorrectly (per-epoch instead of per-batch). This was the primary bug preventing learning.
    3.  **Stability Improvements:** Replaced `BatchNorm` with `GroupNorm` to better handle the small effective batch sizes after patching.
    4.  **Learning Rate Tuning:** Experimented with and lowered the learning rate to prevent gradient explosion in the early phases of training.
*   **Result: SUCCESS (Functionally)**
    *   **Performance:** Training was still too slow for practical experimentation (multiple hours per epoch).
    *   **Accuracy:** The model was now demonstrably learning, with validation accuracy rising above the random-guess baseline.

### **V4: The Performant & Optimized Model (Current State)**

*   **Hypothesis:** The bottleneck is no longer the implementation but the sheer number of quantum circuit evaluations. We must reduce the quantum workload without sacrificing model capacity.
*   **Changes:**
    1.  **Classical Pre-processing:** Introduced a `Conv2d(stride=2) + MaxPool2d` block *before* the quantum layer. This reduced the input feature map from 32x32 to 8x8, cutting the number of quantum patches by **16x**.
    2.  **Deeper Classical Backbone:** To compensate for the reduced quantum workload, the classical CNN *after* the quantum layer was made deeper.
    3.  **Batch Size Increase:** Increased batch size from 64 to 128 to improve GPU utilization.
    4.  **Quantum Kernel Caching:** Enabled PennyLane's disk cache for compiled kernels to dramatically speed up the first epoch on subsequent runs.
*   **Result: OPTIMIZED BASELINE**
    *   **Performance:** Training time is now manageable (~1-2 hours per epoch).
    *   **Accuracy:** The model achieves a stable learning trajectory, providing a solid baseline for further experiments.

---

## 4. Future Work & Potential Improvements

*   **Hyperparameter Tuning:** Systematically tune learning rate, dropout rate, and optimizer parameters (e.g., `weight_decay` in AdamW).
*   **Data Augmentation:** Apply standard computer vision data augmentation techniques (random rotations, shifts, scaling) to improve model generalization.
*   **Advanced Quantum Circuits:** Experiment with different quantum circuit architectures (e.g., different entangling layers, data encoding strategies).
*   **Transfer Learning:** Use a pre-trained classical CNN (like ResNet18) as the feature extractor and apply the Quanvolutional layer to its output feature maps.
*   **Hardware Execution:** Adapt the model to run on real quantum hardware through services like Amazon Braket, which would require exploring noise-aware training techniques. 