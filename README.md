# Hybrid Quantum-Classical Convolutional Neural Network for Ottoman-Turkish Character Recognition

A research-grade implementation of a hybrid quantum-classical neural network for classifying handwritten Ottoman-Turkish characters (44 classes). Built with PyTorch and PennyLane, leveraging GPU-accelerated quantum simulation and advanced training techniques.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.32+-green.svg)](https://pennylane.ai/)

## 📝 Project Overview

This project explores quantum machine learning (QML) for handwritten character recognition in historical scripts. Ottoman-Turkish characters present unique challenges due to high intra-class variance and complex morphology. Our hybrid architecture uses quantum circuits as trainable feature extractors, targeting quantum advantages in feature correlation learning.

**Research Goal**: Achieve ≥90% accuracy by evolving from fixed quantum layers (82% baseline from master's thesis) to fully trainable quantum circuits with advanced architectural innovations.

### 🏛️ Current Best Architecture (V4)

Our model has evolved through 6 major iterations. **V4 provides the optimal balance** between training speed and accuracy:

```
Input (32×32 grayscale)
    ↓
[Classical Preprocessing]
    Conv2d(1→4, stride=2) → 16×16
    MaxPool2d(2) → 8×8
    ↓
[Quantum Layer - 4 qubits]
    16 patches (2×2 each)
    AngleEmbedding → Rot gates → CNOT chain
    16 quantum circuit evaluations/image
    ↓
[Classical Postprocessing]
    Conv2d(16→32) + GroupNorm
    Conv2d(32→64) + GroupNorm
    MaxPool2d(2)
    ↓
[Classification Head]
    Linear(64*features → 64)
    Dropout(0.5)
    Linear(64 → 44)
```

**Key Components**:
- **Vectorized Quantum Layer**: Batched processing of all image patches on GPU
- **GroupNorm**: Stable training with small effective batch sizes
- **Mixed Precision Training**: AMP for GPU efficiency
- **Per-Batch LR Scheduling**: Warmup + cosine annealing

### 🧪 Evolution & Performance Benchmarks

| Version | Feature Map | Quantum Calls | Epoch Time | Accuracy | Status |
|---------|------------|---------------|------------|----------|---------|
| V1 | 32×32 | 256 | >8h | 2.3% | ❌ Infeasible |
| V2 | 32×32 (vec) | 256 | ~8h | 3.3% | ❌ Scheduler bug |
| V3 | 16×16 | 64 | ~5.5h | 6.41% | ⚠️ Learning |
| **V4** | **8×8** | **16** | **~1.5h** | **8.75%** | **✅ Stable** |
| V5 | 4×4 | 4 | ~51s/batch | 2.04% | ❌ Info bottleneck |
| V6 | 6×6 | 9 | ~117s/batch | 0.00% | ❌ Gradient vanish |

**Key Achievements**:
- ✅ **93.75% reduction** in quantum evaluations (256→16)
- ✅ **Vectorized implementation** enabling GPU parallelism
- ✅ **Stable training** with gradient flow diagnostics
- ✅ **Disk caching** for compiled quantum kernels

**Critical Findings**:
- 🔴 **V6 failure**: Faster (43%) but gradient collapse prevents learning
- 🔴 **Information bottleneck**: <8×8 feature maps lose critical spatial information
- 🟡 **Circuit expressivity**: Single-layer limits model capacity

### 📊 Current Status & Next Steps

**Stable Baseline**: V4 (8×8 feature maps, 8.75% accuracy, 1.5h/epoch)

**Immediate Priorities** (see [docs/AUDIT_REPORT.md](docs/AUDIT_REPORT.md)):
1. **V7** (Week 1-2): Gradient stabilization → Target 25% accuracy
2. **V8** (Week 3-4): Multi-scale processing → Target 40% accuracy
3. **V9** (Week 5-6): Selective quantum → Target 60% accuracy
4. **V10** (Week 7-8): Trainable quantum → Target 90% accuracy

For detailed experimental results, see [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | For implementation roadmap, see [docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md)

## 🚀 Getting Started

### Prerequisites

**Recommended Setup** (see [docs/COMPUTING_RESOURCES_2025.md](docs/COMPUTING_RESOURCES_2025.md) for details):

*   **Python 3.12.x** (or 3.13.x) - Full PyTorch 2.6+ and PennyLane 0.43+ support
*   **Google Colab Pro** with A100 GPU (CUDA 12.1) - Essential for quantum training
*   **VS Code** with Google Colab extension - Seamless local development + cloud execution
*   **Git** for version control

**Important**: M4 Mac Mini lacks CUDA support - cannot run `lightning.gpu` quantum simulator. Use Colab Pro for training.

### ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd Quanvolutional-Neural-Network
    ```

2.  **Install dependencies:** It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Dataset:**
    This project expects a dataset of handwritten Ottoman characters located in a Google Drive folder. You must ensure your dataset is available at the paths specified in `src/config.py` (`TRAIN_PATH` and `TEST_PATH`). When running in Google Colab, you will need to mount your Drive.

### 🏃‍♀️ Running the Training

The training script supports starting from scratch or resuming from a checkpoint.

*   **To start a new training session:**
    *   (Optional) If you want to ensure a clean start, delete any existing checkpoints: `rm -f models/checkpoint_latest.pth`
    *   Run the training script:
        ```bash
        python -m src.train
        ```

*   **To resume from the last checkpoint:**
    ```bash
    python -m src.train --resume
    ```
The script will automatically save the model with the best validation accuracy to `models/best_quanv_net.pth` and the latest state for resuming to `models/checkpoint_latest.pth`.

## 📁 Project Structure

```
Quanvolutional-Neural-Network/
├── src/                           # Core source code
│   ├── config.py                  # Hyperparameters and paths
│   ├── dataset.py                 # Ottoman character data loading
│   ├── model.py                   # Base quantum-classical hybrid (V4/V6)
│   ├── train.py                   # Training pipeline with AMP
│   ├── trainable_quantum_model.py # Enhanced trainable circuits
│   └── enhanced_training.py       # Advanced training framework
├── docs/                          # Documentation
│   ├── AUDIT_REPORT.md            # Comprehensive codebase audit
│   ├── EXPERIMENTS.md             # Detailed experimental log (V1-V6)
│   ├── IMPLEMENTATION_GUIDE.md    # Step-by-step development guide
│   ├── QUANTUM_ML_RECOMMENDATIONS.md  # QML best practices
│   ├── RESEARCH_ROADMAP.md        # Publication roadmap
│   ├── TRAINING_PLATFORM_GUIDE.md # Colab/Mac setup guides
│   └── COLAB_SETUP.md             # Google Colab configuration
├── experiments/                   # Experimental scripts
│   └── run_experiments.py         # Automated ablation studies
├── improved_model.py              # Alternative architectures
├── improved_training.py           # Training optimizations
├── improved_quantum_circuit.py    # Enhanced circuit designs
├── performance_optimizations.py   # Benchmarking utilities
├── models/                        # Saved checkpoints (created at runtime)
├── CLAUDE.md                      # AI assistant instructions
├── requirements.txt               # Python dependencies
└── README.md                      # This file
``` 