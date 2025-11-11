# Hybrid Quantum-Classical Convolutional Neural Network for Ottoman-Turkish Character Recognition

This project implements a hybrid neural network that combines classical convolutional layers with a quantum convolutional layer (Quanvolutional) to classify handwritten Ottoman-Turkish characters. The model is built using PyTorch and PennyLane, leveraging GPU acceleration for both classical and quantum components.

## 📝 Project Overview

The goal of this project is to explore the potential of quantum machine learning (QML) in a real-world computer vision task. Handwritten character recognition for historical scripts like Ottoman-Turkish presents unique challenges due to high intra-class variance and complex character shapes. This hybrid model uses a quantum circuit as a feature extractor, aiming to capture complex correlations that might be challenging for classical CNNs alone.

### 🏛️ Model Architecture

The final model architecture is a result of several optimization iterations aimed at balancing performance and accuracy.

1.  **Classical Pre-processing:** The input image (32x32) is first passed through a classical `Conv2d` layer with a stride of 2, followed by a `MaxPool2d` layer. This block acts as an efficient feature summarizer, reducing the spatial dimensions to 8x8 while increasing the feature channels. This significantly reduces the number of quantum circuit evaluations needed in the next step.
2.  **Quantum Convolutional Layer (`QuanvLayer`):** The 8x8 feature map is processed by a vectorized Quanvolutional layer.
    *   **Patching:** The feature map is divided into 2x2 patches.
    *   **Quantum Circuit:** Each patch's data is encoded into a 4-qubit quantum circuit using `AngleEmbedding`. A trainable layer of `Rot` (rotation) gates, followed by an entangling layer of `CNOT` gates, processes the quantum state.
    *   **Measurement:** The expectation value of the Pauli-Z operator for each qubit is measured, yielding a 4-element feature vector for each patch.
    *   **Vectorization:** This entire process is heavily vectorized to run all patches for a batch in parallel on the GPU, minimizing slow Python loops.
3.  **Classical Post-processing:** The output from the quantum layer is treated as a new feature map and is passed through a deeper classical CNN stack, consisting of multiple `Conv2d`, `GroupNorm`, and `MaxPool2d` layers, to learn higher-level features.
4.  **Classification Head:** A final `Linear` layer with a `Dropout` layer classifies the features into one of the 44 character classes.

### 🧪 Journey & Key Optimizations

The project evolved significantly to overcome two primary challenges: **extreme training slowness** and **stagnant learning (low accuracy)**.

* **Performance:** Initial training times were over 8 hours per epoch. We reduced this to ~1 hour by:
  * Reducing the number of quantum operations via the classical pre-processing layer.
  * Vectorizing the Quanvolutional layer to leverage GPU parallelism.
  * Switching to a high-performance GPU-accelerated quantum simulator (`lightning.gpu`).
  * Enabling a disk cache for compiled quantum kernels (`qml.transforms.dynamic_dispatch.enable_tape_cache()`).
* **Accuracy:** The model initially failed to learn. We addressed this by:
  * Implementing a robust diagnostic process to check gradient flow and quantum layer output variance.
  * Replacing `BatchNorm` with `GroupNorm` for more stable learning with small effective batch sizes.
  * Implementing a learning rate scheduler with a warm-up phase to prevent initial learning instability.
  * Refining the model architecture to have sufficient classical processing power.

### 📊 Current Status & Results

The current model (V6 architecture with 6x6 feature maps) uses:

* **Quantum Circuit Evaluations:** 9 per image (28.4x reduction from initial 256)
* **Training Time:** ~117 seconds per batch (~43% faster than V4)
* **Architecture:** 32x32 → 16x16 → 8x8 → 6x6 (classical preprocessing) → Quantum layer → Classical post-processing

For detailed experimental results and evolution history, see [experiments.md](experiments.md).

## 🚀 Getting Started

### Prerequisites

*   Python 3.9+
*   An NVIDIA GPU with CUDA support
*   Git

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
├── models/                        # Saved model checkpoints
├── src/                           # Core source code
│   ├── __init__.py                # Makes 'src' a Python package
│   ├── config.py                  # All hyperparameters and configuration
│   ├── dataset.py                 # Data loading and preprocessing
│   ├── model.py                   # Quantum circuit and hybrid model definition
│   ├── train.py                   # Main training and evaluation script
│   ├── trainable_quantum_model.py # Enhanced trainable quantum layers
│   └── enhanced_training.py       # Advanced training with separate optimizers
├── experiments/                   # Experimental scripts
│   └── run_experiments.py         # Automated experiment runner
├── improved_model.py              # Alternative model architectures
├── improved_training.py           # Training enhancements and optimizations
├── improved_quantum_circuit.py    # Enhanced quantum circuit designs
├── performance_optimizations.py   # Performance benchmarking utilities
├── experiments.md                 # Detailed log of all experiments and results
├── prd.md                         # Product Requirements & Journey Document
├── CLAUDE.md                      # Claude Code assistant instructions
├── QUANTUM_ML_RECOMMENDATIONS.md  # QML best practices and recommendations
├── IMPLEMENTATION_GUIDE.md        # Step-by-step implementation guide
├── requirements.txt               # Python dependencies
└── README.md                      # This file
``` 