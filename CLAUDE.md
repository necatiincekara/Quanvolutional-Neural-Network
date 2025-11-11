# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Hybrid Quantum-Classical Convolutional Neural Network implementation for Ottoman-Turkish handwritten character recognition. The project combines PyTorch with PennyLane quantum machine learning framework to create a "Quanvolutional" layer that processes image features through quantum circuits.

## Architecture & Key Components

### Core Structure
- **Quantum Layer (`QuanvLayer`)**: A vectorized quantum convolutional layer using 4-qubit circuits with angle embedding and trainable rotation gates
- **Hybrid Model (`QuanvNet`)**: Combines classical CNN preprocessing, quantum feature extraction, and classical post-processing
- **GPU Acceleration**: Uses `lightning.gpu` quantum simulator for performance (requires CUDA-enabled GPU)

### Key Files

#### Core Implementation
- `src/model.py`: Quantum circuit definition and hybrid model architecture
- `src/train.py`: Training loop with mixed precision, gradient scaling, and checkpoint management
- `src/dataset.py`: Data loading for Ottoman character images
- `src/config.py`: Centralized configuration for hyperparameters and paths

#### Enhanced Implementations
- `src/trainable_quantum_model.py`: Advanced trainable quantum circuits with data re-uploading
- `src/enhanced_training.py`: Separate optimizers for quantum/classical parameters
- `improved_model.py`: Alternative model architectures and enhancements
- `improved_training.py`: Training optimizations and performance benchmarks
- `improved_quantum_circuit.py`: Enhanced quantum circuit designs
- `performance_optimizations.py`: Performance measurement and optimization utilities

#### Experiments & Documentation
- `experiments/run_experiments.py`: Automated experiment runner for ablation studies
- `experiments.md`: Detailed log of all experimental results
- `QUANTUM_ML_RECOMMENDATIONS.md`: Best practices and optimization recommendations
- `IMPLEMENTATION_GUIDE.md`: Step-by-step guide for achieving 90% accuracy target

## Common Development Commands

### Training
```bash
# Start new training from scratch
python -m src.train

# Resume from latest checkpoint
python -m src.train --resume

# Monitor training (if tensorboard is configured)
tensorboard --logdir=runs
```

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Critical Implementation Details

### Quantum Circuit Optimization
The quantum layer has been heavily optimized through multiple iterations:
- Classical preprocessing reduces input from 32x32 to 8x8 or 6x6 feature maps
- Vectorized batch processing of quantum circuits using PyTorch unfold/reshape operations
- Disk caching enabled via `qml.transforms.dynamic_dispatch.enable_tape_cache()`
- Adjoint differentiation method for better GPU performance

### Training Considerations
- **Learning Rate Schedule**: Uses warmup phase with LambdaLR scheduler (stepped per batch, not epoch)
- **Normalization**: GroupNorm instead of BatchNorm for stability with small effective batch sizes
- **Mixed Precision**: AMP (Automatic Mixed Precision) enabled for GPU training
- **Checkpointing**: Saves best model to `models/best_quanv_net.pth` and latest state to `models/checkpoint_latest.pth`

### Performance Benchmarks
- Original implementation (V1): >8 hours per epoch, 256 quantum evaluations/image
- V4 architecture (8x8 feature maps): ~1.5 hours per epoch, 16 quantum evaluations/image
- V6 architecture (6x6 feature maps): ~117 seconds/batch, 9 quantum evaluations/image (~43% faster than V4)
- Key optimization: From 256 to 9 quantum circuit evaluations per image (28.4x reduction)

### Dataset Configuration
The model expects Ottoman character images organized in folders by class label. Default paths point to Google Drive locations (see `config.py`). When running locally, update:
- `TRAIN_PATH`: Training data directory
- `TEST_PATH`: Test data directory

### Debugging Features
The training script includes debug outputs for monitoring quantum layer behavior:
- Quantum output standard deviation (`q_out std`)
- Gradient magnitudes after backward pass
- Per-batch loss and accuracy metrics

## Important Notes

### Quantum Device Selection
- Development/CPU: Set `QUANTUM_DEVICE = 'default.qubit'` in config.py
- Production/GPU: Use `QUANTUM_DEVICE = 'lightning.gpu'` (requires pennylane-lightning-gpu)

### Model Variations
The repository tracks multiple experimental architectures (V1-V6) with different preprocessing strategies. See `experiments.md` for detailed performance comparisons and architectural choices.

**Current Best Architecture (V6):**
- Classical preprocessing: 32x32 → 16x16 → 8x8 → 6x6
- Quantum layer: Processes 6x6 feature map (9 quantum circuit evaluations)
- Classical post-processing: Deep CNN with GroupNorm and Dropout

**Alternative Implementations:**
- See `improved_quantum_circuit.py` for enhanced circuit designs with data re-uploading
- See `trainable_quantum_model.py` for trainable quantum layers targeting higher accuracy
- See `IMPLEMENTATION_GUIDE.md` for roadmap to achieve 90% accuracy

### Known Issues & Solutions
- **Slow first epoch**: Quantum kernel compilation happens on first run; subsequent epochs are faster due to disk caching
- **Learning stagnation**: Check gradient flow diagnostics and ensure learning rate scheduler is stepped per batch
- **Memory issues**: Reduce batch size or feature map dimensions before quantum layer
- **V6 learning failure (0% accuracy)**: Extreme spatial reduction (4x4) creates information bottleneck; 6x6 is minimum viable size
- **Barren plateaus**: Use separate learning rates for quantum (0.0001) and classical (0.0005) parameters