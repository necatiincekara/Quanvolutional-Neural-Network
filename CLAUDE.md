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
- `src/model.py`: Quantum circuit definition and hybrid model architecture
- `src/train.py`: Training loop with mixed precision, gradient scaling, and checkpoint management
- `src/dataset.py`: Data loading for Ottoman character images
- `src/config.py`: Centralized configuration for hyperparameters and paths

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
- Original implementation: >8 hours per epoch
- Current optimized version: ~1-1.5 hours per epoch
- Key reduction: From 256 to 16-36 quantum circuit evaluations per image

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

### Known Issues & Solutions
- **Slow first epoch**: Quantum kernel compilation happens on first run; subsequent epochs are faster due to disk caching
- **Learning stagnation**: Check gradient flow diagnostics and ensure learning rate scheduler is stepped per batch
- **Memory issues**: Reduce batch size or feature map dimensions before quantum layer