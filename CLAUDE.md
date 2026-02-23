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
- `docs/AUDIT_REPORT.md`: Comprehensive codebase audit and development roadmap
- `docs/EXPERIMENTS.md`: Detailed log of all experimental results (V1-V6)
- `docs/QUANTUM_ML_RECOMMENDATIONS.md`: Best practices and optimization recommendations
- `docs/IMPLEMENTATION_GUIDE.md`: Step-by-step guide for V7-V10 development
- `docs/RESEARCH_ROADMAP.md`: Publication strategy and research timeline

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

### Performance Benchmarks & Current Status

| Version | Feature Map | Quantum Calls | Epoch Time | Accuracy | Status |
|---------|------------|---------------|------------|----------|---------|
| V1 | 32×32 | 256 | >8h | 2.3% | ❌ Infeasible |
| V4 | 8×8 | 16 | ~1.5h | 8.75% | ✅ **Stable Baseline** |
| V6 | 6×6 | 9 | ~117s/batch | 0.00% | ❌ Gradient vanishing |

**Current Best**: V4 (93.75% reduction in quantum evaluations from V1)
**Critical Issue**: V6 achieves 43% speedup but suffers from complete gradient collapse
**Next Steps**: V7 gradient stabilization (see docs/AUDIT_REPORT.md)

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
- **V6 gradient vanishing (0% accuracy)**: Aggressive spatial reduction without gradient stabilization causes complete gradient collapse; requires residual connections and gradient scaling (V7 solution)
- **Information bottleneck**: Feature maps <8×8 lose critical spatial information; V4 (8×8) is current optimal
- **Barren plateaus**: Use separate learning rates for quantum (0.001) and classical (0.005) parameters with gradient clipping

### Development Priorities (2025)

**Immediate** (Week 1-2):
- Implement V7 gradient stabilization techniques
- Add gradient monitoring to all training loops
- Set up Weights & Biases experiment tracking

**Short-term** (Month 1):
- Achieve 25% accuracy with V7
- Implement V8 multi-scale processing
- Complete comprehensive ablation studies

**Medium-term** (Month 2-3):
- Deploy V9 selective quantum processing (target: 60% accuracy)
- Implement V10 fully trainable quantum circuits (target: 90% accuracy)
- Prepare publication materials

For detailed roadmap, see `docs/AUDIT_REPORT.md` and `docs/IMPLEMENTATION_GUIDE.md`