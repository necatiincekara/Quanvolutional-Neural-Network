---
name: gradient-check
description: Diagnose vanishing or exploding gradients in the hybrid quantum-classical training path. Use especially for V6 collapse, trainable quantum instability, or AMP-related failures.
---

# Gradient Check

Use this skill when learning stalls or quantum parameters look inactive.

## Workflow

1. Inspect:
   - `src/model.py`
   - `src/trainable_quantum_model.py`
   - `src/enhanced_training.py`
   - any active training script
2. Verify where gradients should flow and where dtype conversions happen.
3. Compare quantum vs classical gradient magnitudes.
4. Flag:
   - vanishing gradients
   - exploding gradients
   - barren plateau-like behavior
   - AMP or dtype boundary issues
5. Give concrete fixes in priority order:
   - residual paths
   - gradient scaling
   - optimizer separation
   - circuit depth or initialization changes
   - precision boundaries around the quantum layer

Repository note: the newer trainable path already introduced float32 boundaries and stabilization logic; check that path before assuming the base model behavior still applies.
