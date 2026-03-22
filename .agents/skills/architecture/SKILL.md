---
name: architecture
description: Design or analyze model architecture for this Ottoman QML study. Use for Turkish or English requests about V7-V10 ideas, bottlenecks, layer flow, parameter tradeoffs, or implementation planning.
---

# Architecture

Use this skill when the task is to design a new model version or analyze an existing one.

## Workflow

1. Read `AGENTS.md` first so you do not repeat stale claims from `CLAUDE.md` or `docs/AUDIT_REPORT.md`.
2. Inspect the relevant implementation files:
   - `src/model.py`
   - `src/trainable_quantum_model.py`
   - `src/thesis_models.py`
   - `src/benchmark_protocol.py`
   - `src/enhanced_training.py`
   - `train_ablation_local.py` or `train_thesis_models.py`
   - `improved_model.py` if the task touches alternative designs
3. Reconcile the design against current study reality:
   - local ablations currently outperform the documented V7 result
   - feature maps below `8x8` caused major information loss or collapse in prior work
4. Produce:
   - a short architecture diagram
   - bottlenecks and tradeoffs
   - parameter/runtime implications
   - the smallest defensible implementation plan
5. If code changes are requested, modify the minimal set of files and explain how the change affects training behavior.
