---
name: train
description: Orchestrate training for the baseline or enhanced hybrid models in this repo, including checkpoint handling and platform-aware execution choices.
---

# Train

Use this skill when the user wants to start, resume, or configure training.

## Workflow

1. Inspect the relevant training path first:
   - `python -m src.train` for the older baseline
   - `python train_v7.py` for the enhanced trainable-quantum path
   - `train_ablation_local.py` for local ablation comparisons
2. Check checkpoints and recent results before launching another run.
3. Confirm the execution platform assumptions:
   - macOS or CPU work is mainly for development and debugging
   - CUDA or Colab is preferred for serious training
4. Watch for repo-specific caveats:
   - `train_v7.py --target` is currently not fully wired through
   - stale experiment scripts may not match current trainer APIs
5. During or after training, summarize:
   - metrics
   - runtime behavior
   - gradient health
   - whether the run changes the current study conclusion
