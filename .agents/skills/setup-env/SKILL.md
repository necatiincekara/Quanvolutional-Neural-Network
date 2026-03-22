---
name: setup-env
description: Set up or verify the development environment for local work or Google Colab training in this quantum ML repo.
---

# Setup Env

Use this skill when the task is environment setup, verification, or platform troubleshooting.

## Workflow

1. Detect the platform and Python version.
2. For local development, prefer CPU or platform-native acceleration if available.
3. For real training, prefer Colab or another CUDA environment for `lightning.gpu`.
4. Verify:
   - dependency imports
   - quantum backend availability
   - dataset paths in `src/config.py`
   - a small forward pass
5. Watch for known issues:
   - `numpy<2.0` constraints around PennyLane-era stacks
   - missing CUDA support on macOS
   - stale path assumptions pointing to Google Drive
6. Report the exact blocker and the minimum fix, not a generic setup dump.
