---
name: sync-colab
description: Manage code and result synchronization between the local repo and Google Colab without losing checkpoints or experiment metadata.
---

# Sync Colab

Use this skill for local-to-Colab or Colab-to-local synchronization work.

## Workflow

1. Check git state before suggesting sync steps.
2. Decide the direction:
   - local to Colab code push
   - Colab to local result pull
   - Colab environment bootstrap
3. Preserve large artifacts outside git when appropriate:
   - checkpoints
   - notebook outputs
   - large logs
4. After pulling new results back, reconcile them against `docs/EXPERIMENTS.md` before updating any narrative files.
5. Prefer explicit instructions over vague "sync everything" advice.
