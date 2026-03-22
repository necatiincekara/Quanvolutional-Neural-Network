---
name: experiment
description: Plan or run an experiment for this study, including ablations, circuit comparisons, gradient diagnostics, and result logging.
---

# Experiment

Use this skill when the task is to run, plan, or interpret an experiment.

## Workflow

1. Read `AGENTS.md` and inspect the current experiment entrypoints:
   - `train_v7.py`
   - `train_ablation_local.py`
   - `src/train.py`
   - `src/enhanced_training.py`
2. Check whether older automation such as `experiments/run_experiments.py` is actually compatible before using it.
3. Define the experiment type:
   - baseline comparison
   - circuit comparison
   - ablation study
   - gradient-flow check
   - reproducibility rerun
4. Record the exact config, platform, and dataset split assumptions before running long jobs.
5. After the run, summarize:
   - primary metrics
   - training stability
   - deviations from expected behavior
   - whether docs now need updating
6. If the run produced meaningful results, follow up with the `log-result` skill.
