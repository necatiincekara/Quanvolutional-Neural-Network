# E0–E3 command record

Executed from `/Users/necatiincekara/Dev/Quanvolutional-Neural-Network` using the existing virtual environment. Experiment launches used `MPLCONFIGDIR=/private/tmp/astra-mpl XDG_CACHE_HOME=/private/tmp/astra-cache`.

## State inspection

- `git status --short`, `git diff --stat`, `git diff --numstat`
- `git diff -- src/dataset.py src/enhanced_training.py src/trainable_quantum_model.py train_v7.py scripts/aggregate_low_data.py scripts/statistical_evidence.py`
- Read the governing plan, prior script/evidence, source loader/training paths and cache metadata; compare all 140 prior manifest hashes.
- `git show 126d396^:src/enhanced_training.py`
- A standard-library hash snapshot preserved 4,404 existing files and `git diff --binary HEAD` in before_state.json / before_tracked.patch.

## Experimental launch order

```bash
venv/bin/python scripts/build_research_dataset_manifest.py --output-dir research/e0_e3_20260915
venv/bin/python scripts/replay_quantum_features.py --root research/e0_e3_20260915
# Above attempt failed on relative/absolute metadata path bookkeeping. Retained under e1/.
# Corrected path handling, then launched a fresh output directory:
venv/bin/python scripts/replay_quantum_features.py --root research/e0_e3_20260915 --output-name e1_verified
venv/bin/python scripts/test_research_foundation.py
venv/bin/python scripts/run_feature_probe.py --root research/e0_e3_20260915 --config research/configs/e2_pilot.json
venv/bin/python scripts/diagnose_hybrid_precision.py --root research/e0_e3_20260915 --config research/configs/precision_screen.json
venv/bin/python scripts/verify_research_execution.py --root research/e0_e3_20260915
venv/bin/python scripts/stress_hybrid_scaling.py --root research/e0_e3_20260915
```

## Non-experimental evidence operations

- Corrected E1 identity metadata additively in `e1_verified/run_corrected.json`, retaining the original and its hash. Historical occurrence IDs were already stored; no numeric metric was altered and no accuracy sweep followed a feature mismatch.
- Saved hash-matched executed source versions in source_snapshots, including both E1 script versions.
- Recomputed stored prediction metrics, independently reloaded two E2 checkpoints, checked schemas and all protected hashes, and parsed new Python sources.
- Saved final status and artifact inventory. No commit, push, deployment or paid hardware call.

## Reproduction caveats

Outputs use exclusive creation and refuse overwrite. Reproduce in a clean copy with a fresh execution directory and the saved config/source versions. The foundation test and several metadata path strings identify this particular campaign; adjust paths explicitly in a new protocol rather than modifying these evidence files. Original requirements and installed environment were not changed. CUDA/MPS were unavailable; all measurements are CPU observations.
