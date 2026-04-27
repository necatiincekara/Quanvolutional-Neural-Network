# Colab V7 Handoff

**Date:** April 6, 2026

This document is the operational handoff for the trainable V7 Colab path.

> Update, April 23, 2026:
> One fresh L4 rerun has already been completed successfully at `72.89%` best validation and `72.53%` test accuracy.
> Remaining Colab budget is approximately `245` computing units.
> Therefore this document should now be read mainly as a resume / recovery / artifact-sync reference, not as an instruction to start another default Colab run.

## 1. Goal

The goal is not to prove a new benchmark lead. The goal is to produce a fresh, reproducible,
artifact-backed V7 rerun that can be cited safely in the paper as the current trainable-quantum
case-study result.

## 2. Platform Choice

- Preferred platform: Colab GPU
- First choice: L4 if available
- Second choice: A100 if available and unit cost is acceptable
- Do not run this rerun on the local Mac as the primary path

## 3. Pre-Flight Checks

Before running training on Colab:

1. Mount Google Drive.
2. Verify GPU is available and `torch.cuda.is_available()` is `True`.
3. Keep the dataset on Drive as the source of truth, but copy it to local Colab disk for speed.
4. Expect the fast local paths to be:
   - `/content/local_data/train`
   - `/content/local_data/test`
5. After copy, verify the raw folder counts:
   - train directory: `3428` files on disk
   - test directory: `466` files on disk
6. Remember that one malformed training filename is skipped by the loader, so effective training
   examples are `3427`.

Recommended staging commands:

```bash
mkdir -p /content/local_data/train /content/local_data/test
rsync -a /content/drive/MyDrive/set/train/ /content/local_data/train/
rsync -a /content/drive/MyDrive/set/test/ /content/local_data/test/
find /content/local_data/train -type f | wc -l
find /content/local_data/test -type f | wc -l
```

## 4. Exact Training Commands

Fresh run from scratch:

```bash
python train_v7.py \
  --circuit data_reuploading \
  --epochs 10 \
  --target 60 \
  --drive-backup-path /content/drive/MyDrive/quanv_results/v7 \
  --train-path /content/local_data/train \
  --test-path /content/local_data/test
```

Resume from latest checkpoint for the same target regime:

```bash
python train_v7.py \
  --circuit data_reuploading \
  --epochs 10 \
  --target 60 \
  --resume \
  --drive-backup-path /content/drive/MyDrive/quanv_results/v7 \
  --train-path /content/local_data/train \
  --test-path /content/local_data/test
```

Continue past an earlier `--target 60` stop without starting from scratch:

```bash
python train_v7.py \
  --circuit data_reuploading \
  --epochs 10 \
  --target 90 \
  --resume \
  --drive-backup-path /content/drive/MyDrive/quanv_results/v7 \
  --train-path /content/local_data/train \
  --test-path /content/local_data/test
```

## 5. What The CLI Now Supports

As of April 6, 2026, `train_v7.py` now supports:

- dataset path overrides with `--train-path` and `--test-path`
- checkpoint resume with `--resume`
- persistent Drive checkpoint backup with `--drive-backup-path`
- explicit target threading with `--target`

This makes the Colab rerun operable from a shell cell rather than only through ad hoc notebook edits.

## 6. Artifact Expectations

During the run, the following outputs matter:

- `models/checkpoint_latest_v7.pth`
- `models/best_v7_model.pth`
- `experiments/v7_<circuit>_<timestamp>/best_model.pth`
- `experiments/v7_<circuit>_<timestamp>/metadata.json`
- `experiments/v7_<circuit>_<timestamp>/training_curves.png`

If `--drive-backup-path` is set, the backup path should also receive:

- `checkpoint_latest_v7.pth`
- `best_v7_model.pth`

## 7. Post-Run Reconciliation

After the Colab run finishes:

1. Copy the relevant `models/` and `experiments/` artifacts back to the local repo workspace.
2. Do not update paper claims from terminal memory alone.
3. Reconcile the new run against:
   - `experiments/*.json`
   - `docs/EXPERIMENTS.md`
   - `docs/BENCHMARK_SUMMARY.md`
   - `paper/draft.md`
4. If the fresh V7 rerun still does not change the main ranking, keep V7 framed as:
   - a trainable-quantum engineering case-study
   - not the benchmark leader

At the current repository state, this post-run reconciliation is still incomplete: the benchmark tables and paper already reflect the fresh rerun metrics, and the Drive-backed checkpoint files are now synced locally, but the remote `experiments/v7_*` directory has not yet been recovered into the local repo workspace.

## 8. Stop Conditions

Stop after the first clean artifact-backed rerun if any of the following hold:

- no NaN failure occurs
- checkpoints and metadata are captured correctly
- the new result does not materially change the current benchmark hierarchy

Only consider extra V7 seeds after this first rerun if the new run is both stable and paper-relevant, and if the expected value clearly justifies the remaining `245` CU budget. The current default is to spend zero additional Colab units until artifact sync and manuscript tightening are complete.
