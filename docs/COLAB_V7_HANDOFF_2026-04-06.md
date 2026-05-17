# Colab V7 Handoff

**Date:** April 6, 2026

This document is the operational handoff for the trainable V7 Colab path.

> Update, April 30, 2026:
> The April 6 resumed L4 rerun completed successfully at `72.89%` best validation and `72.53%` test accuracy.
> The April 27 clean non-resumed L4 rerun completed at `69.97%` best validation and `65.88%` test accuracy; its JSON row is reconstructed from captured notebook output because the runtime disconnected before the artifact-copy cell ran.
> The last user-reported Colab budget before the May 2026 low-data confirmation was approximately `245` computing units; remeasure the current balance before any new Colab work.
> Therefore this document should now be read mainly as a resume / recovery / artifact-sync reference, not as an instruction to start another default V7 Colab run.

> Update, May 16, 2026:
> V7 is closed as a trainable-quantum engineering case-study unless reviewer feedback specifically demands another controlled rerun.
> Do not spend Colab units on V7 folder hygiene; the current paper-impact path is submission cleanup or a separately justified robustness/statistics extension.

## 1. Goal

The historical goal was not to prove a new benchmark lead. It was to produce a fresh, reproducible,
artifact-backed V7 rerun that could be cited safely in the paper as a trainable-quantum
case-study result.

At the current repository state, that goal has been satisfied by the April 2026 reruns and reconstructed JSON rows. Use this document for resume/recovery instructions only.

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
4. If a V7 rerun still does not change the main ranking, keep V7 framed as:
   - a trainable-quantum engineering case-study
   - not the benchmark leader

At the current repository state, this post-run reconciliation should preserve the provenance labels: the benchmark tables and paper reflect the April rerun metrics, the April 27 Drive folder contains checkpoint files, and its `experiments/` subfolder is empty because the artifact-copy cell did not run before runtime disconnect.

## 8. Stop Conditions

Stop after the first clean artifact-backed rerun if any of the following hold:

- no NaN failure occurs
- checkpoints and metadata are captured correctly
- the new result does not materially change the current benchmark hierarchy

Only consider extra V7 seeds if reviewer feedback makes them paper-relevant and if the expected value clearly justifies the Colab spend. The current default is to avoid more V7 training for artifact hygiene; any future Colab experiment should address a concrete reviewer risk or stronger venue target.
