# CUDA/data closure command and decision record

Investigation/access: 16 September 2026. Report/verification resumed 17 September 2026 after a usage-limit interruption. Working directory: repository root. No experiments were restarted after resumption.

## Repository and prior evidence

- `git rev-parse HEAD`, `git status --short`, `git diff --stat` and `git diff --binary`.
- Read the governing program, completed E0–E3 report, artifact inventory, manifests, protocol, E3 outputs and command record. Hashed the union of 4,404 protected old files, previous E0–E3 deliverables and tracked files: 5,075 paths; no saved-hash disagreements.
- Saved `before_state.json` and `before_tracked.patch` before new diagnostic outputs.
- Inspected notebook JSON outputs from `train_v7_colab.ipynb` and `colab_v7_rerun_clean.ipynb`; saved selected environment lines with zero-based cell/output indexes.
- `git show 126d396^:src/enhanced_training.py`, `git show 126d396^:src/trainable_quantum_model.py`, `git show 126d396^:src/config.py`; saved text snapshots.

## Main diagnostic

```bash
venv/bin/python scripts/investigate_dataset_closure.py
```

Exact executed source is saved as `executed_investigator_source.py`; config contains its SHA-256. To reproduce in a fresh output location:

```bash
venv/bin/python scripts/investigate_dataset_closure.py --output /private/tmp/astra-closure-reproduction
```

The output directory must have none of the script's output filenames. This command performs data/environment diagnosis only, never CUDA training. Native image distances are in 0–255 grayscale units. The config was written before comparisons. Four nearest-pair distances were independently recalculated in float64 while creating `adjudication_review.json`; foreground-mask semantics are corrected in that explicit addendum.

## Source research

Exact queries, URLs, access date and saved responses are in `source_catalog.json`.

- Read original Kaggle metadata through its public `/api/v1/datasets/view/alpbintuuzun/ottoman-turkish-characters` endpoint, and its public dataset search endpoint.
- Read version-1 ZIP into memory through the public download endpoint; save its SHA-256 and ZIP inventory. No files extracted into `set/`, no archive redistributed, no new image dataset created.
- Read dataset-author GitHub public repository listing. Search did not identify crop-to-page/writer metadata. Do not equate this bounded search with proof that such metadata cannot exist.
- Read authors' publisher article/PDF; save URL/hash/access evidence. PDF rendered from temporary storage with:

```bash
pdftoppm -f 3 -l 4 -scale-to 1500 -png /private/tmp/astra-authors-2021.pdf /private/tmp/astra-authors-page
pdftotext -layout 'Necati Incekara Master Thesis.pdf' /private/tmp/astra-closure-thesis.txt
```

- Visually inspected publication pages 583–584, conflict contact sheet, invalid-sample comparisons and all four near-duplicate pairs. Visual review is by Codex, not independent Ottoman paleographic adjudication.
- Checked PyTorch 2.10 AMP examples for stable future stepping requirements.

## Failed attempts and limits

1. Initial new investigator draft had a missing list bracket; it failed at Python parse time before any output. Fixed before execution.
2. Launch with default `python` failed because that interpreter lacked `cv2`. Used existing `venv/bin/python`; installed no dependencies.
3. Kaggle's web-rendered page exposed no readable body; its public API succeeded.
4. Web PDF screenshots timed out twice. Direct publisher retrieval and local rendering succeeded; Fontconfig emitted cache-write warnings, but rendered pages were inspected successfully.
5. Raw investigator fields named `ink_iou`/`ink_pixels_below_128` actually measure dark pixels. Original outputs and executed source are retained. `adjudication_review.json` supplies correctly identified white foreground measures; neither mask statistic selected candidates.
6. CUDA absent. No Level A/B/C CUDA run, backend operation probe, real V7 batch, model training or E2 rerun was launched. CPU E3 evidence was not rerun.

## Final verification

```bash
venv/bin/python scripts/verify_dataset_closure.py
```

Checks all protected hashes and tracked patch identity, new investigator/source identity, primary metadata class-map equality, quarantine exclusions, candidate pixel metrics, artifact/report references, 17 report sections and final stop text. Final identity and every new file's purpose/hash are recorded in `final_verification.json` and `artifact_inventory.json`.
