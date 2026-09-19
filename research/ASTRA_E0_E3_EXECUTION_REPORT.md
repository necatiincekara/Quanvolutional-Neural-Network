# Astra E0–E3 execution report

**Date:** 15 September 2026. **Governing plan:** [ASTRA_QUANTUM_RESEARCH_PROGRAM.md](ASTRA_QUANTUM_RESEARCH_PROGRAM.md). **Scope:** E0, E1, E2 and E3 only. All execution used the local CPU; no paid hardware, accelerator campaign, powered confirmation, V8+ model, or manuscript rewrite was performed.

## 1. Executive result

Three scientific conclusions are now substantially better supported.

**The existing fixed quantum transformation is exactly executable classically at repository level.** Six filter banks, all locally available corresponding train/test caches, and 27 historical checkpoints were checked. Analytic and reference inputs produced identical downstream logits and predictions. This closes irreducible quantum-computation claims for the current noiseless RX/Rot/forward-CNOT-chain/Z-expectation map. It does not establish the same product formula for V7.

**The low-data signal does not currently justify a powered confirmation.** A common-pipeline pilot selected random Fourier features (RFF) as the strongest screened classical control. On a separate validation assessment partition, the analytic fixed-Q map's macro-F1 learning-curve area was **1.46 points lower** on average. The crossed-factor 95% bootstrap interval was **[−3.52, +0.35]**; adding evaluation-image resampling in a secondary sensitivity check gave **[−3.62, +1.13]**. Neither supports the prespecified +2-point quantum-derived benefit. This is an exploratory, conditional result, not proof that RFF always wins or that every possible quantum feature map is useless.

**The numerical failure has separable causes.** The historical direct optimizer step bypasses GradScaler's nonfinite-step protection even after gradients have been unscaled. Controlled tests corrupted both analytic/classical and quantum parameters under that misuse; correct scaler stepping prevented corruption. Separately, this CPU runtime cannot execute a float16 complex exponential used in quantum rotation construction. A float32 Q boundary resolves that CPU support error, but does not by itself prevent scaled gradients from overflowing elsewhere in a float16 network. The original L4/CUDA overflow trigger remains unresolved because that hardware/runtime was unavailable.

E0 also exposed a material foundation issue: **72 duplicate groups, including 15 crossing the historical train/test boundary and five with conflicting labels.** Historical files and scores were preserved. The new pilot uses a separately named, content-deduplicated protocol.

## 2. Repository state

Before execution, HEAD was `95d6fb85f27c394e4b33b2cd5e5321c0c61e724a` on `master`, with 38 tracked modified files and extensive untracked prior work. The full expanded status had 205 entries. The four previous Astra files were present; all 140 inputs in the prior audit manifest still matched their saved hashes. The governing plan was read from the current workspace and retained unchanged.

Initial stage status was:

| Stage | State before this task |
|---|---|
| E0 | Audit input hashes existed; deterministic sample manifest, new split protocol and run schema did not |
| E1 | Small source-AST output/Jacobian verification was complete; cache/checkpoint replay was absent |
| E2 | Not started |
| E3 | Not started |

The original small E1 diagnostic was not rerun merely to repeat it. Its source identity was checked, and its extraction method was reused to validate the new analytic implementation on repository data.

The [before-state record](e0_e3_20260915/before_state.json) and [tracked patch](e0_e3_20260915/before_tracked.patch) preserve the starting status and tracked changes. At completion HEAD is unchanged, and the tracked diff is unchanged. Hash verification found **zero changes to 4,404 protected pre-existing files**, including data, checkpoints, cached features, prior Astra evidence, manuscript files and old experiment artifacts. New work is additive under the paths listed below. No commit or push was made.

The [final state record](e0_e3_20260915/after_state.json) records completion-state identity. The [artifact inventory](e0_e3_20260915/artifact_inventory.json) enumerates new deliverables with hashes and purposes. These files are local, uncommitted evidence; no remote archival durability is claimed.

## 3. E0 results

The [deterministic dataset manifest](e0_e3_20260915/dataset_manifest.json) records every source file's relative path, original folder split, byte hash, native decoded-pixel hash, content-based sample ID, occurrence/record ID, label and explicit exclusion reason. Every image is 32×32. Rebuilding the manifest yielded identical content and checksum.

| Finding | Result |
|---|---:|
| Historical training folder | 3,428 files; 3,427 valid |
| Historical test folder | 466 files; 466 valid |
| Invalid label/file records | One unknown `00` label; no image decode failures |
| Exact byte duplicate groups | 72 |
| Decoded-pixel duplicate groups | 72 |
| Duplicate groups crossing source train/test folders | 15 |
| Decoded duplicate groups with conflicting valid labels | 5 |
| New protocol training examples | 2,688 |
| New protocol validation examples | 668 |
| New protocol sealed test examples | 458 |

The invalid record is `set/train/TBMM-007-00000000000000000000000000000000000.png`. It remains on disk and in the manifest. Its pixels were hashed even though its label is invalid.

The new protocol quarantines content groups with conflicting labels, keeps test-overlapping content out of new training, and selects one deterministic representative of duplicate content. Eleven nonconflicting overlap groups receive the explicit “excluded from new train” reason; overlaps that also have conflicting labels are already quarantined. A deterministic, per-class 20% validation selection leaves at least one training example per class. All three resulting splits cover all 44 classes. This is **content grouping**, not writer grouping: writer/source identities and near-duplicate relationships remain unverified.

The [historical membership record](e0_e3_20260915/historical_membership.json) separately preserves all original folder occurrences and today's observed local loader order. It does not pretend to reconstruct unrecorded historical remote train/validation membership. The [new protocol](e0_e3_20260915/protocol.json) is named `astra_deduplicated_validation_pilot_v1`; no existing benchmark is relabeled or silently corrected. Its scores must not be pooled with the old publication or low-data results.

The new [run schema](schemas/run.schema.json) covers experiment/architecture identity, commit/dirty state and source hashes, dataset/protocol checksums, exact split IDs, separate seed fields, hyperparameters, parameter counts, optimizer/scheduler, augmentation, precision, backend/differentiation, versions/hardware/runtime, checkpoint criterion, epoch/gradient records, prediction paths and final class-aware metrics. A small fail-closed validator implements the schema subset used here; it is not advertised as a general JSON-Schema implementation. Semantic checks reject content leakage, missing completed-training evidence and final-test evaluation from the pilot training runner.

Four foundation tests passed before the E2 training pilot: duplicate/conflict quarantine and deterministic rebuilding; patch/feature layout and deterministic banks; rejection of incomplete/leaky/test-access training records; and metric/exclusive-write checks. [E0 evidence](e0_e3_20260915/e0_checks.json) and [verification evidence](e0_e3_20260915/verification.json) contain the results. Dataset rights and provenance are recorded as unresolved rather than inferred; no redistribution was performed.

## 4. E1 results

The analytic implementation computes, for each four-value patch and each filter,

`z_i = cos(theta_i) cos(x_i) − sin(theta_i) sin(phi_i) sin(x_i)`

followed by the four prefix products of `z`. Patch rows, patch columns, within-patch coordinates, filter order and output-wire order reproduce the historical implementation. Inputs first receive the historical float32 image normalization; the reference-equivalent analytic feature computation uses float64 angles/arithmetic and then stores float32 outputs.

The actual quantum function is extracted from `train_ablation_local.py` through its AST. For each bank, every patch in 16 distributed training and 16 distributed test images is compared against that function. All three available bank caches (42–44) are compared in full for both training and test. For banks 45–47, the entire test feature tensor is additionally regenerated through the quantum function because a historical local cache is absent. The downstream replay covers three full-data plus 24 low-data fixed-Q checkpoints.

| Check | Coverage / result | Declared tolerance |
|---|---|---|
| Quantum versus analytic float64 | Six banks; 196,608 selected-image patch/filter instances, plus full test generation for banks without caches; max error **1.22×10⁻¹⁵** | Absolute 10⁻¹⁰ |
| New analytic input/angle Jacobians | 24 filter cases on real patches; all passed | Absolute 10⁻¹⁰ |
| Native float32 analytic versus float64 | Max error **1.86×10⁻⁷** | Absolute 5×10⁻⁷ |
| Full cached representations | Six train/test tensors, banks 42–44; max error **7.11×10⁻¹⁵** | Absolute 2×10⁻⁷ |
| Cached labels/order | All checked arrays exactly matched current local historical ordering | Exact equality |
| Downstream logits | 27 checkpoints; max difference **0** | `atol=3e-5`, `rtol=1e-5` |
| Predictions | All 27 × 466 predictions identical | Exact equality |
| Reported historical accuracy | All replayed accuracies matched source JSON rounding | ≤0.011 percentage points |

The float32 tolerances allow small rounding differences of bounded expectation values; float64/Jacobian checks are much tighter. Logit tolerances allow numerical amplification in the saved classical head, but observed logits were identical. The accuracy tolerance is far below one test-example change (approximately 0.215 points); exact prediction agreement is the stronger categorical check.

The [replay results](e0_e3_20260915/e1_verified/replay_results.json) include feature/cache/checkpoint hashes and full class-aware metrics. All prediction archives retain occurrence IDs, targets and both logit arrays. The authoritative replay identity record is [run_corrected.json](e0_e3_20260915/e1_verified/run_corrected.json): an explicit metadata-only addendum fixes an initial generic run template that listed future-pilot IDs instead of the actual historical occurrence IDs. The original record is retained, its hash is referenced, and numerical results were not altered.

**Claim closure:** this fixed noiseless measured transformation is classically executable; its useful predictions do not require quantum computation. This is stronger than an approximate surrogate tie. The result does not cover V7 re-uploading, a different observable set, arbitrary noise, or every quantum feature family. Historical remote training membership and missing raw V7 artifacts remain unknown. The separate thesis amplitude-encoding map was not part of this requested fixed-four-qubit replay and remains an optional future control.

## 5. E2 results

### Design and separation from historical results

The [prespecified configuration](configs/e2_pilot.json) was saved before training. All seven feature families received the same raw images, 2×2 non-overlapping patch extraction, 16 output channels, average pooling to 4×4, a 256→44 linear readout, train-only initial feature standardization, no augmentation, AdamW (`lr=0.003`, weight decay `0.0001`), batch size 128 and 40 epochs. Checkpoints were selected by tuning-partition 44-class macro-F1, with earliest ties retained. There was no learning-rate or architecture sweep.

The controls were the exact analytic fixed-Q map, frozen random convolution/tanh, orthogonal random features, random Fourier features, a nonlinear random MLP, polynomial/product features, and a learned patch convolution. The learned convolution is the **raw-patch 1→16 counterpart** of the existing convolution replacement, with 80 map parameters. The historical full classical network was not inserted into this common pipeline because its learned stem/readout would reintroduce the confound. Every frozen-map arm uses the same 11,308-parameter readout; the learned map adds 80 parameters. Frozen coefficient counts are recorded separately.

The 668 validation examples were split into **325 tuning** and **343 assessment** examples by a fixed, class-aware rule. The assessment set covers all 44 classes, but 23 classes have five or fewer examples and the minimum support is one. Final test pixels/labels were not loaded by E2. Both partitions' IDs are stored in [validation_partition.json](e0_e3_20260915/e2/validation_partition.json).

Screening used two predeclared feature-bank seeds, one subset seed, one training seed, and three fractions (10%, 25%, 100%): 42 inexpensive fits. RFF had the highest mean tuning macro-F1 log-count ALC, 55.04 versus 52.19 for analytic fixed-Q. That selection was saved before assessment evaluation. The main pilot then crossed **three new subset seeds × three new bank seeds × two training seeds**, comparing only analytic fixed-Q with selected RFF at the three fractions: 108 fits. Total: **150 small readout/map fits**, not 150 quantum-training runs. The 25% middle point was included before outcomes to make ALC less dependent on two endpoints.

Training-seed variation includes readout initialization and shuffle variation; the initialization seed is separately recorded as a deterministic offset, but these two sources are not independently crossed. Bank and subset randomness are independently crossed. At 100% the subset seeds select identical examples; these repeated cells are not treated as extra independent data subsets. The fits are cheap, but 24 of the full-data pilot fits repeat an identical bank/training configuration across nominal subset seeds; future runners can reuse those outputs.

### Results

The endpoint is macro-F1 ALC integrated over log **actual training count**, normalized by the log-count range. All class metrics are on the untouched-by-selection assessment partition, not the historical test set.

| Actual training count | Analytic fixed-Q macro-F1 | RFF macro-F1 | Analytic fixed-Q accuracy | RFF accuracy |
|---:|---:|---:|---:|---:|
| 270 (nominal 10%) | 45.21 ± 4.55 | 47.33 ± 3.59 | 59.54 ± 2.34 | 61.55 ± 2.35 |
| 669 (nominal 25%) | 56.28 ± 2.30 | 57.26 ± 3.05 | 68.06 ± 1.40 | 68.79 ± 2.39 |
| 2,688 (100%) | 71.13 ± 2.23 | 72.94 ± 2.35 | 76.19 ± 1.20 | 77.26 ± 1.33 |

These are means ± descriptive SD across 18 crossed cells at the first two counts and six distinct bank/training configurations at full data. They are not independent-seed confidence intervals. Balanced accuracy likewise favored RFF on the mean: 49.52 vs 47.50, 58.53 vs 57.92, and 72.82 vs 71.49.

| Primary pilot summary | Result |
|---|---:|
| Analytic fixed-Q macro-F1 ALC | 58.59 |
| RFF macro-F1 ALC | 60.04 |
| Paired analytic fixed-Q − RFF difference | **−1.46 points** |
| Crossed subset/bank/training bootstrap interval | **[−3.52, +0.35]** |
| Secondary paired image + crossed-factor interval | **[−3.62, +1.13]** |
| Accuracy ALC, analytic fixed-Q / RFF | 68.84 / 69.92 |
| Prespecified worthwhile quantum-derived advantage | +2 points macro-F1 ALC |

[Primary evidence](e0_e3_20260915/e2/pilot_summary.json); [secondary uncertainty and fraction metrics](e0_e3_20260915/e2/secondary_uncertainty.json).

The finite-grid variance decomposition of paired ALC differences is 1.770 bank, 0.932 subset, 0.015 training, and 0.814 interaction variance units (points²). Approximately half of the tested-grid variability comes from bank main effects. These are **descriptive functional components of this small crossed grid**, not precise population variance estimates. Three bank/subset levels and two training levels cannot establish a reliable universal variance model.

The primary bootstrap resamples crossed factors and conditions on assessment examples. The secondary analysis also resamples assessment images in pairs; it was added after the main result as a sensitivity check, did not change model selection, and assumes independent examples. Unknown writer/near-duplicate groups and sparse-class bootstrap behavior remain limitations. Neither interval is a confirmatory significance claim.

**Decision: NO-GO for a powered low-data confirmation now.** The effect is in the wrong direction on average, fails the +2-point gate, and is not rescued by evaluation-image uncertainty. The result supports a classical explanation/control of the exploratory feature benefit, within this common small-readout pipeline. It does not establish practical equivalence across all models or universal classical superiority.

An automatic illustrative normal calculation stored in the summary returns eight independent pairs to detect a 2-point effect if one naively borrows the cell SD. **That is not a recommended sample size:** the 18 cells share factors and the mean does not favor the hypothesis. A future positive mechanism would require a new variance-component power design; no powered confirmation was run here.

## 6. E3 results

The diagnostic is a linear 4→4 classical stem, the actual source-extracted fixed circuit or V7 circuit (or the exact analytic fixed map), and a linear 4→3 head. It uses eight fixed training-image patches and synthetic three-class targets. It does not instantiate or train the full V7 OCR network. The models have 47 parameters for fixed/analytic and 59 for V7.

The [screen configuration](configs/precision_screen.json) contains five initialization seeds, nine necessary settings and ten updates per cell: 135 tiny cells. Per-step files record float64 reference gradients, scaled/unscaled parameter gradients, Q-input gradients, stem/head gradients, finite masks, forward distributions, clipping, update/parameter norms, loss scales, executed/skipped steps per optimizer and observed boundary dtypes. A separate float64 check compares backprop, parameter shift and lightning adjoint for both circuits.

| Setting | Analytic map | Fixed quantum | V7 quantum | Interpretation |
|---|---|---|---|---|
| Float64 and float32 references | Stable | Stable | Stable | No failure on these small fixed inputs |
| CPU AMP fp16, inherited Q-input dtype | Stable | 5/5 unsupported-operation errors | 5/5 unsupported-operation errors | `exp_vml_cpu` does not implement `ComplexHalf` |
| CPU AMP fp16, explicit float32 Q region, scaler on | Stable | Stable | Stable | Boundary resolves this CPU support problem |
| CPU AMP fp16/bfloat16, float32 Q region, scaler off | Stable in ten-update screen | Stable | Stable | Short diagnostic only; not a recommendation to disable scaling generally |
| Controlled Q-gradient Inf, correct scaler stepping | 5/5 protected | 5/5 protected | 5/5 protected | Q optimizer skipped; finite classical optimizer still stepped |
| Same Inf, historical direct stepping after unscale | 5/5 corrupted | 5/5 corrupted | 5/5 corrupted | Generic missing nonfinite-step protection |
| Historical direct stepping with finite gradients | Stable in screen | Stable | Stable | Direct stepping after unscale is dangerous when nonfinite gradients occur, not automatically numerically different on every finite step |

Overall: 110 completed cells, 15 deliberately induced parameter-corruption cells, ten unsupported-operation cells. These failure outcomes are retained as evidence, not discarded runs. [Summary](e0_e3_20260915/e3/precision_summary.json); [aggregated checks](e0_e3_20260915/e3/aggregate_checks.json).

The CPU support exception was reduced to a primitive: both `qml.RZ.compute_matrix(float16_angle)` and a direct PyTorch complex exponential created from float16 fail with the same `ComplexHalf` error. Float32 and bfloat16 versions execute using complex64 in this primitive probe. This is a runtime operation-support limitation, not evidence of a quantum gradient plateau. [Operator probe](e0_e3_20260915/e3/operator_probe.json).

A final six-case controlled stress test used an initial loss scale of **2²⁴**, rather than the historical/default 2¹⁶, without injecting Inf manually. All three map families first showed nonfinite gradients at the **backward head-output boundary**, despite an explicit float32 Q region. Correct scaler semantics skipped both optimizers and preserved finite parameters; historical direct stepping corrupted parameters. This establishes that surrounding fp16 arithmetic can generate the dangerous gradients without requiring a uniquely quantum mechanism. It does **not** prove the original L4 run encountered this exact scale/operation. [Stress evidence](e0_e3_20260915/e3/scaling_stress.json).

The historical source before `126d396` already called `unscale_` for both optimizers, then called the custom direct optimizer step. The defect isolated here is bypassing finite-step skipping, not simply forgetting to unscale. Correct multi-optimizer behavior follows the installed-version [PyTorch 2.10 AMP examples](https://docs.pytorch.org/docs/2.10/notes/amp_examples.html): unscale before inspection/clipping, scaler.step per optimizer, one scaler.update. The documentation was checked on 15 September 2026. A live PennyLane documentation fetch returned HTTP 429; executable installed source and actual behavior were used instead of inferring undocumented support.

Backprop/parameter-shift/lightning-adjoint input and angle Jacobians agreed within **8.89×10⁻¹⁶** in the small float64 checks. Output interface dtypes were recorded and differed by backend; internal simulator state dtype was not instrumented. These checks do not establish broad conditioning or long-run trainability. No new result establishes a V6 barren plateau.

**Remaining uncertainty:** no CUDA or MPS device was exposed in this environment. The original L4 tensors, environment, exact overflow origin and mixed-precision kernel behavior were not reproduced. E3 is complete as a bounded local minimum reproduction with an explicit CUDA gap, not a claim to have completely reconstructed the historical GPU event.

## 7. New artifacts

The [complete inventory](e0_e3_20260915/artifact_inventory.json) lists every new deliverable file, its purpose, size and hash. This table groups repeated run artifacts to keep the report readable. Runtime Python bytecode caches are not research deliverables.

| Path or artifact group | Purpose |
|---|---|
| `src/research_protocol.py` | Content identities, deterministic protocol, verified loading, metrics, run metadata/schema checks and exclusive evidence writes |
| `src/classical_quantum_controls.py` | Exact fixed-map expectation function and historical-layout feature computation |
| `src/research_feature_maps.py` | Seven small common-pipeline feature-map controls; no V8+ implementation |
| `scripts/build_research_dataset_manifest.py` | Deterministic E0 inventory, duplicate audit, historical membership and new split creation |
| `scripts/replay_quantum_features.py` | E1 source-circuit/cache/checkpoint replay with explicit numerical thresholds |
| `scripts/run_feature_probe.py` | Prespecified E2 screening and crossed pilot, validation-only selection and analysis |
| `scripts/diagnose_hybrid_precision.py` | E3 minimum numerical reproduction, per-step trace and derivative checks |
| `scripts/stress_hybrid_scaling.py` | Additional nonfinite-gradient stress without manual Inf injection |
| `scripts/test_research_foundation.py` | Four lightweight integrity tests |
| `scripts/verify_research_execution.py` | Independent metric/checkpoint verification, secondary bootstrap and primitive support probe |
| `research/schemas/run.schema.json` | Schema for new research run records |
| `research/configs/e2_pilot.json`, `precision_screen.json` | Frozen experiment designs |
| `research/e0_e3_20260915/before_state.json`, `before_tracked.patch` | Starting repository/evidence identity |
| `dataset_manifest.json`, `historical_membership.json`, `protocol.json`, `e0_checks.json` in the execution directory | E0 records, preserved old membership, explicitly new protocol and checks |
| `e1/` | Preserved failed first attempt and its partial prediction artifact |
| `e1_verified/` | Successful replay, original/corrected identity records, per-bank/per-checkpoint results and 27 prediction archives |
| `e2/` | 150 run JSONs, 150 checkpoints, 150 prediction archives, selection/partition records, primary and secondary summaries |
| `e3/` | 135 case traces, aggregate run/summary records, primitive probe, derivative results and six-case scaling stress |
| `source_snapshots/` | Hash-matched copies of exact executed source versions, including the pre-correction E1 version |
| `verification.json`, `after_state.json`, `artifact_inventory.json`, `commands.md` | Verification, completion state, complete output index and reproduction commands |
| This report | Scientific results, limits and next-stage decision request |

No pre-existing source, result, dataset, checkpoint, old log, manuscript or governing-plan file was modified. Additive infrastructure and evidence are the only changes.

## 8. Commands and experiments executed

The [command record](e0_e3_20260915/commands.md) contains the launch sequence, including the failed attempt. Commands run from the repository root using the existing `venv/bin/python`; no dependency installation was required. `MPLCONFIGDIR=/private/tmp/astra-mpl` and `XDG_CACHE_HOME=/private/tmp/astra-cache` kept temporary library caches separate.

```bash
venv/bin/python scripts/build_research_dataset_manifest.py --output-dir research/e0_e3_20260915
venv/bin/python scripts/replay_quantum_features.py --root research/e0_e3_20260915 --output-name e1_verified
venv/bin/python scripts/test_research_foundation.py
venv/bin/python scripts/run_feature_probe.py --root research/e0_e3_20260915 --config research/configs/e2_pilot.json
venv/bin/python scripts/diagnose_hybrid_precision.py --root research/e0_e3_20260915 --config research/configs/precision_screen.json
venv/bin/python scripts/verify_research_execution.py --root research/e0_e3_20260915
venv/bin/python scripts/stress_hybrid_scaling.py --root research/e0_e3_20260915
```

These are the actual execution locations; scripts intentionally refuse to overwrite existing outputs. A reproduction should use a clean copy with a fresh execution root. The foundation test is tied to this saved campaign's manifest. After execution, manifest/partition path handling was hardened so new CLI runs derive those references from their requested root; the integrity suite passed again. Exact pre-hardening source snapshots preserve the recorded experiments, and no training was repeated for this metadata-only improvement. The tools are research infrastructure for this phase, not a fully generalized experiment service.

Independent verification recomputed all 150 E2 result metrics from prediction archives, verified all 27 E1 archives, and loaded two E2 checkpoints/maps to regenerate logits from raw assessment images; both replay errors were zero. All new training run records passed the schema checks. Additional semantic validation checked content split disjointness and unchanged protected files.

## 9. Compute consumed

| Work | Recorded process CPU seconds | Recorded wall seconds |
|---|---:|---:|
| E0 manifest and deterministic rebuild | 0.57 | 0.65 |
| E1 failed first attempt | 1.79 | 11.94 |
| E1 successful full replay | 5.49 | 6.18 |
| E2 complete 150-fit pilot, including feature construction and analysis | 33.53 | 41.63 |
| E3 main screen and derivative checks | 10.44 | 11.52 |
| Evidence verification and secondary uncertainty | 1.91 | 2.47 |
| E3 additional scaling stress | 0.39 | 0.64 |

The core recorded total is approximately **54.1 process CPU seconds**, or 0.015 CPU-hours. These timers exclude some Python imports, initial repository inspection, source editing and final reporting; they are not a whole-host energy/accounting measurement. Initial protected-file hashing took about 0.83 wall seconds, and the foundation suite about 0.17 seconds. No GPU/accelerator time or paid quantum hardware was consumed. Every stage was far below its cap. Output artifacts occupy roughly 0.12 GB before final indexing.

No performance speedup is inferred from this timing table; stages perform different work. E1 establishes functional equivalence, not a controlled end-to-end timing comparison.

## 10. Failures and unresolved questions

- The first E1 attempt failed while recording a prediction path: a relative output path was passed to `relative_to` against an absolute root. This was a bookkeeping error after successful numerical checks, not an observed feature mismatch. The failed record and partial artifact remain in `e1/`; an explicit fresh output directory holds the successful run.
- The successful E1 generic run template initially carried the future-pilot split IDs. The correction is additive and fully disclosed in `run_corrected.json`; actual replay ordering was already stored in the historical-membership record. Exact executed source was preserved by hash.
- Ten E3 cases encounter the unsupported CPU `ComplexHalf` operation. Fifteen main-screen cases and three scaling-stress cases intentionally demonstrate unsafe parameter updates. They are reported, not excluded to make stability look better.
- Historical benchmark internal split IDs, remote dataset differences, original V7 GPU overflow origin and missing raw V7 metadata are still unrecoverable from current evidence.
- Content duplicates and conflicting labels are handled explicitly for the new pilot, but unknown writer/source groups, near duplicates, label adjudication and rights/provenance remain open.
- E2 uses a small common readout and three data counts. It does not reproduce the historical residual network or establish a universal low-data ranking. The two training levels and three bank/subset levels provide limited variance information; no confirmatory p-value is claimed.
- Absence of a CUDA device limits the E3 conclusion. The observed CPU support exception must not be presented as the exact historical L4 failure.

## 11. Scientific claim ledger

| Claim | Classification | Scope / evidence |
|---|---|---|
| Historical folders contain 72 duplicate groups, 15 crossing train/test and five conflicting-label groups | **Observed and verified** | Manifest hashes and deterministic rebuilding |
| New pilot split has disjoint content IDs and leaves historical data unchanged | **Verified** | Split checks and 4,404 protected-file hashes |
| Current fixed circuit's expectation map is a trigonometric prefix-product function | **Derived**, then **verified** | Prior algebra; new circuit, derivative and full-cache replay |
| The 27 replayed checkpoints produce identical analytic/reference predictions | **Verified** | Logit arrays and ID-keyed prediction archives |
| This fixed measured map does not require irreducible quantum computation for its reported predictions | **Derived from verified equivalence** | No claim about V7, all quantum maps or hardware noise |
| RFF has a higher mean ALC than analytic fixed-Q in this pilot | **Observed** | −1.46-point Q−RFF difference on separate assessment validation |
| A worthwhile low-data quantum-derived advantage survives strong controls | **Not supported; hypothesis not advanced** | Gate fails; intervals remain exploratory |
| Feature-bank variability is substantial in the tested grid | **Observed descriptive decomposition** | Not a population variance theorem |
| Direct stepping after unscale can corrupt parameters on nonfinite gradients | **Verified** | Analytic/fixed/V7 controlled Inf and scaling-stress tests |
| Explicit float32 Q computation is sufficient to make all mixed-precision training safe | **Falsified in the scaling stress** | Surrounding fp16 head gradients can still overflow |
| CPU float16 Q failure here reduces to an unsupported complex exponential | **Verified** | Q rotation and direct PyTorch primitive give the same exception |
| The exact initial cause of the historical L4 NaN is known | **Unresolved** | GPU runtime and tensors not reproduced |
| V6 suffered a barren plateau | **Unresolved / not established** | No appropriate new width/depth/ensemble evidence |
| V7 or a future spectral/spatial circuit has a quantum-specific benefit | **Hypothesis** | Not tested or authorized for training in this phase |

## 12. GO / NO-GO assessment

| Branch | Decision | Reason and boundary |
|---|---|---|
| Low-data fixed-feature branch | **NO-GO for powered confirmation now** | Prespecified meaningful-effect gate fails after classical counterattack; stop this branch at the current evidence |
| Trainable-quantum / V8 branch | **CONDITIONAL GO for resolving numerical prerequisites only** | Generic unsafe-update mechanism is established, but CUDA fidelity and stable operating configuration still need a small targeted check. This is not authorization to start E4 or train V8 |
| V9 spectral/representation branch | **NOT YET AUTHORIZED; empirical justification decreases** | The fixed-feature rationale is weakened by exact classical replay and RFF opposition. V7's distinct spectral mechanism itself remains untested |
| V10 spatial branch | **NOT YET AUTHORIZED; justification unchanged** | No spatial/bottleneck mechanism experiment was performed; data hygiene improved but does not establish a quantum-spatial effect |

Current results do not authorize any later stage automatically. The existing benchmark families and historical leaders remain separate; these new validation-pilot scores are not replacements for the publication test table.

## 13. Next decisive experiments

The smallest useful next work is:

1. **A tightly bounded CUDA version of the numerical reproduction**, using the pinned environment, existing V7 circuit, analytic counterpart, correct versus historical stepping, explicit boundary and controlled scale stress. Record actual dtypes and first nonfinite boundary. This is a numerical diagnosis, not an OCR training campaign or new architecture.
2. **Adjudicate the five conflicting-label content groups and investigate source/writer grouping**, preserving original files and recording decisions in a new annotation layer. This reduces uncertainty in every future benchmark more directly than another model version.
3. Only after those prerequisites, and only with explicit authorization, consider **one checkpoint dependence intervention** followed by the smallest necessary retrained control if warranted. An inference-only zero/permutation intervention can reveal dependence but cannot alone prove that a Q block is necessary after retraining. No such E4 intervention was executed here.

No powered low-data confirmation, second-dataset training or V9/V10 search is recommended from these results alone. An entirely new mechanism would require a new bounded hypothesis and comparison, not additional seeds to rescue the present mean difference.

## 14. Decision request

Authorization is required before moving beyond this completed E0–E3 phase. The next proposed authorization is limited to a CUDA numerical reproduction and provenance/label adjudication, not V8+ training. All experiments in this report have stopped.

STOP — E0–E3 complete. Awaiting authorization for the next research stage.
