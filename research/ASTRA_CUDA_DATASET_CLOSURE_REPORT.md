# Astra CUDA / dataset closure report

**Investigation:** 16 September 2026. **Report completed:** 17 September 2026 after resuming an interrupted session. Governing evidence: [research program](ASTRA_QUANTUM_RESEARCH_PROGRAM.md), [E0–E3 execution report](ASTRA_E0_E3_EXECUTION_REPORT.md), and immutable [E0–E3 artifacts](e0_e3_20260915/artifact_inventory.json). This is a diagnostic follow-up only.

## 1. Executive conclusions

**The historical CUDA initiating failure remains unresolved (N5).** This host exposes no CUDA device or runtime. Part A stopped at environment verification and static preparation; no CPU experiment was substituted for a CUDA reproduction. Previously verified generic optimizer-protection failure remains valid, but cannot establish the original L4 initiating operation.

**Retain quarantine for all five conflicting-content groups and the invalid `00` record.** Exact image identity and contradictory filename labels are verified. Available documentation and visual inspection do not justify corrected labels. The annotation layer records uncertainty rather than choosing plausible classes.

**Exact deduplication leaves four very close image pairs.** Three cross the current new protocol's split boundaries: two train/validation and one train/test. Review these before freezing another experimental protocol. Their effect on model metrics was not measured; no E2 rerun was performed.

**Source attribution is substantially clearer; writer independence is not.** Original uploader metadata confirms the label rule and class mapping. Filename prefixes remain provisional grouping clues. No authoritative per-image writer/page crosswalk was recovered.

The E1 conclusion—this fixed noiseless measured map is exactly classically executable—and E2 **NO-GO for powered confirmation** remain unchanged. Nothing here establishes universal RFF superiority or authorizes a new architecture.

## 2. Environment fidelity

The table distinguishes recoverable notebook evidence from the exact failed invocation, whose complete environment was not archived. April 27 is a later clean run, not the original NaN run. March 5 captures an L4; a historical March 2 documentation configuration also says A100 80GB. That conflict prevents treating every old configuration field as a measured property of the failed L4 run.

[Environment probe](cuda_dataset_closure_20260916/environment.json), [structured comparison](cuda_dataset_closure_20260916/environment_fidelity.json), and [indexed notebook excerpts](cuda_dataset_closure_20260916/historical_environment_extract.json) retain the evidence.

| Field | Historical environment | Current reproduction environment | Match status | Likely relevance |
|---|---|---|---|---|
| GPU | L4 captured March 5 and April 27; exact failed invocation not captured | No exposed GPU; CUDA device_count=0 | different | Cannot reproduce CUDA kernels or hardware range behavior |
| CUDA runtime | April: torch cu128 and nvidia-cuda-runtime-cu12 12.8.90; March failed run unknown | Unavailable | different | CUDA operation support and rounding depend on this |
| NVIDIA driver | 580.82.07 in March 5 and April 27 notebook outputs | Unavailable | different | No NVIDIA stack locally |
| Driver CUDA compatibility | nvidia-smi reports 13.0 in both notebooks; this is NOT the installed PyTorch CUDA runtime | Unavailable | different | Separates driver capability from CUDA 12.8 package runtime |
| PyTorch | April 27: 2.10.0+cu128; exact failed-run version unknown | 2.10.0 CPU build | close match | Core version matches later run; backend does not |
| PennyLane | April 27 installation: 0.44.1; exact failed run unknown | 0.44.0 | close match | Patch and backend/interface differences could matter |
| pennylane-lightning | April 27: 0.44.0; exact failed run unknown | 0.44.0 | exact match | Package version only; no same-device execution |
| pennylane-lightning-gpu | April 27: 0.44.0; exact failed run unknown | not installed | different | Historical GPU adjoint path is unavailable |
| cuStateVec | April 27: custatevec-cu12 1.13.1; exact failed run unknown | not installed | different | Quantum GPU simulation dependency |
| Python | April 27: 3.12 from package paths/cp312 wheel; patch unknown | 3.13.7 | different | ABI/environment difference |
| OS/environment | Colab /content paths; Linux x86_64 wheel; exact OS release/kernel unknown | macOS 26.6.2 arm64 | different | Not a CUDA reproduction environment |
| Quantum backend | lightning.gpu in April run output and older configured source | none executed; CPU default.qubit/lightning.qubit evidence retained from E3 | different | Installed lightning.qubit is not lightning.gpu |
| Differentiation | adjoint in pre-fix V7 source; exact failure runtime configuration not archived | none executed | unknown | No CUDA adjoint-versus-backprop comparison this stage |
| AMP/autocast | Pre-fix code requests CUDA float16 autocast; observed boundary dtypes in failed run absent | CUDA autocast not executable/tested | unknown | Source intent is not evidence of actual tensor dtype |
| GradScaler | Pre-fix constructor defaults; state/initial scale at first nonfinite unknown; source unscales both then direct steps | No numerical run | unknown | Optimizer bypass is source-verified; initiating scale not known |
| fp16/bfloat16 operations | No failed-run primitive capability trace or first-nonfinite tensor trace | Not probed: CUDA unavailable | unknown | CPU ComplexHalf failure is not CUDA evidence |
| Code identity | April 27 cede472 captured; failed-run dirty source unknown; pre-126d396 source is a reconstruction anchor | 95d6fb85f27c394e4b33b2cd5e5321c0c61e724a + dirty worktree | different | Current code includes post-failure fixes |

An “exact match” above means only the named package version matches the stated later capture. It does not mean the failed-run environment was reconstructed. No fp16/bfloat16 CUDA operation support was inferred from CPU behavior. `nvidia-smi`'s displayed CUDA 13.0 is driver compatibility information; April's installed PyTorch build/runtime packages are CUDA 12.8.

Repository HEAD before and after is `95d6fb85f27c394e4b33b2cd5e5321c0c61e724a`, branch `master`. The starting worktree already had 38 tracked modified files and extensive untracked evidence. This stage is additive. The tracked patch is unchanged and final verification checks all **5,075 protected pre-existing files**, including datasets, checkpoints, manuscript and E0–E3 results. No commit or push was made. See [before state](cuda_dataset_closure_20260916/before_state.json) and [final verification](cuda_dataset_closure_20260916/final_verification.json).

## 3. CUDA numerical reproduction

**Status: blocked by unavailable CUDA.** The probe reports `torch.cuda.is_available() == False`, device count zero, `torch.version.cuda == None`, no `nvidia-smi`, and no installed `pennylane-lightning-gpu`. MPS is also unavailable, but would not substitute for the requested CUDA experiment.

| Requested level | Execution in this stage | Outcome |
|---|---|---|
| A: normal/default-scale CUDA tiny hybrid | Not run | No CUDA device |
| B: naturally generated overflow under controlled scale stress | Not run | Same blocker |
| C: artificial Inf protection control on CUDA | Not run | Same blocker; prior CPU control retained |
| Backend/dtype comparison, one real V7 batch, several updates | Not run | Escalation gate not reached |

A [static CUDA handoff record](cuda_dataset_closure_20260916/cuda_handoff.json) specifies reuse of the existing small analytic/fixed-Q/V7 diagnostic, deterministic inputs, correct versus historical semantics, boundary fields, staged stress and a target below one GPU-hour, hard maximum two GPU-hours. It is preparation, not an executed or tested CUDA runner. Existing CPU-specific device contexts require adaptation on a CUDA host. No remote session or paid hardware was provisioned.

## 4. First-nonfinite analysis

No new CUDA tensor traces exist. The following is the **previous E3 evidence**, preserved and not rerun:

| Condition | First supported localization | Scope of inference |
|---|---|---|
| CPU fp16 inherited quantum input | Unsupported complex exponential in rotation construction (`exp_vml_cpu` / ComplexHalf) | Operation-support exception, not a finite-to-NaN CUDA transition |
| CPU explicit float32 quantum region, default-scale tiny screen | No nonfinite boundary observed in the bounded screen | Short-run local stability only |
| CPU fp16 surrounding arithmetic, float32 Q region, scale 2²⁴ without injection | First instrumented nonfinite gradient at backward head-output boundary for analytic, fixed-Q and V7 | Generic arithmetic can initiate the dangerous gradient; exact primitive within that boundary and historical CUDA trigger are not established |
| Injected Inf control | Injection itself is the initiating event; direct stepping corrupts parameters, scaler stepping protects them | Optimizer semantics only; not evidence of historical initiation |

Sources: [prior precision summary](e0_e3_20260915/e3/precision_summary.json), [operator probe](e0_e3_20260915/e3/operator_probe.json), [scale stress](e0_e3_20260915/e3/scaling_stress.json). An observed boundary is not automatically the first internal primitive operation. Dtypes, finite counts, gradient/update norms and optimizer state at the original historical event remain unavailable.

## 5. Historical V7 failure classification

**N5 — unresolved.** The available environment cannot reconstruct the initiating historical failure. The evidence separates four questions:

- **Initiating event:** unknown for the original GPU run. Generic fp16 overflow is a plausible explanation, not a reconstructed cause.
- **Optimizer protection failure:** verified from pre-fix source and previous controlled CPU tests. The historical code unscaled both optimizers, then directly stepped them and updated the scaler. Direct stepping bypassed the scaler's nonfinite skip logic. It was not simply a failure to unscale.
- **CPU-only operation support:** verified ComplexHalf exponential limitation, avoided by a float32 quantum region in that CPU diagnostic. This cannot be generalized to L4 operation support.
- **Quantum-specific contribution:** unresolved on CUDA. No new evidence establishes N2, N3 or N4. Neither the prior CPU stress nor these data findings establish a V6 barren plateau.

The prior generic corruption control has an N1-like mechanism, but assigning N1 to the original historical event would overstate the evidence. Historical documentation's stronger quantum-overflow attribution remains a hypothesis; it was not silently edited.

## 6. Stable operating requirements

Before any future trainable-Q experiment, require a finite float32 reference on the actual backend and reproducible input/parameter gradients. If mixed precision is used, require scaled backward, unscale each optimizer once before inspection/clipping, `scaler.step` for each optimizer, and one `scaler.update` after both. Record per-optimizer skipped updates and distinguish scaled activation gradients from unscaled parameter gradients. These semantics follow the [PyTorch 2.10 AMP examples](https://docs.pytorch.org/docs/2.10/notes/amp_examples.html) (accessed 16 September 2026).

Use an explicitly disabled-autocast float32 quantum region as the conservative initial boundary, then verify its actual inputs/outputs on the chosen CUDA backend. This is a starting requirement, not a guarantee against overflow in the stem/head. Inspect nonfinite gradients before clipping; preserve traces rather than replacing NaNs with zeros. Check loss, parameters and optimizer state for finiteness, archive the first failing batch and scale, and stop on unexplained corruption. Log backend/differentiation, driver/runtime/package versions, source hashes, all split IDs, RNG/scaler/optimizer state and precision settings.

A finite-only screen must precede any optimization claim. This is **not authorization to train V8**.

## 7. Conflicting-label adjudication

All five groups contain two byte-identical as well as pixel-identical files; all use the `TBMM-005` prefix. Four cross historical train/test folders. All ten occurrences are absent from the current deduplicated protocol. This localization suggests a source annotation/context issue, but does not identify the correct class.

| Group / pixel-hash prefix | Every original path and assigned class | Historical membership | Decision |
|---|---|---|---|
| C1 / `2f1d67862d22…` | `set/train/TBMM-005-000011.png` — 11 ze<br>`set/test/TBMM-005-00009.png` — 09 zel | Train/test | Insufficient evidence; retain quarantine |
| C2 / `4a2a5f5f2858…` | `set/train/TBMM-005-00000034.png` — 34 1<br>`set/train/TBMM-005-00001.png` — 01 elif | Train/train | Insufficient evidence; retain quarantine |
| C3 / `53dae2c71785…` | `set/train/TBMM-005-00000000000000024.png` — 24 mim<br>`set/test/TBMM-005-00000026.png` — 26 he | Train/test | Insufficient evidence; retain quarantine |
| C4 / `8d403dd561c4…` | `set/train/TBMM-005-000035.png` — 35 2<br>`set/test/TBMM-005-000002.png` — 02 be | Train/test | Insufficient evidence; retain quarantine |
| C5 / `d64a2e31bf2b…` | `set/train/TBMM-005-000038.png` — 38 5<br>`set/test/TBMM-005-0000026.png` — 26 he | Train/test | Insufficient evidence; retain quarantine |

The [full conflict records](cuda_dataset_closure_20260916/conflict_records.json) contain complete content/sample and occurrence IDs, full byte/pixel hashes, filenames, original split, zero-based label and code/name mapping, PNG metadata and exact-identity checks for every file. PNG ancillary metadata was empty. The [review addendum](cuda_dataset_closure_20260916/adjudication_review.json) records visual status, source evidence, confidence and rationale; the [contact sheet](cuda_dataset_closure_20260916/conflict_contact_sheet.png) shows both instances of every group.

C1 is a marked curved form; C2 is a single upright/slanted stroke, for which `elif` versus numeral `1` is context-sensitive; C3 is a low-resolution broad connected form; C4 is forked with a descending stroke; C5 is looped. These descriptions do not establish Ottoman class ground truth. Particularly plausible numeral readings are not sufficient for relabeling. No original word/page context or independent expert adjudication was recovered.

For every group: **high confidence in exact identity/conflict and quarantine; insufficient confidence in a corrected label**. Outcome is “insufficient evidence,” not a proven class A/B correction or proof of intrinsic ambiguity. Original filenames, images and labels remain unchanged. The new annotation layer contains `corrected_label: null`.

## 8. Invalid sample investigation

`set/train/TBMM-007-00000000000000000000000000000000000.png` remains invalid/quarantined. Its pixel hash is `0160416bafc67499e3b46655f57b7801937fbe2c76604f6a89e1c1255ac25d37`; its byte hash and occurrence ID are in the [invalid record](cuda_dataset_closure_20260916/invalid_sample.json).

The image decodes correctly and is not blank (range 0–255, standard deviation 104.61). It has no exact duplicate and no informative PNG metadata. Neighbors in the filename family use long zero runs plus class suffixes; neither the runs nor lexicographic adjacency supplies a missing label. The uploader rule confirms `00` is outside the valid mapping; the digit zero is code **43**.

The nearest valid aligned raster has RMSE **46.05/255**, far outside the conservative near-duplicate queue. Five inspected neighbors include `08` (dal) and `10` (ra), but resemblance is not independent label evidence. The [comparison sheet](cuda_dataset_closure_20260916/invalid_contact_sheet.png) and review addendum preserve that observation. No corrected class is proposed.

## 9. Writer/source/provenance findings

The [original uploader metadata](https://www.kaggle.com/api/v1/datasets/view/alpbintuuzun/ottoman-turkish-characters) identifies Alperen Özer and Alp Bintuğ Uzun, release version 1 dated 26 August 2020, 3,894 images, the last-two-filename-digit rule, and all 44 class definitions. The mapping exactly matches the repository. It reports 1,371 Talik, 411 Rika, 1,974 Matbu and 138 mixed characters, and a **“GPL 2” uploader license label**. This is the original source, not a third-party Kaggle summary. Access date: 16 September 2026; response retained in [kaggle_metadata.json](cuda_dataset_closure_20260916/kaggle_metadata.json).

The [authors' 2021 publication](https://dergipark.org.tr/en/pub/jeps/article/888164), section III, documents manual character selection and TBMM-supplied historical documents. It describes 4,038 characters (2,114 Matbu, 413 Rika, 1,373 Talik, 138 mixed), differing from the downloadable release by 144. It provides aggregate class/style tables, not a crop-to-page/writer crosswalk or conflict adjudication. Printed pages 583–584 were visually checked. The release discrepancy remains unexplained; publication totals cannot be used to repair individual filenames. [Publisher PDF](https://dergipark.org.tr/en/download/article-file/1607142), accessed 16 September 2026.

The version-1 archive SHA-256 is still `35b68d7f7e677e591d2305c573ce350914042f0f7e5b2fa79d2cb16415563885`, matching the August source-identity record. Its [inventory](cuda_dataset_closure_20260916/source_archive_inventory.json) contains 3,894 PNGs and no sidecar metadata or license text file. The previous 3,894-image content match was preserved rather than rerun. Local data hashes remain unchanged. No independent upstream corrections were found in the bounded search; that is not proof none exist.

| Grouping evidence | Classification | What can be used |
|---|---|---|
| Original filename label suffix | Verified class-code convention | Label parsing, with `00` explicitly invalid |
| Writer identity per image | **Unavailable** | No writer-independent generalization claim |
| Verified source/page identity per image | **Unavailable** | Aggregate source history cannot supply per-image groups |
| `TBMM-*`, `koc_img*`, `talik_*`, UUID-like stems | **Plausible grouping requiring validation** | Candidate crosswalk for source-owner review only |
| Dataset-level origin/style description | Verified as author/uploader statements | Dataset description, not sample-level style assignment |

The deterministic [filename inventory](cuda_dataset_closure_20260916/filename_grouping.json) produces **56 provisional prefixes**, 17 spanning historical folders and 48 spanning new protocol splits. Prefix extraction assumptions—including fixed-width `koc_img` parsing—are explicit; these counts are not counts of writers/pages. None of the 3,894 PNGs has informative ancillary metadata. Prefix diversity and class coverage were inspected; repeated prefixes are not assumed to be writers, and differing prefixes do not guarantee independent source content.

A future verified crosswalk should map content IDs to original document/page and, only where known, writer. Then split entire verified source/writer groups with class-coverage checks and retain duplicate/near-duplicate components together. No such replacement protocol was made here.

The uploader license label is verified; a complete upstream rights chain and scan-specific redistribution permissions remain unresolved. The archive itself provides no additional license document. This stage does not interpret a dataset label as proof of every original scan's permissions. Exact URLs, access dates, limitations and bounded author-repository searches are in the [source catalog](cuda_dataset_closure_20260916/source_catalog.json).

## 10. Near-duplicate findings

The prespecified inexpensive screen compared all **7,290,471 unordered pairs** of the 3,819 distinct valid-label content groups at native 32×32 alignment. Exact duplicates were already collapsed for comparison. A float32 matrix shortlist with a one-MSE-unit guard was verified in float64. Thresholds were RMSE ≤5 and ≤10 in 0–255 grayscale units. Sixty independently checked matrix distances had maximum MSE discrepancy 0.004883, below the guard; all four selected pair distances were separately recalculated in float64.

| Pair | RMSE | Differing pixels / 1,024 | Current new protocol | Visual assessment |
|---|---:|---:|---|---|
| ND0001 | 3.4641 | 3 | Train / validation | Near-identical `talik_01` hı rasters |
| ND0002 | 4.9467 | 3 | Train / validation | Near-identical `talik_01` çim rasters |
| ND0003 | 4.9467 | 3 | Train / train | Near-identical same-UUID gef rasters |
| ND0004 | 7.3587 | 15 | Train / test | Very similar simple sin stroke, different UUID prefixes; common origin uncertain |

The [review queue](cuda_dataset_closure_20260916/near_duplicate_queue.json) contains complete paths, IDs, labels, distances and split memberships. All four were visually inspected in the [review sheet](cuda_dataset_closure_20260916/near_duplicate_review_1.png). The first three have three changed pixel values; their related-raster explanation is strong, but original crop lineage is not verified. The fourth is a low-information stroke, making source identity particularly uncertain.

At threshold 5: three pairs, six content IDs, two cross-protocol pairs. At threshold 10: four pairs, eight IDs, three cross-protocol pairs. Cross-boundary candidates touch three training, two validation and one test example: **2/668 validation (0.30%)** and **1/458 test (0.22%)**. Counts are review scope, not estimated score inflation. Class-specific impact can exceed those overall proportions; no model was evaluated.

This screen does not cover shifted, rotated, differently cropped or strongly rescaled copies. Thus few aligned matches are not evidence that broader leakage is negligible. No similarity threshold deleted a sample. The raw fields called `ink_iou` and `ink_pixels_below_128` measure dark pixels, mostly background. Their naming error is explicitly corrected by white-foreground statistics in the review addendum; candidate selection used RMSE only and is unaffected.

## 11. Dataset protocol implications

**The existing deduplicated v1 protocol needs a small, explicit revision before another experiment uses it as a frozen foundation.** Keep its historical record and all E2 outputs unchanged. Review the four queued pairs without reference to model predictions: treat the three-pixel pairs as likely related inputs; seek original context for the simple-stroke pair. A proposed v2 should group confirmed related rasters within a single split or explicitly quarantine unresolved cross-split candidates. Preserve final-test separation and record every membership difference. Do not choose a repair using its effect on accuracy.

Retain the five conflicting-content quarantines and the invalid `00` exclusion. Source/writer grouping remains a separate unresolved limitation. The intended next study can support within-dataset causal diagnostics under that limitation; it cannot claim writer-independent generalization. This task **does not create or substitute v2**.

The historical test has already been reused, and this task inspected pixels/labels for data quality with explicit authorization. It is not a newly pristine confirmatory holdout. The E2 NO-GO decision is unchanged, and no new seeds or “rescued” effect estimates were produced.

## 12. New artifacts

Every new file, purpose, byte count and hash is enumerated in the [artifact inventory](cuda_dataset_closure_20260916/artifact_inventory.json). No pre-existing research file was modified.

| New artifact or group | Purpose |
|---|---|
| `research/ASTRA_CUDA_DATASET_CLOSURE_REPORT.md` | This bounded follow-up report and readiness assessment |
| `scripts/investigate_dataset_closure.py` | Read-only environment, conflict, filename and aligned-distance investigator |
| `scripts/verify_dataset_closure.py` | Final protected-file, evidence, metric and report-integrity checks |
| `before_state.json`, `before_tracked.patch` | Starting identity and preservation baseline |
| `environment.json`, `environment_fidelity.json`, `historical_environment_extract.json` | CUDA blocker and historically qualified environment comparison |
| `historical_config.txt`, `historical_enhanced_training.txt`, `historical_trainable_quantum_model.txt` | Pre-fix Git source snapshots, not executable replacements |
| `cuda_handoff.json` | Static numerical reproduction specification; not executed |
| `investigation_config.json`, `investigation_runtime.json`, `executed_investigator_source.py` | Frozen thresholds, measured runtime and exact executed source |
| `conflict_records.json`, `conflict_contact_sheet.png` | Complete five-group identities and comparison images |
| `invalid_sample.json`, `invalid_contact_sheet.png` | Invalid-record evidence and bounded nearest-image review |
| `filename_grouping.json` | Provisional prefix membership and metadata audit |
| `near_duplicate_queue.json`, `near_duplicate_review_1.png` | Four-pair review queue and comparison images |
| `adjudication_review.json` | Visual decisions, confidence, source-map validation and foreground-statistic correction |
| `kaggle_metadata.json`, `kaggle_search_response.json`, `source_archive_inventory.json` | Original source responses and archive inventory/hash |
| `author_repository_listing.json`, `primary_publication_access.json`, `source_catalog.json` | Bounded provenance search and primary-source access records |
| `commands.md`, `final_verification.json`, `artifact_inventory.json` | Reproduction commands, failures, final preservation result and complete inventory |

Artifact basenames in this table are under `research/cuda_dataset_closure_20260916/` unless a full repository-relative path is shown. Temporary publisher PDF/renderings and thesis text are reading intermediates outside the deliverable set.

The [command record](cuda_dataset_closure_20260916/commands.md) documents exact launches, source access, original failed draft/interpreter attempts and the interruption/resumption. No broad training or test campaign was launched.

## 13. Compute consumed

**GPU: 0 seconds. Training runs: 0. Paid hardware cost: 0.**

| Measured operation | Process CPU seconds | Wall seconds |
|---|---:|---:|
| Starting protected-file snapshot | 0.432 | 0.866 |
| Main environment/data investigator | 0.936 | 1.193 |
| Near-duplicate comparisons, included above | 0.080 | 0.132 |
| Independent candidate/annotation verification | 0.008 | 0.008 |

Final verification CPU/wall time is recorded separately in `final_verification.json`. Source-archive retrieval took 2.50 wall seconds for 2,352,247 bytes. Other reads, PDF rendering, searches and report writing were not jointly instrumented; no exact all-process CPU total is claimed. The interrupted elapsed calendar interval is not compute consumption. No stage approached its compute cap.

## 14. Unresolved questions and failures

- Original CUDA first-nonfinite event, actual failing-batch/scaler state, and any backend/interface contribution: **unknown**, not a negative CUDA finding.
- Exact failed-run environment and historical A100/L4 documentation discrepancy: unresolved. Later April versions are proxies only.
- Correct labels for the five conflicts and `00`: unknown; quarantine retained. No expert validation or crop context is available.
- Per-sample writer/source/page identities and broader transformed near duplicates: unknown. Filename grouping is provisional.
- Release/publication count discrepancy and complete upstream rights chain: unresolved despite verified uploader metadata.
- Quantitative effect of the four review pairs on evaluation: unmeasured. Current E2 decision remains unchanged.
- Initial investigator parse error and default-interpreter missing `cv2` were fixed before the successful run; no dependencies installed. Web screenshot timeouts were resolved by local PDF rendering. Fontconfig warnings did not prevent visual inspection.
- Misnamed dark-pixel statistics are retained transparently with a correction addendum; no candidate/label/split decision depends on them.

No failed CUDA experiment is hidden: none could be launched. Part A remains blocked, and the final stop wording denotes completion of the authorized investigation/reporting, not successful CUDA closure.

## 15. Updated scientific claim ledger

| Claim | Classification | Evidence / limit |
|---|---|---|
| This host exposes no CUDA device/runtime | **observed** | Environment probe |
| Later L4 notebook captures contain the stated driver/package versions | **verified** | Indexed notebook outputs; not exact failure environment |
| Historical direct steps bypassed scaler protection after unscale | **verified** | Archived pre-fix source + prior E3 controls |
| Generic overflow explains the original historical initiation | **hypothesis** | Plausible; cannot assign N1 to that event yet |
| A quantum-specific CUDA defect/conditioning mechanism caused the event | **unresolved** | No current CUDA trace |
| CPU ComplexHalf support failure and generic head-boundary stress are separable | **verified** | Prior E3 evidence, not rerun |
| Five groups carry conflicting valid labels on identical bytes/pixels | **verified** | Manifest, byte checks, pixel equality and visual review |
| One particular label is correct for any quarantined group or `00` | **unresolved** | No authoritative per-image evidence |
| Four aligned near-duplicate candidate pairs exist; three cross new splits | **verified** | Exhaustive bounded metric, exact-distance recheck, saved IDs |
| Three-pixel pairs derive from related rendering/cropping | **hypothesis** | Strong raster similarity; lineage unavailable |
| Near-duplicate leakage materially changes a model score | **unresolved** | No score experiment performed |
| Source filename mapping equals repository's 44-class map | **verified** | Original uploader metadata comparison |
| Uploader publicly labels the release GPL 2 | **observed** | Original source API; not full rights-chain verification |
| Prefixes identify writers or exact source pages | **unresolved** | No authoritative mapping |
| Existing v1 needs explicit review/revision before new frozen experiments | **derived** | Known cross-split near-identical rasters |
| Fixed noiseless map is classically executable | **verified** | Existing E1 algebra and replay, unchanged |
| Fixed-feature low-data powered confirmation is currently NO-GO | **derived** | Existing E2 decision retained; not universal classical superiority |

## 16. E4 readiness decision

**CONDITIONALLY READY**

E4 can be **designed**, provided its protocol incorporates these unresolved foundation gates before execution:

1. A CUDA-host minimal numerical diagnostic must pass valid semantics with actual backend boundary traces. Historical N5 need not be replaced by a fabricated causal story, but an unexplained failure of the valid path blocks training.
2. Review the four near-duplicate candidates and freeze a separately versioned data protocol with explicit treatment of confirmed/unresolved cross-split pairs. Retain all current conflict/invalid quarantines; never rewrite v1 evidence.
3. Restrict claims to the known dataset/source limitations. Writer-independent or confirmatory generalization requires validated groups or a suitable independent holdout; source rights limitations remain explicit.
4. Freeze exact environment, data IDs, checkpoint criterion, causal contrasts and stop rules before looking at outcomes.

This is readiness for protocol design, **not authorization to implement or run E4, V8, V9, V10 or another training stage**.

## 17. Smallest next scientific step

First, on an available CUDA host, consider the already specified tiny analytic/fixed-Q/V7 normal-scale comparison with boundary tracing; add scale stress only if needed. In parallel with protocol drafting, resolve the four-item data queue and obtain crop context if possible. These are smaller and more decisive than launching another architecture.

Once those gates pass and authorization is granted, the smallest E4 causal question to consider is whether the existing trainable quantum path contributes to predictions under a fixed gain/skip setting, using a matched frozen-map replacement and a zero-path intervention with the same samples and downstream learner. Prespecify the contrast and stop if the path has no measurable necessity; only then consider expanding gain/bypass conditions. This is a recommendation only. No scaffold, architecture, accuracy tuning, extra E2 seeds or manuscript edits were performed.

STOP — CUDA/data closure complete. Awaiting authorization for E4 or another explicitly approved stage.
