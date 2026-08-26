# Phase 0 Research Protocol

## Camera-pose-aware same-instance matching on MOVi-D and MOVi-E

**Protocol ID:** MOVI-DE-POSE-001  
**Version:** 0.2  
**Date:** August 25, 2026  
**Status:** Approved and frozen after pilot feasibility audit; before confirmatory selection, pair generation, or model fitting  
**Parent experiment:** Depth-derived geometry for within-video same-instance matching on MOVi-A  
**Protocol seed:** `20260825`

## Protocol decision

This study will extend the existing pipeline rather than replace it. MOVi-A remains the locked, fixed-camera parent experiment. The extension will use MOVi-E as the moving-camera condition and MOVi-D as the fixed-camera control because D and E share the same broader scene family, object source, and background source. This makes D-versus-E a more interpretable camera-motion comparison than A-versus-E.

## 1. Primary research question

> **For within-video same-instance matching in MOVi-E, does transforming depth-derived visible-surface geometry into a common world coordinate system using the simulator-provided camera pose improve discrimination over an otherwise identical system that uses pose-unaware camera-space geometry?**

The primary estimand is the paired test-set difference in area under the receiver operating characteristic curve:

`Delta_E = AUROC(pose-aligned geometry, MOVi-E) - AUROC(camera-space geometry, MOVi-E)`

Both systems will be trained and evaluated on the same locked MOVi-E video split and the same locked pair manifests. They will use the same RGB encoder, 2D controls, classifier family, regularization search, training data, and evaluation procedure. The only intended difference is whether depth-derived 3D geometry is aligned with the provided camera pose.

### Decision rule

The primary hypothesis is supported only if the two-sided 95% paired video-cluster bootstrap confidence interval for `Delta_E` lies entirely above zero. AUROC is the sole primary metric. Other metrics and comparisons are secondary and will be reported with confidence intervals.

## 2. Why this question matters

The parent MOVi-A experiment showed that depth-derived camera-space geometry improved same-instance matching under a fixed camera. A moving camera changes an object's apparent camera-space position even when the object itself is static. This can make a pose-unaware 3D representation inconsistent across frames. Camera pose should allow visible depth points from different frames to be expressed in one shared coordinate system, separating camera motion from object motion.

The proposed experiment isolates that mechanism. It does not test a deployable monocular system: depth, masks, intrinsics, and camera poses are simulator-provided oracle signals. The result will answer whether correct pose alignment is useful under controlled camera motion, not whether pose can be estimated reliably in real video.

## 3. Secondary research questions

1. How much does moving from fixed-camera MOVi-D to moving-camera MOVi-E change each system's performance?
2. Is the benefit of pose alignment larger for pairs with greater camera displacement or relative camera rotation?
3. Does pose alignment provide little or no benefit on fixed-camera MOVi-D, as expected?
4. How quickly does performance degrade when controlled translation and rotation noise are added to the camera pose?
5. Does a shuffled-pose negative control fail to improve performance, indicating that improvement depends on correct pose correspondence rather than extra features or model capacity?
6. How well do systems trained or calibrated on MOVi-D transfer to MOVi-E without refitting the decision threshold?
7. Are effects different for static versus dynamic objects, long versus short temporal gaps, low versus high visibility, and easy versus hard negative pairs?

## 4. Hypotheses

**H1 — Primary pose-alignment benefit.** On locked MOVi-E test pairs, the pose-aligned system will have higher AUROC than the pose-unaware camera-space system.

**H2 — Operational benefit.** At each system's dev-locked threshold targeting 90% recall, pose alignment will reduce false-match rate on MOVi-E without a material loss of achieved recall.

**H3 — Motion interaction.** The pose-alignment benefit will increase with camera displacement and relative camera rotation.

**H4 — Fixed-camera falsification check.** The pose-aligned and camera-space systems will perform similarly on MOVi-D. A large improvement on D would suggest an implementation difference or unplanned information channel rather than a camera-motion-specific effect.

**H5 — Correspondence check.** Shuffling camera poses among frames within the same split will not improve performance over the camera-space system and should impair geometry consistency.

**H6 — Pose-noise sensitivity.** Performance will degrade as injected translation or rotation error increases. Small errors may be tolerated; large errors should reduce or eliminate the pose-alignment benefit.

## 5. Study design

### 5.1 Datasets and experimental roles

| Dataset | Camera regime | Role in this protocol |
|---|---|---|
| MOVi-A | Fixed | Locked historical parent experiment; context only, not pooled with the new primary analysis |
| MOVi-D | Fixed random camera | New fixed-camera control from the same broader D/E scene family |
| MOVi-E | Linearly moving camera that looks toward the scene origin | New moving-camera treatment and primary analysis dataset |

Official Kubric documentation describes MOVi-D and MOVi-E as synthetic multi-object video datasets using Google Scanned Objects and HDRI backgrounds. MOVi-E adds camera motion with start and end positions sampled on a half-sphere and a documented displacement range of 0–4 scene units. The exact downloaded release, feature schema, and checksums will be recorded before selection is locked.

### 5.2 Pilot and confirmatory samples

The pilot used 20 MOVi-D and 20 MOVi-E videos to validate parsing, coordinate transforms, pair capacity, feature ranges, runtime, and failure logging. Effective with version 0.2, all 20 pilot videos per dataset are permanently excluded from every confirmatory pool (train, development, and test). They may be used only for pipeline validation and exploratory feasibility summaries.

The confirmatory sample is frozen at exactly 150 eligible, non-pilot videos per dataset, selected independently and split as follows:

| Pool | Videos per dataset | Pairs per dataset | Purpose |
|---|---:|---:|---|
| Train | 90 | 6,000 | Fit model parameters |
| Development | 30 | 2,000 | Select regularization and lock thresholds |
| Test | 30 | 2,000 | Final evaluation only |

The optional 250-video design is withdrawn. The official validation split contains 250 videos per dataset; after excluding 20 pilot videos, 230 remain available, leaving an 80-video eligibility margin above the frozen 150-video design. Any later increase above 150 requires a new prospective amendment made before confirmatory pair generation and may not be motivated by development or test performance.

### 5.2.1 Pilot feasibility decision

The locked pilot audit found all 20 videos eligible in each dataset. Aggregate candidate capacity exceeded the pilot-equivalent quota by at least 62-fold for hard negatives and 71-fold for positives. No stopping rule was triggered. The confirmatory experiment therefore proceeds under the 150-video design, subject to a post-split capacity gate before pair manifests are frozen.

### 5.3 Video eligibility and selection

A video is eligible when all required RGB frames, instance masks, depth maps, camera intrinsics, camera extrinsics, and object metadata are readable; at least two object instances have at least two eligible observations; and the video can supply positive and negative candidates under the rules below. Eligibility thresholds inherited from the parent experiment are a minimum visible mask area of 32 pixels and a minimum of 32 valid depth pixels in the instance mask. Crops use 15% padding and are resized to 96 × 96 pixels.

Videos will be selected with a fixed seed. Selection will balance, as feasible, the number of visible objects, number of dynamic objects, aggregate visibility, and camera-motion magnitude. MOVi-D and MOVi-E are independent samples and will not be treated as paired scenes. Video IDs and reasons for exclusion will be recorded in machine-readable manifests.

### 5.4 Split locking and leakage prevention

Videos, not frames or pairs, are the split unit. A non-pilot video may appear in exactly one pool, and pilot video IDs are forbidden from all three confirmatory pools. Pair generation will run independently inside each locked pool. All benchmark pairs will be within-video; no cross-video pair will be included. Test labels may be used only by the final evaluation and failure-analysis stages after configurations and thresholds are locked.

## 6. Pair-generation protocol

Each dataset will receive independent train/dev/test pair manifests with a 50% positive, 25% hard-negative, and 25% easy-negative mix. These quotas apply to each locked pool as a whole; no individual video is required to supply every pair type. Video contribution caps and deterministic balancing may be used to prevent domination by high-capacity videos, provided the pool-level quotas remain exact.

**Positive pair.** Two eligible observations of the same simulator object instance from different frames.

**Hard negative.** Two different object instances in the same video and different frames that share the same semantic category and have similar projected scale. Similar scale is defined by an absolute log area-ratio no greater than the train-derived cutoff that retains at least 25% of eligible same-category negative candidates. The cutoff is computed independently on each dataset's locked training pool only and then frozen for that dataset's development and test pools. Same-asset, different-instance candidates, when present, are flagged as a separate “very hard” diagnostic subgroup but are not guaranteed a quota.

**Easy negative.** Two different object instances in the same video and different frames with different semantic categories.

Pairs will be deduplicated as unordered observation pairs. Pair controls will record temporal gap, minimum visibility, mask areas, scale ratio, category relation, asset relation when available, object dynamic/static status, camera displacement, relative camera rotation, and normalized camera displacement. Sampling shortfalls and deterministic fallback use will be logged rather than silently changing the target mix.

## 7. Systems and controls

All learned variants use the frozen ImageNet-pretrained ResNet-18 image embedding and standardized logistic regression used in the parent experiment. Regularization is selected from `C = {0.01, 0.1, 1, 10, 100}` on the development pool. Pair ordering and training examples are identical across comparable systems.

| ID | System | Allowed model inputs | Purpose |
|---|---|---|---|
| A | RGB | Frozen RGB embeddings | Appearance baseline |
| B | RGB + 2D | RGB plus boxes, areas, image-plane centroids, visibility controls | Ordinary 2D control |
| C | RGB + 2D + camera-space geometry | B plus depth-derived visible-surface features in each frame's camera coordinates | Pose-unaware geometry baseline |
| D | RGB + 2D + pose-aligned geometry | B plus the same visible-surface features transformed into a common world coordinate system | Primary system |
| G | Geometry only | Camera-space or pose-aligned geometry without RGB | Diagnostic, not a production candidate |
| P | Pose only | Relative camera translation and rotation, with no object geometry | Detect direct pose-label shortcuts |
| S | Shuffled pose | D with poses deterministically reassigned among eligible frames within the same split | Negative control |
| N | Noisy pose | D after predeclared translation/rotation perturbations | Sensitivity analysis |

System D must not receive an additional feature merely because it is available in world coordinates. The camera-space and pose-aligned geometry vectors must have the same dimensionality and corresponding definitions wherever mathematically possible.

## 8. Geometry and camera-pose computation

For every eligible object observation, valid pixels inside the instance mask will be back-projected using the depth map and camera intrinsics. The visible-surface centroid and dispersion descriptors will first be computed in camera coordinates. For the pose-aware variant, the same 3D points or their sufficient statistics will be transformed using the documented camera-to-world extrinsic transform.

Before any final pair generation, the implementation must pass these gates on pilot data:

1. Camera-to-world and world-to-camera transforms round-trip within a documented numerical tolerance.
2. Reprojected 3D points agree with their source pixels within a documented pixel tolerance.
3. Static-object world-coordinate centroids are more stable across camera motion than their camera-coordinate centroids, after accounting for visibility truncation.
4. Quaternion order, handedness, axis convention, and metric/scene-unit convention are asserted in code and recorded in the extraction manifest.
5. Failed, empty, non-finite, or implausible reconstructions are excluded with a reason code.

Every extraction manifest must explicitly record `world_handedness`, `world_up_axis`, `cv_to_kubric_axis_multiplier`, quaternion order, transform direction, and the raw-scene-unit convention. The exclusion taxonomy must include `non_finite_reconstruction` and `implausible_reconstruction` in addition to mask-, visibility-, depth-, metadata-, and coordinate-validation failures. A corrupt camera pose that compromises an entire video remains a video-level hard failure and is logged separately.

The pose-aligned representation is named **oracle-pose-aligned visible-surface geometry**. It must not be described as ground-truth object position because occlusion and mask truncation make it a view-dependent surface estimate.

## 9. Information-boundary and leakage policy

**Allowed at inference for systems that declare them:** RGB, instance masks, depth, camera intrinsics, camera extrinsics/pose, and features deterministically derived from those inputs.

**Evaluation or sampling only:** simulator object identity, asset ID, semantic category, ground-truth object position, object velocity, dynamic/static flag, and future-frame information not represented in the pair. These variables may define labels, strata, exclusions, or diagnostics but may not enter model features.

**Forbidden:** any feature that directly or indirectly encodes the same-instance label, test-derived normalization, test-informed hyperparameter choice, test-informed threshold adjustment, or selection of examples based on final model errors before the confirmatory analysis is frozen.

The feature builder will emit an allowlist audit for every system. Unit tests must fail if an evaluation-only field appears in a model matrix.

## 10. Training, calibration, and operating thresholds

Feature normalization and model fitting use train data only. Regularization is chosen on development AUROC, with deterministic tie-breaking toward stronger regularization. The following thresholds are then locked independently for each system:

1. **90%-recall threshold:** the highest development threshold achieving at least 90% recall; primary operational threshold.
2. **Maximum-F1 threshold:** development threshold maximizing F1; secondary operating point.

The locked thresholds are applied unchanged to test predictions. For D-to-E transfer, the MOVi-D-trained model and D-development threshold are applied to MOVi-E without refitting; this is secondary and reported separately from the within-E primary comparison.

## 11. Outcomes and statistical analysis

### 11.1 Primary outcome

The primary outcome is `Delta_E`, the paired AUROC difference between D and C on the identical MOVi-E test pairs. The confidence interval will use 10,000 paired bootstrap replicates that resample test videos with replacement and retain all pairs attached to the sampled video. Random seed: `20260825`.

### 11.2 Secondary outcomes

Secondary outcomes are PR-AUC, false-match rate at the dev-locked 90%-recall threshold, achieved recall at that threshold, maximum-F1 operating-point precision/recall/F1, calibration summaries, and latency components. Paired differences and 95% video-cluster bootstrap confidence intervals will be reported where systems share test pairs.

The camera-motion interaction will be summarized by the difference-in-differences:

`Interaction = (AUROC_D - AUROC_C on E) - (AUROC_D - AUROC_C on D)`

Secondary analyses are descriptive and hypothesis-generating unless explicitly designated by an approved amendment before test access. The report will show effect sizes and confidence intervals rather than promoting isolated p-values.

### 11.3 Predeclared strata

Test results will be stratified by temporal gap; minimum pair visibility; object dynamic/static status; positive/negative label; negative difficulty; camera translation; relative camera rotation; and camera displacement normalized by mean camera-to-scene distance. Continuous stratum boundaries will be train-derived tertiles and frozen before development or test reporting. MOVi-D camera translation and rotation are structurally degenerate, so D will be reported as a single zero-motion falsification stratum; low/medium/high camera-motion tertiles apply to MOVi-E only. Sparse strata will be reported with counts and uncertainty and will not be overinterpreted.

## 12. Pose-noise and falsification analyses

The noisy-pose analysis will evaluate the full Cartesian grid of independent translation and rotation perturbations:

- Translation standard deviation: `0, 0.01, 0.05, 0.10, 0.25, 0.50` scene units.
- Rotation standard deviation: `0, 0.1, 0.5, 1, 2, 5` degrees.

Perturbations will be seeded and applied independently per frame, then held fixed across all pairs using that frame. Noise affects only the coordinate transform. RGB, masks, depth, pair labels, and pair membership remain unchanged. The shuffled-pose system uses a deterministic permutation within dataset and split, with no frame retaining its own pose where a derangement is feasible.

## 13. Failure analysis

After the confirmatory metrics are generated, create a gallery of at least 24 unique test errors at the fixed 90%-recall thresholds. Balance false positives and false negatives across C and D, MOVi-D and MOVi-E, low/high camera motion, static/dynamic objects, and easy/hard negatives where capacity allows. Each item must include side-by-side crops, system scores, locked thresholds, pair controls, camera-motion values, and a one-line diagnosis. Predefined diagnosis tags include matched-category distractor, matched-asset distractor, long-gap motion, camera-motion misalignment, occlusion-truncated mask, biased depth estimate, pose-transform error, and ambiguous appearance.

Failure-gallery selection is qualitative and occurs only after primary analysis. It cannot alter the reported metrics, pair manifests, thresholds, or success decision.

## 14. Exclusions, deviations, and stopping rules

Observations may be excluded only for predeclared technical reasons: unreadable data, insufficient mask/depth support, non-finite geometry, failed coordinate validation, or missing required metadata. Counts and reason codes will be published by dataset and split.

The study may stop for infeasibility if the official data cannot be obtained, coordinate conventions cannot be validated, hard-negative capacity cannot support the design, or compute/storage requirements exceed available resources. A stopped pilot does not generate confirmatory claims.

After the 90/30/30 video split is selected and before any pair manifest is generated, a machine-readable capacity audit must verify every pool's aggregate positive, hard-negative, and easy-negative capacity under the locked definitions. If a pool is short, videos may be deterministically reselected using eligibility and candidate-capacity metadata only; no development/test model scores or test outcomes may be accessed. The original split attempt, reason for reselection, and final split hashes must be retained.

Any substantive change after approval requires a dated amendment that states the reason, affected sections, whether test labels or results had been accessed, and whether the change is prospective or exploratory. Original and amended versions must both be retained. Changes made after test access cannot redefine the primary success criterion.

## 15. Reproducibility and required outputs

The extension will preserve the parent repository's one-command execution model while using dataset-agnostic modules and separate locked manifests for MOVi-D and MOVi-E. Required outputs are:

1. Versioned protocol and amendment log.
2. Download/source manifest with release IDs, URLs, file sizes, and checksums.
3. Seeded eligible-video inventory, selection manifest, and split definitions.
4. Independent pair manifests and balance/capacity audits for every dataset and pool.
5. Fixed system configurations and feature allowlist audits.
6. Per-pair predictions for every system, including pose-noise conditions.
7. Aggregate, paired-difference, stratified, transfer, calibration, and latency results.
8. At least 24 failure-gallery items and a conclusions-and-next-steps document.
9. Source and license documentation for data, models, and dependencies.

## 16. Interpretation limits

The protocol deliberately studies an oracle-geometry setting. It uses simulator-provided instance masks, depth, calibration, and camera poses. Performance therefore does not estimate end-to-end real-world tracking accuracy. MOVi-D and MOVi-E are synthetic and are independent scene samples rather than counterfactual renders of identical scenes. MOVi-E's camera follows a constrained synthetic trajectory and looks toward the scene origin; results may not generalize to handheld rotation, rolling shutter, motion blur, inaccurate calibration, depth-estimation noise, articulated objects, or real occlusion patterns. Pairing is within-video, so the study does not test re-identification across videos.

## 17. Approval checklist

- [x] The primary research question and `Delta_E` estimand are approved.
- [x] MOVi-D is approved as the fixed-camera control and MOVi-E as the moving-camera condition.
- [x] The 150-video minimum and 90/30/30 split per dataset are feasible.
- [x] Pair definitions, quotas, and within-video-only rule are approved.
- [x] Systems A–D and controls G/P/S/N are approved.
- [x] Oracle input and leakage boundaries are approved.
- [x] Primary metric, bootstrap unit, success rule, and threshold procedures are approved.
- [x] Pose-noise grid, strata, and failure-analysis plan are approved.
- [x] Limitations and amendment rules are accepted.

Approval name: User approval recorded in Codex task  Date: August 25, 2026

## 18. Amendment log

| Version | Date | Change | Test results accessed? | Approved by |
|---|---|---|---|---|
| 0.1 | August 25, 2026 | Initial Phase 0 protocol | No | Approved in Codex task |
| 0.2 | August 25, 2026 | Froze exactly 150 non-pilot videos per dataset; excluded pilot videos from all confirmatory pools; clarified pool-level pair quotas, dataset-specific train-only cutoffs, post-split capacity gate, MOVi-D zero-motion reporting, convention manifest fields, and reconstruction reason codes. Primary estimand and success rule unchanged. | No confirmatory results; pilot feasibility diagnostics only | Approved and frozen in Codex task |

## 19. Sources

1. Greff, K. et al. *Kubric: A scalable dataset generator.* CVPR 2022. https://arxiv.org/abs/2203.03570
2. Google Research. *Kubric repository and MOVi dataset documentation.* https://github.com/google-research/kubric
3. Google Research. *MOVi dataset README: MOVi-D and MOVi-E definitions and download information.* https://github.com/google-research/kubric/blob/main/challenges/movi/README.md
4. Google Research. *MOVi-D/E data-generation worker.* https://github.com/google-research/kubric/blob/main/challenges/movi/movi_def_worker.py
5. TensorFlow Datasets. *MOVi-E catalog entry.* https://www.tensorflow.org/datasets/catalog/movi_e

Source descriptions and release details must be revalidated and recorded in the download manifest when implementation begins.
