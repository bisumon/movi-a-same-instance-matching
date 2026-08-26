# MOVi-D/E Phase 7 pose-noise study

**Protocol:** MOVI-DE-POSE-001 v0.2  
**Configuration:** Phase 6 systems v1.0.0  
**Seed:** 20260825  
**Status:** Complete and locked

## Question and design

This study measures how the clean system D classifier responds when camera pose becomes inaccurate. MOVi-D and MOVi-E each use their independently fitted clean D model: frozen ResNet-18 RGB similarity, 2D controls, radial-depth controls, and oracle-pose-aligned visible-surface geometry. Normalization and logistic-regression fitting use training pairs only. Regularization and the 90%-recall and maximum-F1 thresholds are selected on the clean development pool. The fitted clean model and thresholds are then applied unchanged to all 36 pose-noise conditions; no noisy condition is refit or retuned.

The grid crosses translation standard deviations of 0, 0.01, 0.05, 0.10, 0.25, and 0.50 scene units with rotation standard deviations of 0, 0.1, 0.5, 1, 2, and 5 degrees. Perturbations are deterministic and independent per dataset, split, frame, and condition. A frame receives one perturbation that is reused by every observation and pair containing it. RGB, masks, depth, labels, and pair membership never change.

Visible-surface centroids and extents are the frozen Phase 1 sufficient statistics. Centroids receive the corresponding rigid pose perturbation. For nonzero rotation, the clean axis-aligned q05–q95 extent vector is propagated as `abs(delta_rotation) × clean_extent`; raw masked point clouds are not retained or re-quantiled for every noise condition. The zero-noise path uses the clean summary directly and is byte-identical to system D. Accordingly, this is a sensitivity study of the declared sufficient-statistics representation, not a claim about every possible point-cloud implementation.

## Main results

All values below are on the locked 2,000-pair test pool. AUROC differences and 95% intervals compare each condition with clean D using 10,000 paired video-cluster bootstrap replicates over the 30 test videos.

| Dataset | Condition | Test AUROC | Noise − clean AUROC (95% CI) | Recall at clean D 90%-recall threshold |
|---|---:|---:|---:|---:|
| MOVi-D | Clean (0, 0) | 0.984952 | 0.000000 (0.000000, 0.000000) | 0.875 |
| MOVi-D | Rotation 5° only | 0.980915 | -0.004037 (-0.006312, -0.002260) | 0.860 |
| MOVi-D | Translation 0.50 only | 0.975144 | -0.009808 (-0.014091, -0.006244) | 0.828 |
| MOVi-D | Translation 0.50 + rotation 5° | 0.968656 | -0.016296 (-0.020680, -0.012199) | 0.812 |
| MOVi-E | Clean (0, 0) | 0.994183 | 0.000000 (0.000000, 0.000000) | 0.916 |
| MOVi-E | Rotation 5° only | 0.988140 | -0.006043 (-0.009199, -0.003457) | 0.856 |
| MOVi-E | Translation 0.50 only | 0.986574 | -0.007609 (-0.011320, -0.004464) | 0.850 |
| MOVi-E | Translation 0.50 + rotation 5° | 0.977072 | -0.017111 (-0.024285, -0.011295) | 0.773 |

Small perturbations were generally tolerated. Rotation through 2 degrees and translation through 0.10 scene units produced small AUROC changes whose paired intervals included zero in both datasets. At 0.25 translation, MOVi-D showed a small detectable decline (-0.000948; 95% CI -0.001874 to -0.000116), while the MOVi-E interval still included zero. The 5-degree, 0.50-unit, and combined maximum conditions produced clear degradation. The largest decline occurred under the maximum combined condition in each dataset.

The fixed operating threshold is more sensitive than AUROC. Under maximum combined noise, MOVi-E recall fell from 0.916 to 0.773 even though AUROC remained 0.977. This is expected: AUROC measures ranking across all thresholds, whereas the operating point remains locked to clean development behavior.

## Interpretation and limitations

The result supports a graded robustness conclusion: the pose-aligned system is stable to small errors under this synthetic setup but is not invariant to large pose errors. Translation and rotation errors compound at the top of the grid. MOVi-E shows the largest operating-point recall loss, which is relevant because it is the moving-camera condition.

MOVi-D noise is an artificial falsification stress test applied to a dataset whose real camera is fixed; it does not describe naturally occurring MOVi-D pose uncertainty. MOVi-E supplies simulator camera poses, masks, and depth, so this remains an oracle-geometry experiment. The analysis does not establish real-sensor robustness, monocular depth robustness, or deployable pose-estimation performance. It also does not by itself test whether D outperforms C; that requires the separate clean-system comparison on identical test pairs.

The complete table contains 72 dataset/condition combinations for each of development and test (144 metric rows) in `results/movi_de_phase7_pose_noise/phase7_pose_noise_results_table.csv`. Machine-readable per-pair scores for all conditions are retained under `runs/movi_de_confirmatory/phase7/`. The authoritative checksum envelope is `manifests/movi_de/phase7_pose_noise_study_freeze.json`.
