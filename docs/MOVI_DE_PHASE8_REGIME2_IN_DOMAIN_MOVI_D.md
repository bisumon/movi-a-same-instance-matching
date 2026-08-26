# Phase 8 regime 2: in-domain MOVi-D

**Protocol:** MOVI-DE-POSE-001 v0.2  
**Seed:** 20260825  
**Design:** train, tune, and test on locked, video-disjoint MOVi-D pools  
**Status:** Complete and locked

## Design

Regime 2 repeats the eight clean Phase 6 configurations on the fixed-camera MOVi-D control. The locked pools contain 90 training, 30 development, and 30 test videos and independently generated 6,000/2,000/2,000 within-video pairs. Training-only standardization and fitting, development-only regularization and threshold selection, and post-lock test scoring are identical to MOVi-E regime 1.

Every MOVi-D video has zero within-video camera translation and rotation, and every pose-only P input is exactly zero. The absolute static camera pose nevertheless differs across the 150 videos. The frozen S algorithm deranges frame poses across the entire dataset/split, so it swaps different static viewpoints across videos and remains a nontrivial diagnostic. This does not change the fixed-camera status: within each actual video, the camera never moves.

## Test results

| System | Selected C | AUROC | PR-AUC | False-match rate at dev-locked 90%-recall threshold | Achieved recall |
|---|---:|---:|---:|---:|---:|
| A — RGB | 0.01 | 0.935493 | 0.951955 | 0.168 | 0.882 |
| B — RGB + 2D | 10 | 0.971199 | 0.975732 | 0.067 | 0.900 |
| C — Camera-space geometry | 100 | 0.983337 | 0.984806 | 0.034 | 0.881 |
| D — Pose-aligned geometry | 10 | **0.984952** | **0.986031** | **0.032** | 0.875 |
| G-C — Camera geometry only | 100 | 0.979617 | 0.982192 | 0.034 | 0.878 |
| G-D — Pose-aligned geometry only | 1 | 0.979247 | 0.980629 | 0.057 | 0.885 |
| P — Pose only | 0.01 | 0.500000 | 0.500000 | 1.000 | 1.000 |
| S — Shuffled pose | 10 | 0.982156 | 0.983949 | 0.041 | 0.879 |

## Fixed-camera falsification result

The descriptive fixed-camera estimand `AUROC_D − AUROC_C` is **0.001615**, with a paired 95% video-cluster bootstrap interval of **-0.002037 to 0.004542** from 10,000 replicates over the 30 test videos. The interval spans zero, so MOVi-D provides no resolved evidence of a pose-alignment advantage. This contrasts with MOVi-E regime 1, where the corresponding interval was entirely positive.

At the separately locked operating thresholds, D's false-match rate is 0.032 versus 0.034 for C, a difference of -0.002 (-0.013958 to 0.010000). Recall is 0.875 versus 0.881, a difference of -0.006 (-0.020020 to 0.008938). Neither operational difference is resolved from zero.

P is exactly chance because its three declared camera-motion inputs are structural zeros. Shuffled pose is 0.002796 AUROC below D, but its interval (-0.005602 to 0.000814) spans zero. Because the static viewpoint varies between videos, the global shuffle changes the coordinate transform; however, unlike MOVi-E, disrupting that transform does not produce a clearly resolved test effect.

## Interpretation and limitations

Regime 2 is consistent with the camera-motion hypothesis: the benefit of correct pose alignment is resolved in moving-camera MOVi-E but not in fixed-camera MOVi-D. It is not an equivalence test—no practical equivalence margin was predeclared—so the MOVi-D interval must not be interpreted as proof that C and D are identical. The D point estimate remains slightly positive.

The data still use simulator masks, depth, calibration, and camera pose, and pairs remain within-video. Results do not estimate real-world or cross-video re-identification performance. Camera-space and world-space component descriptors can differ by a video-specific static rotation even without within-video motion, so exact numeric equality between C and D is not expected.

The full results are in `results/movi_de_phase8_regime2/`, with per-pair predictions under `runs/movi_de_confirmatory/phase8_regime2/in_domain_movi_d/`. The authoritative checksum envelope is `manifests/movi_de/phase8_regime2_in_domain_movi_d_freeze.json`.
