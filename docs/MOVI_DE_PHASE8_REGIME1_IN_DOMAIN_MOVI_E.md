# Phase 8 regime 1: in-domain MOVi-E

**Protocol:** MOVI-DE-POSE-001 v0.2  
**Seed:** 20260825  
**Design:** train, tune, and test on locked, video-disjoint MOVi-E pools  
**Status:** Complete and locked

## Design

All systems use the same 90 training, 30 development, and 30 test videos and the same independently generated 6,000/2,000/2,000 within-video pair manifests. No video appears in more than one pool. Standardization and logistic-regression fitting use training pairs only. Development AUROC selects `C` from `{0.01, 0.1, 1, 10, 100}`, with ties favoring stronger regularization. Development data also lock each system's highest threshold reaching 90% recall and maximum-F1 threshold. The complete configuration lock is written before test scoring.

Eight clean systems were evaluated: RGB A; RGB+2D B; camera-space geometry C; pose-aligned geometry D; camera and pose-aligned geometry-only diagnostics G-C and G-D; pose-only P; and shuffled-pose S. The 36-condition noisy-pose family remains the separate completed Phase 7 study and is not refit here.

## Test results

| System | Selected C | AUROC | PR-AUC | False-match rate at dev-locked 90%-recall threshold | Achieved recall |
|---|---:|---:|---:|---:|---:|
| A — RGB | 0.01 | 0.949382 | 0.959000 | 0.154 | 0.891 |
| B — RGB + 2D | 100 | 0.981201 | 0.983867 | 0.045 | 0.903 |
| C — Camera-space geometry | 10 | 0.991126 | 0.991915 | 0.009 | 0.906 |
| D — Pose-aligned geometry | 1 | **0.994183** | **0.994770** | **0.007** | **0.916** |
| G-C — Camera geometry only | 1 | 0.987158 | 0.988290 | 0.021 | 0.909 |
| G-D — Pose-aligned geometry only | 10 | 0.992365 | 0.992845 | 0.016 | 0.915 |
| P — Pose only | 100 | 0.499860 | 0.499675 | 0.874 | 0.866 |
| S — Shuffled pose | 100 | 0.989789 | 0.990814 | 0.020 | 0.909 |

## Primary result

The primary estimand is test `AUROC_D − AUROC_C` on identical MOVi-E pairs. The observed difference is **0.003057**, with a paired 95% video-cluster bootstrap interval of **0.001131 to 0.005166** from 10,000 replicates over the 30 test videos. The interval lies entirely above zero, so the predeclared primary success rule is met.

At the separately dev-locked 90%-recall thresholds, D's false-match rate is 0.007 versus 0.009 for C, a difference of -0.002 (95% paired cluster interval -0.006 to 0.001994). Recall is 0.916 versus 0.906, a difference of 0.010 (-0.005976 to 0.026104). Both point estimates favor D, but the operational differences are not individually resolved from zero at the 95% level.

## Controls and interpretation

Pose-only P is at chance (AUROC 0.499860), providing no evidence that relative camera motion directly predicts pair identity. Shuffled-pose S is worse than clean D by 0.004394 AUROC (S − D 95% CI -0.006759 to -0.002317) and has a 0.013 higher false-match rate (0.006979 to 0.019980). This supports the importance of correct frame-to-pose correspondence rather than pose availability alone.

Pose-aligned geometry without RGB or 2D controls remains strong, but G-D is below D by 0.001818 AUROC (-0.003760 to -0.000489) and has a 0.009 higher false-match rate (0.001990 to 0.017034). Appearance and 2D information therefore provide a small complementary benefit. D also has lower test Brier score than C (0.03078 versus 0.03671) and lower log loss (0.10347 versus 0.12369).

The primary gain is statistically resolved but numerically modest because C is already very strong. These results apply to simulator-provided masks, depth, calibration, and camera pose; they do not establish end-to-end real-world tracking accuracy. Pairs remain within-video, and MOVi-E uses a constrained synthetic camera trajectory. The shuffled extent transformation operates on the frozen visible-surface sufficient statistics rather than recomputing every raw point cloud.

The full aggregate, calibration, latency, and paired-difference results are in `results/movi_de_phase8_regime1/movi_e_in_domain_results.json`; the compact table is `movi_e_in_domain_results.csv`. Machine-readable per-pair predictions are retained under `runs/movi_de_confirmatory/phase8_regime1/in_domain_movi_e/`. The authoritative checksum envelope is `manifests/movi_de/phase8_regime1_in_domain_movi_e_freeze.json`.
