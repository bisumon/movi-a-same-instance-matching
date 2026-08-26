# Phase 8 regime 3: MOVi-D to MOVi-E transfer

**Protocol:** MOVI-DE-POSE-001 v0.2  
**Seed:** 20260825  
**Design:** frozen MOVi-D clean-D system applied unchanged to the locked MOVi-E test pool  
**Status:** Complete and locked

## Design

The predeclared transfer analysis applies the clean oracle-pose-aligned system D learned in regime 2 to the 2,000 pairs from the 30-video MOVi-E test pool. The complete MOVi-D training standardization, logistic-regression model, selected regularization value (`C = 10`), 90%-recall threshold (`0.790494637825973`), and maximum-F1 threshold (`0.6500617736951679`) are reused unchanged. No MOVi-E samples are used for normalization, fitting, hyperparameter selection, threshold adjustment, or calibration.

Phase 6 defines transfer for clean D only. A/B/C/G/P/S are therefore not refitted or promoted to transfer systems in this regime. The already locked in-domain MOVi-E C and D results are shown only as reference benchmarks on the identical target test pairs.

## Target test results

| Evaluation | AUROC | PR-AUC | False-match rate at locked 90%-recall threshold | Achieved recall | Maximum-F1 operating-point F1 | Brier score | ECE (10 bins) |
|---|---:|---:|---:|---:|---:|---:|---:|
| MOVi-D → MOVi-E transfer D | **0.994029** | **0.994561** | 0.013 | 0.912 | **0.961910** | **0.029024** | **0.015686** |
| In-domain MOVi-E D | 0.994183 | 0.994770 | **0.007** | **0.916** | 0.960606 | 0.030781 | 0.021950 |
| In-domain MOVi-E C | 0.991126 | 0.991915 | 0.009 | 0.906 | 0.948980 | 0.036707 | 0.022087 |

## Paired transfer comparisons

Relative to the in-domain MOVi-E D system, transfer D changes AUROC by **-0.000154**. The paired 95% video-cluster bootstrap interval is **-0.000983 to 0.000934** from 10,000 replicates over the same 30 target test videos. Ranking performance is therefore not statistically resolved from the in-domain D benchmark. No equivalence margin was predeclared, so this is not a formal equivalence claim.

At the independently selected 90%-recall operating thresholds, transfer D has a false-match rate 0.006 higher than in-domain D, with interval **0.001005 to 0.011952**. Its recall difference is -0.004, with interval **-0.015968 to 0.006006**. Thus the main observable transfer cost is threshold behavior rather than ranking quality: the frozen MOVi-D threshold admits about six additional false matches per 1,000 negatives on this target set.

Relative to in-domain MOVi-E C, transfer D improves AUROC by **0.002903**, with paired interval **0.000639 to 0.005547**. This comparison is descriptive because transfer is a secondary analysis, but it shows that the MOVi-D-trained pose-aligned representation retains the ranking advantage seen for D on moving-camera data.

## Interpretation and limitations

The clean D representation transfers strongly from fixed-camera MOVi-D to moving-camera MOVi-E when oracle camera pose is available: almost all in-domain ranking performance is retained without refitting. The slightly worse fixed-threshold false-match rate shows why ranking transfer and operating-threshold transfer should be reported separately.

This result does not demonstrate transfer to real video. Both domains are synthetic and use simulator masks, depth, intrinsics, and camera poses; D uses oracle-pose-aligned visible-surface geometry. Pairs are within-video, and the analysis does not test cross-video re-identification. The comparison also has no predeclared equivalence margin and cannot establish that source- and target-trained systems are practically identical.

Machine-readable results are under `results/movi_de_phase8_regime3/`, and the 2,000 per-pair transfer predictions are under `runs/movi_de_confirmatory/phase8_regime3/d_to_e_transfer/`. The authoritative checksum envelope is `manifests/movi_de/phase8_regime3_d_to_e_transfer_freeze.json`.
