# Phase 9: evaluation against the frozen criteria

**Protocol:** MOVI-DE-POSE-001 v0.2  
**Seed:** 20260825  
**Inputs:** checksum-locked Phase 7 and Phase 8 results  
**Overall conclusion:** The primary camera-pose hypothesis is supported.

The repository does not contain a separate document titled “Phase 9.” This evaluation therefore uses the six frozen hypotheses H1–H6, the sole confirmatory decision rule, and the predeclared transfer question in the approved protocol. Only H1 has a formal confirmatory success rule. Status labels for H2–H6 and transfer summarize secondary evidence and do not create post-test success criteria.

## Decision summary

| Criterion | Result | Evaluation |
|---|---|---|
| H1 — Pose-alignment benefit on MOVi-E | D − C AUROC = 0.003057; 95% CI 0.001131 to 0.005166 | **Supported; primary rule passed** |
| H2 — Lower false-match rate without recall loss | FMR difference -0.002 (-0.006 to 0.001994); recall difference +0.010 (-0.005976 to 0.026104) | Directionally favorable, not statistically resolved |
| H3 — Benefit increases with camera motion | High-minus-low D − C benefit: +0.005408 for translation and +0.006070 for rotation; both intervals above zero | **Supported within MOVi-E** |
| H4 — Little or no fixed-camera benefit | MOVi-D D − C AUROC = 0.001615 (-0.002037 to 0.004542) | Consistent with H4, but not an equivalence result |
| H5 — Correct pose correspondence matters | S − D AUROC = -0.004394 (-0.006759 to -0.002317); S does not beat C | **Supported** |
| H6 — Pose-noise sensitivity | Large translation, rotation, and combined errors reduce AUROC in both datasets; small errors are tolerated | **Supported as a graded endpoint pattern** |
| D-to-E transfer | Transfer AUROC 0.994029; transfer minus in-domain D = -0.000154 (-0.000983 to 0.000934) | Ranking transfers; operating threshold partially degrades |

## Primary criterion: supported

On the identical 2,000 locked MOVi-E test pairs, D achieves AUROC 0.994183 versus 0.991126 for C. The paired D-minus-C difference is 0.003057, with a two-sided 95% video-cluster bootstrap interval of 0.001131 to 0.005166. Because the entire interval is above zero, the exact frozen primary success rule is satisfied.

The effect is statistically resolved but numerically modest. It supports the claim that correct pose alignment improves discrimination in this controlled moving-camera experiment; it does not by itself imply a large operational gain or real-world deployability.

## Secondary criteria

### H2 — Operational benefit: favorable point estimates, unresolved intervals

At separately dev-locked thresholds targeting 90% recall, D reduces MOVi-E false-match rate from 0.009 to 0.007 and increases achieved recall from 0.906 to 0.916. The paired false-match-rate difference is -0.002, with interval -0.006 to 0.001994. The recall difference is +0.010, with interval -0.005976 to 0.026104. Both intervals include zero. The direction is consistent with H2, but the operational benefit is not statistically resolved, and no “material recall loss” margin was predeclared.

### H3 — Motion interaction: supported within MOVi-E; cross-dataset interaction unresolved

The predeclared MOVi-E-training tertiles show a clear within-dataset motion pattern:

| Motion measure | Low-motion D − C AUROC | High-motion D − C AUROC | High-minus-low difference (95% CI) |
|---|---:|---:|---:|
| Camera displacement | -0.000298 | 0.005111 | **0.005408 (0.001037 to 0.011271)** |
| Relative camera rotation | 0.000071 | 0.006141 | **0.006070 (0.001185 to 0.011617)** |
| Normalized camera displacement | -0.000008 | 0.005317 | **0.005325 (0.000880 to 0.010990)** |

All three high-minus-low intervals are above zero. This supports the intended mechanism: correct pose alignment matters most when the camera moves more.

The separate cross-dataset difference-in-differences is +0.001442, with interval -0.002092 to 0.005585. It is not statistically resolved. This weaker result is not contradictory: MOVi-D and MOVi-E are independent scene samples, while the within-E analysis compares motion strata inside the same locked target dataset.

### H4 — Fixed-camera falsification: consistent, not proven equivalent

On MOVi-D, D exceeds C by only 0.001615 AUROC, with interval -0.002037 to 0.004542. The interval spans zero and no large unexpected advantage is observed. This is consistent with H4 and reduces concern that D benefits from an unintended information channel. Because no equivalence margin was frozen, it cannot prove that D and C are equivalent.

### H5 — Correspondence control: supported

On MOVi-E, shuffled pose S is 0.004394 AUROC below clean D, with an interval entirely below zero. S is also 0.001337 below C, although that S-minus-C interval includes zero. The shuffled system’s false-match rate is 0.013 higher than D. Pose-only P remains at chance (AUROC 0.499860). Together, these controls indicate that correct frame-to-pose correspondence—not pose availability or a direct pose-label shortcut—drives the benefit.

### H6 — Pose-noise sensitivity: supported with graded robustness

Small perturbations through roughly 0.10 scene units of translation and 2 degrees of rotation produce small AUROC changes whose intervals generally include zero. Large errors clearly degrade performance. Under maximum combined noise, AUROC falls by 0.016296 on MOVi-D and 0.017111 on MOVi-E, with both intervals entirely below zero; MOVi-E recall at the clean threshold falls from 0.916 to 0.773. This supports a graded robustness conclusion, not strict monotonic degradation at every one of the 36 grid steps.

### Transfer — ranking succeeds; threshold transfer is imperfect

The frozen MOVi-D D model achieves 0.994029 AUROC on MOVi-E, only 0.000154 below the MOVi-E-trained D model; the paired interval spans zero. Ranking therefore transfers extremely well. However, its unchanged MOVi-D threshold raises false-match rate from 0.007 to 0.013 relative to in-domain D, a resolved increase of 0.006 (0.001005 to 0.011952). Target-free model ranking is strong, but deployment-threshold calibration needs attention.

## Final assessment

The experiment supports the primary research claim: oracle pose alignment improves same-instance discrimination under camera motion. The strongest mechanistic evidence is the combination of a positive primary MOVi-E result, larger benefits at higher camera motion, degradation under shuffled correspondence, and degradation under large pose errors. Operational false-match improvement and the cross-dataset difference-in-differences remain unresolved, while transfer shows that ranking generalizes better than the fixed operating threshold.

The conclusion remains limited to synthetic within-video matching with simulator-provided masks, depth, intrinsics, and camera pose. It does not establish performance with estimated geometry, real camera motion, or cross-video re-identification. No equivalence claim should be made for MOVi-D or transfer because no practical equivalence margin was predeclared.

Machine-readable evidence is in `results/movi_de_phase9/phase9_criteria_evaluation.json`, with the compact status table in `results/movi_de_phase9/phase9_criteria_summary.csv`.
