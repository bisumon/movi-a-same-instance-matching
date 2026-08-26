# MOVi-D/E camera-pose extension: conclusions and next steps

## Bottom line

The primary hypothesis passed. On the locked moving-camera MOVi-E test pool,
pose-aligned geometry (D) improved AUROC over camera-space geometry (C) by
0.003057; the paired 95% video-cluster bootstrap interval was 0.001131 to
0.005166. The improvement was small in absolute terms but reproducible under the
predeclared criterion.

The mechanism evidence is coherent. The D-minus-C advantage was larger in the
high-motion than low-motion stratum for camera translation (difference 0.005408,
95% CI 0.001037 to 0.011271) and camera rotation (0.006070, 95% CI 0.001185 to
0.011617). Shuffling pose removed the advantage, and injected pose noise degraded
performance as the perturbation increased.

The fixed-camera MOVi-D comparison was consistent with the intended
falsification: D minus C was 0.001615 (95% CI -0.002037 to 0.004542). Because no
equivalence margin was frozen, this is evidence of a small unresolved difference,
not proof that the systems are equivalent.

Transfer preserved ranking but not the operating point. The unchanged MOVi-D D
model achieved nearly the same MOVi-E AUROC as in-domain D (difference -0.000154,
95% CI -0.000983 to 0.000934), but false-match rate was 0.006 higher (95% CI
0.001005 to 0.011952) at the MOVi-D-selected threshold. Cross-domain threshold
calibration therefore matters even when discrimination transfers well.

## Limitations

This is an oracle-geometry experiment. It uses simulator masks, depth,
intrinsics, and camera poses. The results isolate whether correct pose alignment
can help; they do not demonstrate a deployable system using estimated masks,
monocular depth, or visual-inertial pose. The failure gallery already shows that
mask truncation, occlusion, matched-attribute distractors, long temporal gaps,
and depth bias remain important error sources.

MOVi-D is a synthetic fixed-camera control. Its result should not be generalized
to all static-camera footage, and the comparison with MOVi-E is not a paired
counterfactual rendering of identical scenes. Both datasets are synthetic, all
benchmark pairs are within-video, and the study does not cover cross-video
re-identification, rolling shutter, real sensor noise, dynamic backgrounds, or
camera-pose estimator failures.

## Recommended next steps

1. Replace oracle pose, depth, and masks one component at a time with estimated
   inputs and trace the accuracy loss to each estimator.
2. Add realistic correlated pose noise and drift, including trajectory bias and
   time-synchronization error, rather than only independent perturbations.
3. Render matched fixed-camera and moving-camera versions of the same scenes to
   create a true camera-motion counterfactual and sharpen the D-minus-C mechanism
   test.
4. Evaluate threshold recalibration and unsupervised domain adaptation while
   preserving a locked target test set.
5. Move to real handheld and fixed-camera video, expand camera trajectories, and
   add cross-video matching only after the within-video estimator-noise study is
   complete.
