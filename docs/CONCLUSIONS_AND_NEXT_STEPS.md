# Conclusions and next steps

## Conclusion

Depth-derived camera-space geometry improved same-instance matching on the
locked MOVi-A test set. Relative to RGB+2D controls, RGB+2D+3D increased AUROC
from 0.8278 to 0.8884 (delta 0.0607; paired 95% video-cluster bootstrap CI
0.0427 to 0.0822) and PR-AUC from 0.8091 to 0.8864 (delta 0.0774; CI 0.0503 to
0.1015). Locked-threshold F1 improved by 0.0350 (CI 0.0175 to 0.0538). At the
dev-selected 90%-recall threshold, false-match rate fell from 0.374 to 0.295
(delta -0.079; CI -0.1220 to -0.0349). The achieved-recall delta was positive
but its interval crossed zero.

The largest AUROC gain was against matched-attribute hard negatives (+0.0965),
supporting the hypothesis that depth-derived geometry contributes information
not captured by RGB appearance and ordinary 2D controls. Gains were present in
all temporal-gap, visibility and motion strata. The balanced failure review
still shows recurring matched-attribute distractors, long-gap motion,
occlusion/truncated masks and biased depth estimates.

## Limitations that bound the claim

### Oracle-geometry limitation

The experiment uses simulator-provided instance masks, depth and camera
calibration to construct crops and camera-space geometry. These are oracle-like
inputs compared with a deployed RGB system. The result estimates the value of
clean geometry when correspondence proposals and visible pixels are already
available; it does not establish that predicted monocular depth, predicted
masks or estimated camera calibration will deliver the same gain. Simulator
velocity and reconstruction errors are evaluation-only, but the inference-time
depth/mask advantage remains privileged.

### Fixed-camera limitation

MOVi-A uses CLEVR-style synthetic scenes viewed from a camera that is fixed
within a video, with only small camera-position jitter across generated scenes
and a consistent look-at convention. The pipeline therefore avoids the camera
pose estimation, moving-camera parallax, calibration drift and dynamic
backgrounds encountered in real video. Performance should not be generalized
to moving-camera or multi-camera tracking without a separate evaluation.

Additional limits include 50 selected validation videos and only 10 test-video
clusters, within-video pairs only, synthetic low-diversity objects, a frozen
ResNet-18 encoder, logistic models, and thresholds selected for this benchmark.
The cluster bootstrap accounts for pair dependence within the 10 videos but
cannot create broader domain diversity.

## Recommended next steps

1. Replace oracle masks and depth with predicted instance masks and monocular or
   stereo depth, retain uncertainty estimates, and measure degradation from the
   oracle ceiling.
2. Add moving-camera MOVi variants and real videos; estimate camera pose and
   evaluate geometry after compensating for ego-motion.
3. Add cross-video negatives and identity matches where labels permit, while
   keeping video-disjoint splits and cluster-level inference.
4. Expand the number and diversity of test videos before comparing more
   backbones, learned fusion models or end-to-end metric learning.
5. Pre-register thresholds and strata for the expanded benchmark, then repeat
   the paired video-cluster analysis and qualitative error review.

## Bottom line

The study provides evidence that clean 3D geometry is useful for resolving
same-instance ambiguity in this controlled benchmark. The next experiment
should test whether that advantage survives predicted geometry and camera
motion; until then, the result is best interpreted as an oracle-geometry upper
bound under a fixed-camera synthetic setting.
