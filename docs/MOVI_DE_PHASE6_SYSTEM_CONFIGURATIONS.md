# MOVi-D/E Phase 6 system configurations

**Protocol:** MOVI-DE-POSE-001 v0.2  
**Seed:** 20260825  
**Status:** Frozen before confirmatory feature construction, model fitting, development tuning, or test access

Phase 6 adds nine machine-readable system definitions. A and B are the appearance and 2D baselines. C is the pose-unaware camera-space geometry baseline. D is the primary oracle-pose-aligned visible-surface system. G is operationalized as two diagnostics—camera-space geometry only and pose-aligned geometry only—so the coordinate comparison remains explicit. P receives only camera displacement, relative rotation, and normalized displacement. S uses deterministically deranged poses. N is a 36-condition pose-noise sensitivity family.

## Comparable geometry

C and D each contain 31 pair features. Both receive the same one RGB, eleven 2D, and seven radial-depth features. C then receives twelve camera-space centroid/extent features; D receives twelve directly corresponding world-aligned centroid/extent features. Thus D receives no extra scalar merely because world coordinates are available. The two G variants each contain the same 19 geometry features and exclude RGB and 2D controls.

Camera pose scalars are direct model features only for P. D, pose-aligned G, S, and N may use a camera pose only to transform the same visible-surface geometry. Simulator identity, asset, category, dynamic/static state, ground-truth object state, labels, negative difficulty, and identifiers are prohibited from model matrices. Identifiers remain available only for integrity joins and cluster definitions.

## Shuffled and noisy pose

S sorts eligible frame keys by a seeded SHA-256 rank independently in each dataset and split, then cyclically shifts pose assignments by one. This is a derangement whenever at least two frames exist. The assigned frame pose is reused in every pair containing that frame. S is trained and tuned as an independent system using shuffled training and development features. On fixed-camera MOVi-D, shuffled clean poses are expected to be structurally identical.

N evaluates the full 6 × 6 Cartesian grid from the protocol: translation standard deviations of 0, 0.01, 0.05, 0.10, 0.25, and 0.50 scene units crossed with rotation standard deviations of 0, 0.1, 0.5, 1, 2, and 5 degrees. Translation noise is independent Gaussian noise on world x/y/z camera-position components. Rotation uses a Gaussian axis-angle magnitude, a uniform axis, and camera-local post-composition. Noise is seeded per frame and condition and held fixed across all pairs. The clean D model and train-fitted standardization are applied without refitting; the zero-noise condition must reproduce clean D features exactly.

## Training and transfer

Every independently fitted system uses standardized logistic regression, train-only normalization and fitting, and `C = {0.01, 0.1, 1, 10, 100}`. Development AUROC selects C with ties favoring the smaller value. Development data also lock the highest threshold reaching 90% recall and the maximum-F1 threshold. Test data cannot alter any choice. The secondary D-to-E analysis applies the MOVi-D D model, standardization, and D-development thresholds to MOVi-E without refitting.

The source of truth is `configs/movi_de_phase6_systems.json`. `manifests/movi_de/phase6_system_configuration_freeze.json` is the authoritative checksum envelope. Any feature, pose-treatment, noise-grid, fit-policy, or threshold-rule change requires a prospective amendment before confirmatory fitting.
