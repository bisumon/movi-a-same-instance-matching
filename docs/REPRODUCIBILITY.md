# Reproducibility protocol

## Immutable decisions

The experiment seed is 20260727. The selected video IDs, split membership, pair
IDs, feature groups, logistic regularization values and thresholds are stored in
versioned JSON/JSONL manifests. Test labels were not used to tune a model,
regularization value, threshold, stratum boundary or failure candidate rule.

## Regeneration path

`run_main_experiments.sh` creates an isolated Python environment and invokes
`run_experiment.py`. The runner downloads the 16 public MOVi-A validation
TFRecord shards unless `--tfrecord-dir` is supplied, then executes:

1. Phase 1 crop, mask, depth and camera-space geometry extraction.
2. Phase 2 split validation and independent within-pool pair generation.
3. Phase 3 frozen ResNet-18 embedding, leakage-safe feature construction, and
   train/dev/test logistic baselines.
4. Phase 4 aggregate/stratified evaluation and paired video-cluster bootstrap.
5. Phase 5 fixed-threshold error selection and gallery generation.

Each stage refuses to overwrite a nonempty incomplete output directory. A
completed manifest acts as the resume marker. All material inputs and outputs
are SHA-256-addressed inside stage manifests.

## Determinism boundary

Selection, split assignment, pair IDs, feature values, models, scores,
bootstrap samples, and Phase 5 selections are seeded and deterministic under
the tested environment. ResNet and logistic code request deterministic
algorithms. Timing measurements are intentionally hardware- and load-dependent;
their exact values are not expected to reproduce across machines.

## Leakage boundary

Instance identity and shape/color/material attributes are used only to build
pairs. Video identity is used only for split integrity and cluster resampling.
Simulator world position/velocity and reconstruction-error diagnostics are
stored separately and used only after predictions are locked for stratified
evaluation and qualitative hypotheses. They are not features.

## Published-artifact audit

Run the unit suite, then verify `REPOSITORY_MANIFEST.sha256` using a standard
SHA-256 checker. The Phase 3 predictions must contain exactly 10,000 unique pair
IDs with 6,000/2,000/2,000 split counts. The Phase 5 manifest must contain 24
unique test pairs and confirm fixed-threshold misclassification.
