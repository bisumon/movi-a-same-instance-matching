# MOVi-D/E reproducibility protocol

## One-command experiment reproduction

From the repository root, run:

```bash
./run_movi_de_experiments.sh
```

Use Python 3.11 or later. The launcher defaults to `python3`; set `PYTHON_BIN`
if your interpreter uses another name.

The command creates `.venv-movi-de`, installs the exact lock file, downloads the
public validation shards when needed, runs the full unit/integrity suite,
regenerates Phase 1 and Phase 7, rebuilds all eight clean-system feature matrices, trains and
tunes the video-disjoint MOVi-E and MOVi-D regimes, evaluates the locked test
pools with 10,000 paired video-cluster bootstrap replicates, and applies the
unchanged MOVi-D clean-D model and development thresholds to MOVi-E.

The default command regenerates the checksum-locked Phase 7 inputs from the
public MOVi-D/E TFRecords, the locked 150-video selections, the locked
90/30/30 pools, the locked 6,000/2,000/2,000 pair manifests, oracle masks/depth,
camera calibration, and frozen ResNet-18 embeddings. To reuse a completed Phase 7 directory:

```bash
./run_movi_de_experiments.sh \
  --source-run /path/to/completed/phase7 \
  --run-dir runs/movi_de_independent_reproduction
```

To reuse downloaded raw data while regenerating all derived inputs:

```bash
./run_movi_de_experiments.sh \
  --tfrecord-root /path/containing/movi_d_validation_and_movi_e_validation \
  --run-dir runs/movi_de_independent_reproduction
```

The raw-data regeneration commands and gates are documented phase by phase in
`MOVI_DE_DATA_ADAPTER.md`, `MOVI_DE_CONFIRMATORY_VIDEO_SELECTION.md`,
`MOVI_DE_PHASE5_PAIR_MANIFESTS.md`, and `MOVI_DE_PHASE7_POSE_NOISE_STUDY.md`.

## Immutable decisions

- Seed: `20260825`.
- Data: MOVi-D and MOVi-E validation, 150 videos per dataset.
- Splits: 90 train / 30 development / 30 test videos per dataset, video-disjoint.
- Pairs: 6,000 / 2,000 / 2,000 independently generated within locked pools.
- Model selection and operating thresholds: development only.
- Confirmatory endpoint: MOVi-E test AUROC difference, D minus C, with a paired
  95% video-cluster bootstrap confidence interval.
- Test labels were not used for video selection, pair rules, model tuning,
  threshold selection, motion boundaries, or failure-gallery candidate rules.

## Determinism and hardware boundary

Selections, pair IDs, features, configuration resolution, model fitting, scores,
bootstrap draws, and failure selection are seeded and checksum-addressed.
Latency is hardware- and system-load-dependent and is therefore expected to
change when rerun. The published latency table records the observed machine,
batch, and measurement scope.

## Published artifact audit

Run the unit suite with:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Then verify `manifests/movi_de/final_release_manifest.json`. It records SHA-256
digests and row/count checks for the final release artifacts. The Phase 5 pair
freeze, Phase 6 configuration freeze, Phase 7 noise freeze, three Phase 8 regime
freezes, Phase 9 evidence manifest, and Phase 10 failure freeze remain the
authoritative stage-level locks.
