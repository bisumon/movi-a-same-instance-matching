# Depth-derived 3D geometry for same-instance matching on MOVi-A

This repository contains the complete, locked experiment for testing whether
depth-derived camera-space geometry improves within-video same-instance matching
over frozen RGB and ordinary 2D controls. It includes the exact 50-video sample,
30/10/10 video split, 10,000 pair manifest, model configurations, per-pair
predictions, cluster-bootstrap results, and 24-item failure review.

## Main result

On the locked test split, adding depth/3D features to RGB+2D controls improved
AUROC by 0.0607 (paired 95% video-cluster bootstrap CI 0.0427 to 0.0822) and
PR-AUC by 0.0774 (0.0503 to 0.1015). At the fixed dev-selected 90%-recall
operating threshold, false-match rate decreased by 0.0790 (-0.1220 to -0.0349).
These results apply to oracle-like simulator depth/masks, calibrated geometry,
synthetic objects, and a fixed-camera MOVi-A regime; they are not evidence of a
deployable monocular system.

## One-command reproduction

Requirements: Python 3.12, internet access, sufficient disk space for the 16
public validation TFRecord shards and generated crops, and several CPU-hours on
an Intel-only machine. From the repository root:

```bash
./run_main_experiments.sh
```

The command creates an isolated `.venv`, installs the tested dependency lock,
downloads MOVi-A validation, and runs Phases 1-5 into `runs/main`. It is
resumable at completed stage manifests. To reuse existing TFRecords or choose a
device/run directory:

```bash
./run_main_experiments.sh \
  --tfrecord-dir /path/to/movi_a_validation \
  --run-dir runs/reproduction \
  --device cpu
```

The ResNet-18 ImageNet-1K V1 checkpoint is downloaded automatically by
torchvision and its SHA-256 is recorded in the embedding manifest.

## Locked protocol

- Seed: `20260727`.
- Dataset: MOVi-A 128x128 validation, 50 preselected videos.
- Split: 30 train / 10 dev / 10 test videos, with no video overlap.
- Pairs: 6,000 / 2,000 / 2,000, generated independently within each pool.
- Pair mix: 50% positives, 25% matched-attribute hard negatives, 25% easy negatives.
- All benchmark pairs are within-video; no cross-video pairs are included.
- Encoder: frozen torchvision ResNet-18 ImageNet-1K V1, 512-D pooled embeddings.
- Models: standardized logistic regressions trained on train only.
- Regularization and operating thresholds: selected on dev only.
- Inference configurations: A RGB only; B RGB+2D; C RGB+2D+depth/3D; geometry-only diagnostic.
- Inference: 2D controls, masked depth and depth-derived camera-space geometry only.
- Evaluation-only diagnostics: simulator velocity and reconstruction errors are never model features.
- Inference: 10,000 paired video-cluster bootstrap replicates over the 10 test videos.

The machine-readable source of truth is `configs/protocol.json` plus the locked
configuration and stage manifests.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Selection, extraction, pairing, baseline, evaluation and failure-review code |
| `tests/` | Unit and integrity tests for all phases |
| `configs/` | Locked experiment protocol |
| `manifests/selection/` | Inventory, eligibility audit, selected 50 and seeded split |
| `manifests/splits/` | Locked 30/10/10 split definitions |
| `manifests/pairs/` | Seeded train/dev/test pair manifests and balance audit |
| `artifacts/phase3/` | Locked model configuration, compact fitted models and result manifests |
| `predictions/` | Machine-readable per-pair scores for all 10,000 pairs |
| `results/phase4/` | Aggregate, per-video, stratified, bootstrap, paired-difference and latency outputs |
| `failure_gallery/` | 24 unique fixed-threshold errors, crop assets, CSV/JSONL and HTML gallery |
| `docs/` | Results workbook, conclusions, reproducibility notes, and source/license documentation |

Raw TFRecords, generated crops, RGB embeddings, and feature matrices are not
redistributed in this repository. They are deterministic/reproducible
intermediates created by the one-command runner. Published hashes in the stage
manifests allow regenerated artifacts to be audited; runtime latency fields can
vary by hardware and system load.

## Tests

After installing `requirements-lock.txt`:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

## Results and qualitative review

- `docs/results_tables.xlsx` consolidates baseline/control metrics, confidence
  intervals, pair strata, pair-control balance and latency.
- `docs/CONCLUSIONS_AND_NEXT_STEPS.md` and `.docx` summarize conclusions and
  limitations.
- Open `failure_gallery/phase5_error_gallery.html` for the side-by-side review.
- `predictions/phase3_predictions.jsonl` contains the per-sample scores used by
  all Phase 4 and Phase 5 analyses.

## Sources, licenses and citation

See `docs/SOURCES_AND_LICENSES.md` and `third_party_licenses/`. The repository
does not redistribute MOVi-A TFRecords or the ImageNet-trained ResNet checkpoint.
The original project code currently has no open-source license; see
`LICENSE.md` before redistribution.
