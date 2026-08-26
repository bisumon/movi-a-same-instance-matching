# Depth-derived geometry for same-instance matching on MOVi-A/D/E

This repository contains the complete, locked experiment for testing whether
depth-derived camera-space geometry improves within-video same-instance matching
over frozen RGB and ordinary 2D controls. It includes the exact 50-video sample,
30/10/10 video split, 10,000 pair manifest, model configurations, per-pair
predictions, cluster-bootstrap results, and 24-item failure review.

## MOVi-D/E camera-pose extension

The locked MOVi-A experiment remains unchanged. An additive Phase 1 adapter for
the approved MOVi-D fixed-camera control and MOVi-E moving-camera study is
documented in `docs/MOVI_DE_DATA_ADAPTER.md`. The adapter validates the public
D/E schema, extracts matched camera-space and oracle-pose-aligned visible-surface
geometry, enforces pose/reprojection gates, and separates simulator identity and
object state from model inputs. Approved settings are recorded in
`configs/movi_de_protocol.json`. The confirmatory video pools and Phase 5 pairs
are now locked: each dataset has 150 videos split 90/30/30 and 10,000 independent
within-pool pairs split 6,000/2,000/2,000. Pair manifests, audits, and their
checksum envelope are under `manifests/pairs/movi_de/` and
`manifests/movi_de/phase5_pair_manifest_freeze.json`; the sampling specification
is `configs/movi_de_phase5_pairs.json`. Phase 6 system definitions A/B/C/D,
the two geometry-only G diagnostics, pose-only P, shuffled-pose S, and the
36-condition noisy-pose N family are frozen in
`configs/movi_de_phase6_systems.json`, with resolved feature allowlists and a
checksum lock under `manifests/movi_de/`. The Phase 7 pose-noise study is complete:
per-condition metrics, paired video-cluster intervals, and the full results table
are under `results/movi_de_phase7_pose_noise/`, with machine-readable features and
per-pair predictions in `runs/movi_de_confirmatory/phase7/` and the authoritative
lock at `manifests/movi_de/phase7_pose_noise_study_freeze.json`.
Phase 8 regime 1 is also complete: all clean systems were trained, tuned, and
tested on video-disjoint MOVi-E pools. Aggregate and paired results are under
`results/movi_de_phase8_regime1/`; the primary D-minus-C confidence interval and
the authoritative lock are recorded in
`manifests/movi_de/phase8_regime1_in_domain_movi_e_freeze.json`.
Phase 8 regime 2 repeats the same clean systems on video-disjoint, fixed-camera
MOVi-D. Its report and aggregate results are under
`docs/MOVI_DE_PHASE8_REGIME2_IN_DOMAIN_MOVI_D.md` and
`results/movi_de_phase8_regime2/`; machine-readable predictions are under
`runs/movi_de_confirmatory/phase8_regime2/`, and the authoritative lock is
`manifests/movi_de/phase8_regime2_in_domain_movi_d_freeze.json`.
Phase 8 regime 3 is complete: the frozen MOVi-D clean-D scaler, model, and
development thresholds were applied to the locked MOVi-E test pool without
refitting. The report and results are under
`docs/MOVI_DE_PHASE8_REGIME3_D_TO_E_TRANSFER.md` and
`results/movi_de_phase8_regime3/`; per-pair predictions are under
`runs/movi_de_confirmatory/phase8_regime3/`, and the authoritative lock is
`manifests/movi_de/phase8_regime3_d_to_e_transfer_freeze.json`.
Phase 9 evaluates all frozen hypotheses and the transfer question. The primary
pose-alignment hypothesis is supported; secondary mechanism, falsification,
noise, and transfer findings are documented in
`docs/MOVI_DE_PHASE9_CRITERIA_EVALUATION.md`, with machine-readable evidence in
`results/movi_de_phase9/`.
Phase 10 is complete with a checksum-locked 24-item MOVi-D/E failure analysis.
The balanced interactive gallery, review table, selected-case manifest, and
visual overview are under `failure_gallery/movi_de_phase10/`; the analysis is
documented in `docs/MOVI_DE_PHASE10_FAILURE_ANALYSIS.md`, with the authoritative
lock at `manifests/movi_de/phase10_failure_analysis_freeze.json`.

## Main result

On the locked test split, adding depth/3D features to RGB+2D controls improved
AUROC by 0.0607 (paired 95% video-cluster bootstrap CI 0.0427 to 0.0822) and
PR-AUC by 0.0774 (0.0503 to 0.1015). At the fixed dev-selected 90%-recall
operating threshold, false-match rate decreased by 0.0790 (-0.1220 to -0.0349).
These results apply to oracle-like simulator depth/masks, calibrated geometry,
synthetic objects, and a fixed-camera MOVi-A regime; they are not evidence of a
deployable monocular system.

## One-command reproduction

For the MOVi-D/E camera-pose extension, from the repository root run:

```bash
./run_movi_de_experiments.sh
```

This installs the exact dependency lock, runs the integrity tests, reproduces
both video-disjoint in-domain regimes, and runs the frozen MOVi-D-to-MOVi-E
transfer evaluation from the checksum-locked Phase 7 inputs. See
`docs/MOVI_DE_REPRODUCIBILITY.md` for input regeneration, determinism, and audit
boundaries. `FINAL_DELIVERABLES_MOVI_DE.md` is the extension's release index.

For the original MOVi-A study:

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
- `docs/MOVI_DE_RESULTS_TABLES.xlsx` opens with a presentation-ready table that
  matches the MOVi-A reporting fields, followed by the MOVi-D/E clean systems,
  paired confidence intervals, motion strata, pose-noise controls, transfer, and
  measured latency components.
- `failure_gallery/movi_de_phase10/failure_gallery.html` contains the locked
  24-item MOVi-D/E failure gallery.
- `predictions/movi_de/` contains tracked, machine-readable per-pair predictions
  for both pose-noise studies, both in-domain regimes, and D-to-E transfer.

## Sources, licenses and citation

See `docs/SOURCES_AND_LICENSES.md` and `third_party_licenses/`. The repository
does not redistribute MOVi-A TFRecords or the ImageNet-trained ResNet checkpoint.
The original project code currently has no open-source license; see
`LICENSE.md` before redistribution.
