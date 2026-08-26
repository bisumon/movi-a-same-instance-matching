# MOVi-D/E Phase 1 data adapter

The MOVi-D/E extension is additive: the locked MOVi-A extractor and experiment runner are unchanged. The new adapter normalizes the shared MOVi-D/E TFDS schema and produces both pose-unaware camera-space geometry and oracle-pose-aligned world geometry from the same masked depth points.

## Files

- `src/movi_de_dataset_adapter.py`: dataset definitions, schema validation, dynamic shard discovery, TFRecord iteration, depth decoding, camera transforms, reprojection, camera-motion summaries, and D/E instance-metadata normalization.
- `src/extract_movi_de_phase1.py`: extraction CLI that writes crops, normalized observations, pose validation, exclusions, and manifests.
- `configs/movi_de_protocol.json`: approved Phase 0 settings and numerical validation tolerances.
- `tests/test_movi_de_adapter.py`: schema, shard, coordinate, leakage, and synthetic-record tests.

## Pilot video manifest

Create one JSONL manifest per dataset with exactly 20 pilot videos. Pilot videos must not later enter a confirmatory test pool.

The repository includes a resumable seeded downloader that selects two validation shards per dataset, verifies the public GCS MD5 values, inventories their records, and chooses 20 diverse pilot videos using seed `20260825`:

```bash
python src/download_movi_de_pilot.py \
  --output-root runs/movi_de_pilot/data \
  --manifests-dir manifests/movi_de \
  --seed 20260825 \
  --videos-per-dataset 20 \
  --shards-per-dataset 2
```

The current locked pilot download uses MOVi-D shard indices 3 and 8 and MOVi-E indices 8 and 13. Its source URLs, GCS MD5 values, local SHA-256 values, candidate inventories, and final manifest hashes are recorded in `runs/movi_de_pilot/data/pilot_download_manifest.json`.

```json
{"dataset":"movi_d","video_id":"17","split":"pilot"}
{"dataset":"movi_d","video_id":"42","split":"pilot"}
```

The manifest accepts `pilot`, `train`, `dev`, or `test` as the split. The extraction adapter does not choose videos or silently assign splits.

## Run the adapter

From the repository root, after installing `requirements-lock.txt`:

```bash
python src/extract_movi_de_phase1.py \
  --dataset movi_d \
  --tfrecord-dir /path/to/movi_d/128x128/validation \
  --video-manifest manifests/movi_de/pilot_movi_d_20.jsonl \
  --output-dir runs/movi_de_pilot/movi_d_phase1 \
  --allow-partial-shards

python src/extract_movi_de_phase1.py \
  --dataset movi_e \
  --tfrecord-dir /path/to/movi_e/128x128/validation \
  --video-manifest manifests/movi_de/pilot_movi_e_20.jsonl \
  --output-dir runs/movi_de_pilot/movi_e_phase1 \
  --allow-partial-shards
```

The adapter discovers and validates the declared shard count from filenames such as `movi_e-validation.tfrecord-00000-of-000NN`; it does not assume MOVi-A's layout. Confirmatory runs require the complete split. `--allow-partial-shards` is an explicit pilot-only exception and should be used only with the seeded download manifest.

For a one-video smoke test, add `--only-video-id VIDEO_ID`. The ID must already be present in the supplied manifest.

## Outputs

| Output | Information boundary |
|---|---|
| `model_inputs.jsonl` | Opaque crop paths, 2D controls, masked depth, intrinsics, camera pose, camera-space geometry, and pose-aligned visible-surface geometry |
| `observation_index.jsonl` | Dataset/video/split/frame/instance identity for deterministic joins and pair construction only |
| `instance_metadata.jsonl` | GSO asset ID, category, scale, dynamic/static flag, and background for sampling and evaluation strata only |
| `diagnostics.jsonl` | Simulator object position/velocity and reconstruction errors; forbidden as model features |
| `frame_camera_poses.jsonl` | Identity-bearing pose audit table for selection and validation |
| `pose_validation.json` | Per-video camera-motion and coordinate-validation diagnostics |
| `exclusions.jsonl` | Every excluded observation and its predeclared reason code |
| `video_summary.csv` | Pilot feasibility summary by video |
| `phase1_adapter_manifest.json` | Parameters, source files, leakage boundaries, counts, gates, and output hashes |

## Geometry definitions

MOVi depth is treated as radial distance from the camera center. Valid mask pixels are back-projected into CV camera axes: x right, y down, z forward. Kubric camera coordinates use x right, y up, z backward, so the adapter applies the explicit axis conversion `[1, -1, -1]` before the documented wxyz camera-to-world rotation and translation.

The camera-space and pose-aligned variants have matched definitions:

- coordinate-wise median of visible masked 3D surface points;
- coordinate-wise q95 minus q05 extent of the same points.

The world-aligned result is named **oracle-pose-aligned visible-surface geometry**. It is view-dependent and must not be described as ground-truth object position.

## Hard validation gates

Extraction stops if any selected video exceeds one of the locked tolerances:

1. Camera → world → camera round-trip error: `1e-8` scene units.
2. Back-projection/reprojection error: `1e-6` pixels.
3. Quaternion-derived rotation orthonormality error: `1e-10`.

The adapter also reports whether static-object visible-surface centroids are more stable in world coordinates than camera coordinates. That diagnostic is not a hard gate because mask truncation and occlusion can shift a visible-surface centroid even when the object is static.

## Leakage rule

Camera pose is intentionally permitted for the pose-aware systems. Simulator object position, velocity, identity, asset ID, category, dynamic/static flag, video ID, background, and split are rejected from `model_inputs.jsonl` or isolated in non-model files. Later feature builders must use explicit per-system allowlists so systems A-C cannot consume pose-aligned geometry or camera pose.

## Test

```bash
python -m unittest tests.test_movi_de_adapter -v
```

The synthetic-record test exercises image/depth decoding, both coordinate representations, pose validation, crop writing, normalized D/E metadata, and the inference leakage guard without requiring a dataset download.
