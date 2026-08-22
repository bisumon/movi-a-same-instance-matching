#!/usr/bin/env python3
"""Extract Phase 1 object observations from the fixed MOVi-A video split.

The outputs deliberately separate:
  * model_inputs.jsonl: inference-available RGB/depth/2D/3D features;
  * observation_index.jsonl: identity and split metadata for pair generation;
  * diagnostics.jsonl: ground-truth positions/velocities and reconstruction error.

This script reads the public TFDS TFRecord shards without TensorFlow by using
the lightweight ``tfrecord`` package.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


VERSION = "1.0.0"
FORBIDDEN_MODEL_KEY_FRAGMENTS = (
    "gt_",
    "ground_truth",
    "world_position",
    "velocity",
    "quaternion",
    "instance_index",
    "shape_label",
    "color_label",
    "material_label",
    "size_label",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfrecord-dir", type=Path, required=True)
    parser.add_argument("--video-splits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--crop-padding", type=float, default=0.15)
    parser.add_argument("--min-visibility-pixels", type=int, default=32)
    parser.add_argument("--min-mask-area", type=int, default=32)
    parser.add_argument(
        "--only-video-id",
        action="append",
        default=[],
        help="Optional repeatable video ID filter for smoke tests.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def load_splits(path: Path) -> dict[str, str]:
    splits: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            video_id = str(row["video_id"])
            split = str(row["split"])
            if split not in {"train", "dev", "test"}:
                raise ValueError(f"Unsupported split {split!r} for video {video_id}")
            if video_id in splits:
                raise ValueError(f"Duplicate video ID {video_id} in {path}")
            splits[video_id] = split
    return splits


def scalar(record: dict[str, Any], key: str, cast: type = float) -> Any:
    value = np.asarray(record[key]).reshape(-1)
    if value.size != 1:
        raise ValueError(f"Expected scalar feature {key}, got shape {value.shape}")
    return cast(value[0])


def decode_png(blob: bytes | np.bytes_) -> np.ndarray:
    with Image.open(io.BytesIO(bytes(blob))) as image:
        return np.asarray(image).copy()


def decode_depth(blob: bytes | np.bytes_, depth_range: np.ndarray) -> np.ndarray:
    """Decode TFDS uint16 depth to camera-center distance in scene units."""
    encoded = decode_png(blob).astype(np.float64)
    if encoded.ndim == 3:
        encoded = encoded[..., 0]
    minimum, maximum = (float(depth_range[0]), float(depth_range[1]))
    return encoded / 65535.0 * (maximum - minimum) + minimum


def padded_bbox(mask: np.ndarray, padding: float) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Return tight and padded bboxes as half-open (x0, y0, x1, y1)."""
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        raise ValueError("Cannot compute a bounding box for an empty mask")
    x0, x1 = int(cols.min()), int(cols.max()) + 1
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    width, height = x1 - x0, y1 - y0
    pad_x = int(math.ceil(width * padding))
    pad_y = int(math.ceil(height * padding))
    frame_height, frame_width = mask.shape
    padded = (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(frame_width, x1 + pad_x),
        min(frame_height, y1 + pad_y),
    )
    return (x0, y0, x1, y1), padded


def quaternion_to_rotation_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Convert a Kubric wxyz quaternion into a camera-to-world rotation."""
    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Invalid zero or non-finite camera quaternion")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def masked_camera_points(
    mask: np.ndarray,
    depth: np.ndarray,
    focal_x_px: float,
    focal_y_px: float,
) -> np.ndarray:
    """Back-project a mask using CV axes: x right, y down, z forward.

    MOVi depth is radial distance from the camera center, so each pixel ray is
    normalized before multiplication by depth.
    """
    rows, cols = np.nonzero(mask)
    radial_depth = depth[rows, cols]
    valid = np.isfinite(radial_depth) & (radial_depth > 0)
    rows, cols, radial_depth = rows[valid], cols[valid], radial_depth[valid]
    if radial_depth.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    height, width = mask.shape
    x = (cols.astype(np.float64) + 0.5 - width / 2.0) / focal_x_px
    y = (rows.astype(np.float64) + 0.5 - height / 2.0) / focal_y_px
    rays = np.column_stack((x, y, np.ones_like(x)))
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    return rays * radial_depth[:, None]


def quantile_list(values: np.ndarray, probabilities: tuple[float, ...]) -> list[float]:
    return [float(value) for value in np.quantile(values, probabilities)]


def depth_statistics(values: np.ndarray) -> dict[str, float | int]:
    return {
        "valid_pixel_count": int(values.size),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def stable_observation_id(video_id: str, frame_index: int, instance_index: int) -> str:
    source = f"movi-a|{video_id}|{frame_index}|{instance_index}".encode()
    return hashlib.sha256(source).hexdigest()[:20]


def byte_label(values: np.ndarray, index: int) -> str:
    value = values.reshape(-1)[index]
    return bytes(value).decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)


def json_vector(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values).reshape(-1)]


def validate_model_record(record: dict[str, Any]) -> None:
    def visit(value: Any, path: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                lower = key.lower()
                if any(fragment in lower for fragment in FORBIDDEN_MODEL_KEY_FRAGMENTS):
                    raise ValueError(f"Forbidden diagnostic/identity field in model input: {path}{key}")
                visit(child, f"{path}{key}.")
    visit(record)


def save_resized_crops(
    rgb: np.ndarray,
    mask: np.ndarray,
    crop_box: tuple[int, int, int, int],
    crop_size: int,
    rgb_path: Path,
    mask_path: Path,
) -> None:
    x0, y0, x1, y1 = crop_box
    rgb_crop = Image.fromarray(rgb[y0:y1, x0:x1], mode="RGB").resize(
        (crop_size, crop_size), Image.Resampling.LANCZOS
    )
    mask_crop = Image.fromarray((mask[y0:y1, x0:x1].astype(np.uint8) * 255), mode="L").resize(
        (crop_size, crop_size), Image.Resampling.NEAREST
    )
    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_crop.save(rgb_path, format="PNG", optimize=False)
    mask_crop.save(mask_path, format="PNG", optimize=False)


def process_video(
    record: dict[str, Any],
    video_id: str,
    split: str,
    shard_name: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    num_frames = scalar(record, "metadata/num_frames", int)
    num_instances = scalar(record, "metadata/num_instances", int)
    height = scalar(record, "metadata/height", int)
    width = scalar(record, "metadata/width", int)
    if len(record["video"]) != num_frames or len(record["depth"]) != num_frames:
        raise ValueError(f"Frame count mismatch in video {video_id}")

    visibility = np.asarray(record["instances/visibility"]).reshape(num_instances, num_frames)
    positions = np.asarray(record["instances/positions"], dtype=np.float64).reshape(num_instances, num_frames, 3)
    velocities = np.asarray(record["instances/velocities"], dtype=np.float64).reshape(num_instances, num_frames, 3)
    bboxes_3d = np.asarray(record["instances/bboxes_3d"], dtype=np.float64).reshape(
        num_instances, num_frames, 8, 3
    )
    camera_positions = np.asarray(record["camera/positions"], dtype=np.float64).reshape(num_frames, 3)
    camera_quaternions = np.asarray(record["camera/quaternions"], dtype=np.float64).reshape(num_frames, 4)
    depth_range = np.asarray(record["metadata/depth_range"], dtype=np.float64).reshape(2)
    focal_length_mm = scalar(record, "camera/focal_length")
    sensor_width_mm = scalar(record, "camera/sensor_width")
    focal_x_px = focal_length_mm / sensor_width_mm * width
    focal_y_px = focal_x_px

    instance_rows = []
    for instance_index in range(num_instances):
        instance_rows.append(
            {
                "video_id": video_id,
                "split": split,
                "instance_index": instance_index,
                "segmentation_id": instance_index + 1,
                "shape_label": byte_label(record["instances/shape_label"], instance_index),
                "color_label": byte_label(record["instances/color_label"], instance_index),
                "material_label": byte_label(record["instances/material_label"], instance_index),
                "size_label": byte_label(record["instances/size_label"], instance_index),
            }
        )

    model_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []

    for frame_index in range(num_frames):
        rgb = decode_png(record["video"][frame_index])
        segmentation = decode_png(record["segmentations"][frame_index])
        depth = decode_depth(record["depth"][frame_index], depth_range)
        if segmentation.ndim == 3:
            segmentation = segmentation[..., 0]
        if rgb.shape != (height, width, 3) or segmentation.shape != (height, width):
            raise ValueError(f"Unexpected image shape in video {video_id}, frame {frame_index}")

        camera_rotation = quaternion_to_rotation_matrix(camera_quaternions[frame_index])
        for instance_index in range(num_instances):
            observation_id = stable_observation_id(video_id, frame_index, instance_index)
            mask = segmentation == instance_index + 1
            mask_area = int(mask.sum())
            annotated_visibility = int(visibility[instance_index, frame_index])
            reasons = []
            if annotated_visibility < args.min_visibility_pixels:
                reasons.append("visibility_below_threshold")
            if mask_area < args.min_mask_area:
                reasons.append("mask_area_below_threshold")
            if reasons:
                excluded_rows.append(
                    {
                        "observation_id": observation_id,
                        "video_id": video_id,
                        "split": split,
                        "frame_index": frame_index,
                        "instance_index": instance_index,
                        "visibility": annotated_visibility,
                        "mask_area": mask_area,
                        "reasons": reasons,
                    }
                )
                continue

            tight_box, crop_box = padded_bbox(mask, args.crop_padding)
            x0, y0, x1, y1 = tight_box
            px0, py0, px1, py1 = crop_box
            points_cv = masked_camera_points(mask, depth, focal_x_px, focal_y_px)
            if points_cv.shape[0] < args.min_mask_area:
                excluded_rows.append(
                    {
                        "observation_id": observation_id,
                        "video_id": video_id,
                        "split": split,
                        "frame_index": frame_index,
                        "instance_index": instance_index,
                        "visibility": annotated_visibility,
                        "mask_area": mask_area,
                        "reasons": ["insufficient_valid_depth_pixels"],
                    }
                )
                continue

            masked_depth = np.linalg.norm(points_cv, axis=1)
            center_cv = np.median(points_cv, axis=0)
            lower = np.quantile(points_cv, 0.05, axis=0)
            upper = np.quantile(points_cv, 0.95, axis=0)
            extent_cv = upper - lower
            mask_rows, mask_cols = np.nonzero(mask)
            mask_center_x = float(np.mean(mask_cols) + 0.5)
            mask_center_y = float(np.mean(mask_rows) + 0.5)

            # Keep model-facing paths opaque: split/video/instance identity lives
            # only in observation_index.jsonl, never in model_inputs.jsonl values.
            rgb_relative = Path("crops") / f"{observation_id}.png"
            mask_relative = Path("masks") / f"{observation_id}.png"
            save_resized_crops(
                rgb, mask, crop_box, args.crop_size, output_dir / rgb_relative, output_dir / mask_relative
            )

            model_row = {
                "observation_id": observation_id,
                "rgb_crop_path": rgb_relative.as_posix(),
                "mask_crop_path": mask_relative.as_posix(),
                "frame_index": frame_index,
                "visibility": annotated_visibility,
                "mask_area": mask_area,
                "tight_bbox_xyxy": [x0, y0, x1, y1],
                "padded_bbox_xyxy": [px0, py0, px1, py1],
                "crop_width": x1 - x0,
                "crop_height": y1 - y0,
                "padded_crop_width": px1 - px0,
                "padded_crop_height": py1 - py0,
                "mask_center_x_normalized": mask_center_x / width,
                "mask_center_y_normalized": mask_center_y / height,
                "bbox_aspect_ratio": (x1 - x0) / (y1 - y0),
                "mask_fill_fraction": mask_area / ((x1 - x0) * (y1 - y0)),
                "depth": depth_statistics(masked_depth),
                "camera_space_visible_surface_centroid_xyz": json_vector(center_cv),
                "camera_space_visible_surface_extent_q05_q95_xyz": json_vector(extent_cv),
                "intrinsics": {
                    "focal_x_pixels": focal_x_px,
                    "focal_y_pixels": focal_y_px,
                    "principal_x_pixels": width / 2.0,
                    "principal_y_pixels": height / 2.0,
                    "image_width": width,
                    "image_height": height,
                },
            }
            validate_model_record(model_row)
            model_rows.append(model_row)
            index_rows.append(
                {
                    "observation_id": observation_id,
                    "video_id": video_id,
                    "split": split,
                    "frame_index": frame_index,
                    "instance_index": instance_index,
                    "segmentation_id": instance_index + 1,
                    "source_shard": shard_name,
                }
            )

            # Kubric camera axes are x right, y up, z backward. The inference
            # record uses CV axes x right, y down, z forward.
            center_kubric_camera = center_cv * np.asarray([1.0, -1.0, -1.0])
            reconstructed_world = (
                camera_positions[frame_index] + camera_rotation @ center_kubric_camera
            )
            gt_world_position = positions[instance_index, frame_index]
            gt_camera_kubric = camera_rotation.T @ (
                gt_world_position - camera_positions[frame_index]
            )
            gt_camera_cv = gt_camera_kubric * np.asarray([1.0, -1.0, -1.0])
            gt_bbox_kubric = (
                camera_rotation.T
                @ (bboxes_3d[instance_index, frame_index] - camera_positions[frame_index]).T
            ).T
            gt_bbox_cv = gt_bbox_kubric * np.asarray([1.0, -1.0, -1.0])
            gt_extent_cv = np.ptp(gt_bbox_cv, axis=0)
            diagnostic_rows.append(
                {
                    "observation_id": observation_id,
                    "gt_world_position_xyz": json_vector(gt_world_position),
                    "gt_world_velocity_xyz": json_vector(velocities[instance_index, frame_index]),
                    "gt_camera_center_xyz": json_vector(gt_camera_cv),
                    "gt_camera_bbox_extent_xyz": json_vector(gt_extent_cv),
                    "reconstructed_world_visible_surface_centroid_xyz": json_vector(reconstructed_world),
                    "visible_surface_center_error_l2": float(np.linalg.norm(center_cv - gt_camera_cv)),
                    "radial_depth_error": float(
                        abs(np.linalg.norm(center_cv) - np.linalg.norm(gt_camera_cv))
                    ),
                    "visible_extent_error_l2": float(np.linalg.norm(extent_cv - gt_extent_cv)),
                }
            )

    return model_rows, index_rows, diagnostic_rows, excluded_rows, instance_rows


def main() -> int:
    args = parse_args()
    if args.crop_size <= 0:
        raise ValueError("--crop-size must be positive")
    if not 0 <= args.crop_padding <= 1:
        raise ValueError("--crop-padding must be between 0 and 1")
    if args.min_visibility_pixels < 1 or args.min_mask_area < 1:
        raise ValueError("Visibility and mask-area thresholds must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from tfrecord.reader import tfrecord_loader
    except ImportError as exc:
        raise RuntimeError(
            "The lightweight 'tfrecord' package is required. Install it or add its "
            "target directory to PYTHONPATH."
        ) from exc

    splits = load_splits(args.video_splits)
    if args.only_video_id:
        requested = set(args.only_video_id)
        missing = requested - set(splits)
        if missing:
            raise ValueError(f"Unknown --only-video-id values: {sorted(missing)}")
        splits = {video_id: split for video_id, split in splits.items() if video_id in requested}
    expected_counts = Counter(splits.values())
    if not args.only_video_id and expected_counts != Counter({"train": 30, "dev": 10, "test": 10}):
        raise ValueError(f"Expected fixed 30/10/10 split, got {dict(expected_counts)}")

    shards = sorted(args.tfrecord_dir.glob("movi_a-validation.tfrecord-*-of-*"))
    if len(shards) != 16:
        raise FileNotFoundError(f"Expected 16 validation shards in {args.tfrecord_dir}, found {len(shards)}")

    model_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    instance_rows: list[dict[str, Any]] = []
    found: set[str] = set()

    for shard_index, shard in enumerate(shards, start=1):
        for record in tfrecord_loader(str(shard), None):
            raw_id = record["metadata/video_name"]
            video_id = raw_id.decode("utf-8") if isinstance(raw_id, bytes) else str(raw_id)
            if video_id not in splits:
                continue
            if video_id in found:
                raise ValueError(f"Duplicate selected video {video_id} in TFRecords")
            outputs = process_video(record, video_id, splits[video_id], shard.name, args.output_dir, args)
            model, index, diagnostic, excluded, instances = outputs
            model_rows.extend(model)
            index_rows.extend(index)
            diagnostic_rows.extend(diagnostic)
            excluded_rows.extend(excluded)
            instance_rows.extend(instances)
            found.add(video_id)
            print(
                f"[{len(found):02d}/{len(splits):02d}] video={video_id} split={splits[video_id]} "
                f"included={len(model)} excluded={len(excluded)}",
                flush=True,
            )
        print(f"scanned shard {shard_index:02d}/{len(shards):02d}: {shard.name}", flush=True)

    missing_videos = set(splits) - found
    if missing_videos:
        raise RuntimeError(f"Selected videos not found in TFRecords: {sorted(missing_videos)}")

    order = lambda row: row["observation_id"]
    model_rows.sort(key=order)
    index_rows.sort(key=order)
    diagnostic_rows.sort(key=order)
    excluded_rows.sort(key=order)
    instance_rows.sort(key=lambda row: (row["video_id"], row["instance_index"]))
    write_jsonl(args.output_dir / "model_inputs.jsonl", model_rows)
    write_jsonl(args.output_dir / "observation_index.jsonl", index_rows)
    write_jsonl(args.output_dir / "diagnostics.jsonl", diagnostic_rows)
    write_jsonl(args.output_dir / "exclusions.jsonl", excluded_rows)
    write_jsonl(args.output_dir / "instance_metadata.jsonl", instance_rows)

    index_by_id = {row["observation_id"]: row for row in index_rows}
    included_by_split = Counter(index_by_id[row["observation_id"]]["split"] for row in model_rows)
    excluded_by_reason = Counter(reason for row in excluded_rows for reason in row["reasons"])
    visibility_ratios = [
        row["visibility"] / row["mask_area"] for row in model_rows if row["mask_area"] > 0
    ]
    diagnostic_metrics = {
        key: [float(row[key]) for row in diagnostic_rows]
        for key in (
            "visible_surface_center_error_l2",
            "radial_depth_error",
            "visible_extent_error_l2",
        )
    }

    with (args.output_dir / "video_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["video_id", "split", "included_observations", "excluded_observations"],
        )
        writer.writeheader()
        for video_id in sorted(splits):
            writer.writerow(
                {
                    "video_id": video_id,
                    "split": splits[video_id],
                    "included_observations": sum(
                        row["video_id"] == video_id for row in index_rows
                    ),
                    "excluded_observations": sum(
                        row["video_id"] == video_id for row in excluded_rows
                    ),
                }
            )

    output_hashes = {}
    for filename in (
        "model_inputs.jsonl",
        "observation_index.jsonl",
        "diagnostics.jsonl",
        "exclusions.jsonl",
        "instance_metadata.jsonl",
        "video_summary.csv",
    ):
        output_hashes[filename] = sha256(args.output_dir / filename)
    manifest = {
        "pipeline": "MOVi-A Phase 1 object crop and geometry extraction",
        "version": VERSION,
        "source": {
            "tfrecord_directory": str(args.tfrecord_dir.resolve()),
            "validation_shards": [
                {"filename": shard.name, "size_bytes": shard.stat().st_size} for shard in shards
            ],
            "video_splits": str(args.video_splits.resolve()),
            "video_splits_sha256": sha256(args.video_splits),
        },
        "parameters": {
            "crop_size": [args.crop_size, args.crop_size],
            "crop_padding_fraction_per_side": args.crop_padding,
            "min_visibility_pixels": args.min_visibility_pixels,
            "min_mask_area_at_128x128": args.min_mask_area,
            "depth_decode": "uint16 / 65535 * (depth_max - depth_min) + depth_min",
            "depth_semantics": "radial distance from camera center in scene units",
            "camera_space_convention": "x right, y down, z forward",
            "extent_estimator": "coordinate-wise q95 minus q05 over visible masked surface",
            "centroid_estimator": "coordinate-wise median over visible masked surface",
            "pixel_center_convention": "(column + 0.5, row + 0.5)",
        },
        "leakage_boundaries": {
            "model_inputs.jsonl": "Inference-available opaque RGB/mask paths, 2D controls, masked depth, intrinsics, and camera-space geometry only.",
            "observation_index.jsonl": "Video/split/instance identity for pair construction; never pass these identifiers to a scorer.",
            "instance_metadata.jsonl": "Shape/color/material/size labels for hard/easy-negative sampling only; never pass labels to a scorer.",
            "diagnostics.jsonl": "Ground-truth positions, velocities, and reconstruction errors; diagnostics only and forbidden as model features.",
        },
        "counts": {
            "videos": len(found),
            "videos_by_split": dict(sorted(Counter(splits.values()).items())),
            "instances": len(instance_rows),
            "included_observations": len(model_rows),
            "included_observations_by_split": dict(sorted(included_by_split.items())),
            "excluded_observations": len(excluded_rows),
            "exclusions_by_reason": dict(sorted(excluded_by_reason.items())),
        },
        "quality_checks": {
            "model_index_ids_match": {row["observation_id"] for row in model_rows}
            == {row["observation_id"] for row in index_rows},
            "model_diagnostic_ids_match": {row["observation_id"] for row in model_rows}
            == {row["observation_id"] for row in diagnostic_rows},
            "visibility_to_decoded_mask_area_ratio": {
                "median": float(np.median(visibility_ratios)),
                "p05": float(np.quantile(visibility_ratios, 0.05)),
                "p95": float(np.quantile(visibility_ratios, 0.95)),
                "note": "MOVi-A 128x128 visibility annotations are commonly about 4x decoded mask area; both thresholds are enforced independently.",
            },
            "diagnostic_error_scene_units": {
                key: {
                    "median": float(np.median(values)),
                    "p95": float(np.quantile(values, 0.95)),
                    "maximum": float(np.max(values)),
                }
                for key, values in diagnostic_metrics.items()
            },
            "forbidden_model_key_fragments_checked": list(FORBIDDEN_MODEL_KEY_FRAGMENTS),
        },
        "output_sha256": output_hashes,
    }
    write_json(args.output_dir / "phase1_manifest.json", manifest)
    print(
        f"Complete: {len(found)} videos, {len(model_rows)} included observations, "
        f"{len(excluded_rows)} excluded observations",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
