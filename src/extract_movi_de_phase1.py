#!/usr/bin/env python3
"""Extract normalized Phase 1 observations from MOVi-D or MOVi-E TFRecords.

This is the pilot/confirmatory data adapter for the camera-pose experiment.  It
emits camera-space and oracle-pose-aligned visible-surface geometry with matched
definitions, while isolating simulator identity and object state in separate
sampling/diagnostic files.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from movi_de_dataset_adapter import (
    camera_cv_to_world,
    camera_motion_summary,
    decode_depth,
    decode_png,
    discover_shards,
    get_dataset_spec,
    iter_tfrecords,
    masked_camera_points,
    normalize_instance_metadata,
    padded_bbox,
    project_camera_points,
    record_video_id,
    relative_rotation_degrees,
    rotation_orthonormality_error,
    scalar,
    sha256,
    stable_observation_id,
    validate_record_schema,
    world_to_camera_cv,
)


VERSION = "0.1.0"
ALLOWED_SPLITS = frozenset({"pilot", "train", "dev", "test"})
FORBIDDEN_MODEL_KEY_FRAGMENTS = (
    "gt_",
    "ground_truth",
    "object_position",
    "object_velocity",
    "instance_index",
    "segmentation_id",
    "asset_id",
    "category",
    "is_dynamic",
    "video_id",
    "dataset",
    "source_shard",
    "background",
)


@dataclass(slots=True)
class VideoOutputs:
    model_rows: list[dict[str, Any]]
    index_rows: list[dict[str, Any]]
    diagnostic_rows: list[dict[str, Any]]
    excluded_rows: list[dict[str, Any]]
    instance_rows: list[dict[str, Any]]
    frame_rows: list[dict[str, Any]]
    quality: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("movi_d", "movi_e"))
    parser.add_argument("--tfrecord-dir", type=Path, required=True)
    parser.add_argument("--video-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-split", default="validation")
    parser.add_argument(
        "--allow-partial-shards",
        action="store_true",
        help="Pilot only: accept a documented subset of shards instead of the complete split.",
    )
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--crop-padding", type=float, default=0.15)
    parser.add_argument("--min-visibility-pixels", type=int, default=32)
    parser.add_argument("--min-mask-area", type=int, default=32)
    parser.add_argument("--max-roundtrip-error", type=float, default=1e-8)
    parser.add_argument("--max-reprojection-error-pixels", type=float, default=1e-6)
    parser.add_argument("--max-rotation-orthonormality-error", type=float, default=1e-10)
    parser.add_argument(
        "--only-video-id",
        action="append",
        default=[],
        help="Optional repeatable video ID filter for smoke tests.",
    )
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def json_vector(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values).reshape(-1)]


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


def robust_centroid_and_extent(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(points, axis=0)
    extent = np.quantile(points, 0.95, axis=0) - np.quantile(points, 0.05, axis=0)
    return center, extent


def validate_model_record(record: dict[str, Any]) -> None:
    def visit(value: Any, path: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                lower = key.lower()
                if any(fragment in lower for fragment in FORBIDDEN_MODEL_KEY_FRAGMENTS):
                    raise ValueError(f"Forbidden identity/diagnostic field in model input: {path}{key}")
                visit(child, f"{path}{key}.")
    visit(record)


def load_video_manifest(path: Path, dataset: str) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            row_dataset = str(row.get("dataset", dataset)).lower().replace("-", "_")
            if row_dataset != dataset:
                raise ValueError(
                    f"Manifest line {line_number} declares {row_dataset}, expected {dataset}"
                )
            video_id = str(row["video_id"])
            split = str(row.get("split", "pilot"))
            if split not in ALLOWED_SPLITS:
                raise ValueError(f"Unsupported split {split!r} for video {video_id}")
            if video_id in result:
                raise ValueError(f"Duplicate video ID {video_id} in {path}")
            result[video_id] = split
    if not result:
        raise ValueError(f"Video manifest is empty: {path}")
    return result


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
    mask_crop = Image.fromarray(mask[y0:y1, x0:x1].astype(np.uint8) * 255, mode="L").resize(
        (crop_size, crop_size), Image.Resampling.NEAREST
    )
    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_crop.save(rgb_path, format="PNG", optimize=False)
    mask_crop.save(mask_path, format="PNG", optimize=False)


def static_stability_summary(
    tracks: dict[int, dict[str, list[np.ndarray]]],
) -> dict[str, Any]:
    camera_deviations: list[float] = []
    world_deviations: list[float] = []
    eligible_tracks = 0
    for values in tracks.values():
        camera = np.asarray(values["camera"], dtype=np.float64)
        world = np.asarray(values["world"], dtype=np.float64)
        if len(camera) < 2:
            continue
        eligible_tracks += 1
        camera_center = np.median(camera, axis=0)
        world_center = np.median(world, axis=0)
        camera_deviations.extend(np.linalg.norm(camera - camera_center, axis=1).tolist())
        world_deviations.extend(np.linalg.norm(world - world_center, axis=1).tolist())
    if not camera_deviations:
        return {
            "eligible_static_tracks": eligible_tracks,
            "available": False,
            "reason": "No static instance had at least two included observations.",
        }
    camera_median = float(np.median(camera_deviations))
    world_median = float(np.median(world_deviations))
    return {
        "eligible_static_tracks": eligible_tracks,
        "available": True,
        "camera_space_median_deviation": camera_median,
        "world_space_median_deviation": world_median,
        "world_to_camera_deviation_ratio": world_median / max(camera_median, 1e-12),
        "world_space_more_stable": world_median < camera_median,
        "note": "Visible-surface centroids remain view-dependent under occlusion; this is a pilot diagnostic, not an object-position feature.",
    }


def process_video(
    record: dict[str, Any],
    dataset: str,
    video_id: str,
    split: str,
    shard_name: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> VideoOutputs:
    spec = get_dataset_spec(dataset)
    validate_record_schema(record, spec)
    num_frames = scalar(record, "metadata/num_frames", int)
    num_instances = scalar(record, "metadata/num_instances", int)
    height = scalar(record, "metadata/height", int)
    width = scalar(record, "metadata/width", int)
    visibility = np.asarray(record["instances/visibility"]).reshape(num_instances, num_frames)
    positions = np.asarray(record["instances/positions"], dtype=np.float64).reshape(
        num_instances, num_frames, 3
    )
    velocities = np.asarray(record["instances/velocities"], dtype=np.float64).reshape(
        num_instances, num_frames, 3
    )
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
    instance_rows = normalize_instance_metadata(record, dataset, video_id, split)
    is_dynamic = {row["instance_index"]: bool(row["is_dynamic"]) for row in instance_rows}

    model_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    static_tracks: dict[int, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: {"camera": [], "world": []}
    )
    roundtrip_errors: list[float] = []
    reprojection_errors: list[float] = []
    rotation_errors: list[float] = []

    for frame_index in range(num_frames):
        position = camera_positions[frame_index]
        quaternion = camera_quaternions[frame_index]
        rotation_errors.append(rotation_orthonormality_error(quaternion))
        frame_rows.append(
            {
                "dataset": spec.name,
                "video_id": video_id,
                "split": split,
                "frame_index": frame_index,
                "camera_position_world_xyz": json_vector(position),
                "camera_to_world_quaternion_wxyz": json_vector(quaternion),
                "translation_from_first_scene_units": float(np.linalg.norm(position - camera_positions[0])),
                "rotation_from_first_degrees": relative_rotation_degrees(camera_quaternions[0], quaternion),
            }
        )
        rgb = decode_png(record["video"][frame_index])
        segmentation = decode_png(record["segmentations"][frame_index])
        depth = decode_depth(record["depth"][frame_index], depth_range)
        if segmentation.ndim == 3:
            segmentation = segmentation[..., 0]
        if rgb.shape != (height, width, 3) or segmentation.shape != (height, width):
            raise ValueError(f"Unexpected image shape in {spec.display_name} video {video_id}, frame {frame_index}")

        for instance_index in range(num_instances):
            observation_id = stable_observation_id(dataset, video_id, frame_index, instance_index)
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
                        "dataset": spec.name,
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
            points_cv = masked_camera_points(mask, depth, focal_x_px, focal_y_px)
            if points_cv.shape[0] < args.min_mask_area:
                excluded_rows.append(
                    {
                        "observation_id": observation_id,
                        "dataset": spec.name,
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
            points_world = camera_cv_to_world(points_cv, position, quaternion)
            roundtrip = world_to_camera_cv(points_world, position, quaternion)
            roundtrip_error = float(np.max(np.linalg.norm(roundtrip - points_cv, axis=1)))
            roundtrip_errors.append(roundtrip_error)

            mask_rows, mask_cols = np.nonzero(mask)
            radial_depth = depth[mask_rows, mask_cols]
            valid = np.isfinite(radial_depth) & (radial_depth > 0)
            source_pixels = np.column_stack((mask_cols[valid], mask_rows[valid])).astype(np.float64)
            projected_pixels = project_camera_points(points_cv, focal_x_px, focal_y_px, width, height)
            reprojection_error = float(np.max(np.linalg.norm(projected_pixels - source_pixels, axis=1)))
            reprojection_errors.append(reprojection_error)

            center_cv, extent_cv = robust_centroid_and_extent(points_cv)
            center_world, extent_world = robust_centroid_and_extent(points_world)
            if not is_dynamic[instance_index]:
                static_tracks[instance_index]["camera"].append(center_cv)
                static_tracks[instance_index]["world"].append(center_world)
            x0, y0, x1, y1 = tight_box
            px0, py0, px1, py1 = crop_box
            mask_center_y, mask_center_x = np.mean(np.argwhere(mask), axis=0) + 0.5
            rgb_relative = Path("crops") / f"{observation_id}.png"
            mask_relative = Path("masks") / f"{observation_id}.png"
            save_resized_crops(
                rgb,
                mask,
                crop_box,
                args.crop_size,
                output_dir / rgb_relative,
                output_dir / mask_relative,
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
                "mask_center_x_normalized": float(mask_center_x / width),
                "mask_center_y_normalized": float(mask_center_y / height),
                "bbox_aspect_ratio": (x1 - x0) / (y1 - y0),
                "mask_fill_fraction": mask_area / ((x1 - x0) * (y1 - y0)),
                "depth": depth_statistics(np.linalg.norm(points_cv, axis=1)),
                "camera_space_visible_surface_centroid_xyz": json_vector(center_cv),
                "camera_space_visible_surface_extent_q05_q95_xyz": json_vector(extent_cv),
                "pose_aligned_world_visible_surface_centroid_xyz": json_vector(center_world),
                "pose_aligned_world_visible_surface_extent_q05_q95_xyz": json_vector(extent_world),
                "camera_pose": {
                    "position_world_xyz": json_vector(position),
                    "camera_to_world_quaternion_wxyz": json_vector(quaternion),
                },
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
                    "dataset": spec.name,
                    "video_id": video_id,
                    "split": split,
                    "frame_index": frame_index,
                    "instance_index": instance_index,
                    "segmentation_id": instance_index + 1,
                    "source_shard": shard_name,
                }
            )

            gt_world_position = positions[instance_index, frame_index]
            gt_camera_cv = world_to_camera_cv(gt_world_position[None, :], position, quaternion)[0]
            gt_bbox_cv = world_to_camera_cv(bboxes_3d[instance_index, frame_index], position, quaternion)
            diagnostic_rows.append(
                {
                    "observation_id": observation_id,
                    "gt_world_position_xyz": json_vector(gt_world_position),
                    "gt_world_velocity_xyz": json_vector(velocities[instance_index, frame_index]),
                    "gt_camera_center_xyz": json_vector(gt_camera_cv),
                    "gt_camera_bbox_extent_xyz": json_vector(np.ptp(gt_bbox_cv, axis=0)),
                    "reconstructed_world_visible_surface_centroid_xyz": json_vector(center_world),
                    "visible_surface_center_error_l2": float(np.linalg.norm(center_cv - gt_camera_cv)),
                    "radial_depth_error": float(abs(np.linalg.norm(center_cv) - np.linalg.norm(gt_camera_cv))),
                    "visible_extent_error_l2": float(np.linalg.norm(extent_cv - np.ptp(gt_bbox_cv, axis=0))),
                    "pose_roundtrip_max_error_scene_units": roundtrip_error,
                    "reprojection_max_error_pixels": reprojection_error,
                }
            )

    quality = {
        "camera_motion": camera_motion_summary(camera_positions, camera_quaternions),
        "max_pose_roundtrip_error_scene_units": max(roundtrip_errors, default=0.0),
        "max_reprojection_error_pixels": max(reprojection_errors, default=0.0),
        "max_rotation_orthonormality_error": max(rotation_errors, default=0.0),
        "static_visible_surface_stability": static_stability_summary(static_tracks),
        "dynamic_instance_count": sum(is_dynamic.values()),
        "static_instance_count": sum(not value for value in is_dynamic.values()),
    }
    return VideoOutputs(
        model_rows=model_rows,
        index_rows=index_rows,
        diagnostic_rows=diagnostic_rows,
        excluded_rows=excluded_rows,
        instance_rows=instance_rows,
        frame_rows=frame_rows,
        quality=quality,
    )


def main() -> int:
    args = parse_args()
    spec = get_dataset_spec(args.dataset)
    if args.crop_size <= 0:
        raise ValueError("--crop-size must be positive")
    if not 0 <= args.crop_padding <= 1:
        raise ValueError("--crop-padding must be between 0 and 1")
    if args.min_visibility_pixels < 1 or args.min_mask_area < 1:
        raise ValueError("Visibility and mask-area thresholds must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = load_video_manifest(args.video_manifest, spec.name)
    if args.only_video_id:
        requested = set(args.only_video_id)
        missing = requested - set(selected)
        if missing:
            raise ValueError(f"Unknown --only-video-id values: {sorted(missing)}")
        selected = {video_id: split for video_id, split in selected.items() if video_id in requested}
    shards = discover_shards(
        args.tfrecord_dir,
        spec.name,
        args.source_split,
        require_complete=not args.allow_partial_shards,
    )

    model_rows: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    instance_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    video_quality: dict[str, dict[str, Any]] = {}
    found: set[str] = set()

    for shard, record in iter_tfrecords(shards):
        video_id = record_video_id(record)
        if video_id not in selected:
            continue
        if video_id in found:
            raise ValueError(f"Duplicate selected video {video_id} in TFRecords")
        outputs = process_video(
            record,
            spec.name,
            video_id,
            selected[video_id],
            shard.name,
            args.output_dir,
            args,
        )
        if outputs.quality["max_pose_roundtrip_error_scene_units"] > args.max_roundtrip_error:
            raise RuntimeError(f"Pose round-trip gate failed for video {video_id}: {outputs.quality}")
        if outputs.quality["max_reprojection_error_pixels"] > args.max_reprojection_error_pixels:
            raise RuntimeError(f"Reprojection gate failed for video {video_id}: {outputs.quality}")
        if outputs.quality["max_rotation_orthonormality_error"] > args.max_rotation_orthonormality_error:
            raise RuntimeError(f"Rotation orthonormality gate failed for video {video_id}: {outputs.quality}")
        model_rows.extend(outputs.model_rows)
        index_rows.extend(outputs.index_rows)
        diagnostic_rows.extend(outputs.diagnostic_rows)
        excluded_rows.extend(outputs.excluded_rows)
        instance_rows.extend(outputs.instance_rows)
        frame_rows.extend(outputs.frame_rows)
        video_quality[video_id] = outputs.quality
        found.add(video_id)
        print(
            f"[{len(found):02d}/{len(selected):02d}] dataset={spec.name} video={video_id} "
            f"split={selected[video_id]} included={len(outputs.model_rows)} "
            f"excluded={len(outputs.excluded_rows)}",
            flush=True,
        )

    missing_videos = set(selected) - found
    if missing_videos:
        raise RuntimeError(f"Selected videos not found in TFRecords: {sorted(missing_videos)}")
    by_observation = lambda row: row["observation_id"]
    model_rows.sort(key=by_observation)
    index_rows.sort(key=by_observation)
    diagnostic_rows.sort(key=by_observation)
    excluded_rows.sort(key=by_observation)
    instance_rows.sort(key=lambda row: (row["video_id"], row["instance_index"]))
    frame_rows.sort(key=lambda row: (row["video_id"], row["frame_index"]))

    write_jsonl(args.output_dir / "model_inputs.jsonl", model_rows)
    write_jsonl(args.output_dir / "observation_index.jsonl", index_rows)
    write_jsonl(args.output_dir / "diagnostics.jsonl", diagnostic_rows)
    write_jsonl(args.output_dir / "exclusions.jsonl", excluded_rows)
    write_jsonl(args.output_dir / "instance_metadata.jsonl", instance_rows)
    write_jsonl(args.output_dir / "frame_camera_poses.jsonl", frame_rows)
    write_json(args.output_dir / "pose_validation.json", video_quality)

    with (args.output_dir / "video_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "dataset",
            "video_id",
            "split",
            "included_observations",
            "excluded_observations",
            "static_instances",
            "dynamic_instances",
            "camera_translation_scene_units",
            "camera_rotation_degrees",
            "normalized_camera_translation",
            "world_visible_surface_more_stable",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for video_id in sorted(selected):
            quality = video_quality[video_id]
            motion = quality["camera_motion"]
            stability = quality["static_visible_surface_stability"]
            writer.writerow(
                {
                    "dataset": spec.name,
                    "video_id": video_id,
                    "split": selected[video_id],
                    "included_observations": sum(row["video_id"] == video_id for row in index_rows),
                    "excluded_observations": sum(row["video_id"] == video_id for row in excluded_rows),
                    "static_instances": quality["static_instance_count"],
                    "dynamic_instances": quality["dynamic_instance_count"],
                    "camera_translation_scene_units": motion["translation_start_to_end_scene_units"],
                    "camera_rotation_degrees": motion["rotation_start_to_end_degrees"],
                    "normalized_camera_translation": motion["normalized_start_to_end_translation"],
                    "world_visible_surface_more_stable": stability.get("world_space_more_stable"),
                }
            )

    output_files = (
        "model_inputs.jsonl",
        "observation_index.jsonl",
        "diagnostics.jsonl",
        "exclusions.jsonl",
        "instance_metadata.jsonl",
        "frame_camera_poses.jsonl",
        "pose_validation.json",
        "video_summary.csv",
    )
    model_ids = {row["observation_id"] for row in model_rows}
    manifest = {
        "pipeline": "MOVi-D/E Phase 1 dataset adapter",
        "version": VERSION,
        "dataset": {
            "name": spec.name,
            "display_name": spec.display_name,
            "camera_regime": spec.camera_regime,
            "expected_motion": spec.expected_motion,
            "source_split": args.source_split,
            "partial_shards_allowed": args.allow_partial_shards,
        },
        "source": {
            "tfrecord_directory": str(args.tfrecord_dir.resolve()),
            "shards": [{"filename": path.name, "size_bytes": path.stat().st_size} for path in shards],
            "video_manifest": str(args.video_manifest.resolve()),
            "video_manifest_sha256": sha256(args.video_manifest),
        },
        "parameters": {
            "crop_size": [args.crop_size, args.crop_size],
            "crop_padding_fraction_per_side": args.crop_padding,
            "min_visibility_pixels": args.min_visibility_pixels,
            "min_mask_area_pixels": args.min_mask_area,
            "depth_semantics": "radial distance from camera center in scene units",
            "camera_space_convention": "CV axes: x right, y down, z forward",
            "kubric_camera_convention": "x right, y up, z backward",
            "camera_quaternion_order": "wxyz",
            "camera_quaternion_transform": "camera-to-world",
            "pose_aligned_representation": "oracle-pose-aligned visible-surface geometry",
            "centroid_estimator": "coordinate-wise median over valid masked surface points",
            "extent_estimator": "coordinate-wise q95 minus q05 over valid masked surface points",
            "max_roundtrip_error": args.max_roundtrip_error,
            "max_reprojection_error_pixels": args.max_reprojection_error_pixels,
            "max_rotation_orthonormality_error": args.max_rotation_orthonormality_error,
        },
        "leakage_boundaries": {
            "model_inputs.jsonl": "Inference-available RGB/mask paths, 2D controls, masked depth, intrinsics, camera pose, camera-space geometry, and oracle-pose-aligned visible-surface geometry.",
            "observation_index.jsonl": "Dataset/video/split/instance identity for joins and pair construction; never pass identifiers to a scorer.",
            "instance_metadata.jsonl": "Asset/category/scale/dynamic/background fields for sampling and strata only; never pass to a scorer.",
            "diagnostics.jsonl": "Simulator object positions/velocities and reconstruction errors; evaluation only.",
            "frame_camera_poses.jsonl": "Identity-bearing pose audit table; model-facing pose values are copied into model_inputs without video identity.",
        },
        "counts": {
            "videos": len(found),
            "videos_by_split": dict(sorted(Counter(selected.values()).items())),
            "instances": len(instance_rows),
            "included_observations": len(model_rows),
            "excluded_observations": len(excluded_rows),
            "exclusions_by_reason": dict(
                sorted(Counter(reason for row in excluded_rows for reason in row["reasons"]).items())
            ),
        },
        "quality_checks": {
            "model_index_ids_match": model_ids == {row["observation_id"] for row in index_rows},
            "model_diagnostic_ids_match": model_ids == {row["observation_id"] for row in diagnostic_rows},
            "pose_roundtrip_gate_passed": all(
                value["max_pose_roundtrip_error_scene_units"] <= args.max_roundtrip_error
                for value in video_quality.values()
            ),
            "reprojection_gate_passed": all(
                value["max_reprojection_error_pixels"] <= args.max_reprojection_error_pixels
                for value in video_quality.values()
            ),
            "rotation_orthonormality_gate_passed": all(
                value["max_rotation_orthonormality_error"] <= args.max_rotation_orthonormality_error
                for value in video_quality.values()
            ),
            "forbidden_model_key_fragments_checked": list(FORBIDDEN_MODEL_KEY_FRAGMENTS),
        },
        "output_sha256": {filename: sha256(args.output_dir / filename) for filename in output_files},
    }
    write_json(args.output_dir / "phase1_adapter_manifest.json", manifest)
    print(
        f"Complete: {spec.display_name}, {len(found)} videos, {len(model_rows)} included, "
        f"{len(excluded_rows)} excluded observations",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
