#!/usr/bin/env python3
"""Dataset and coordinate adapter shared by MOVi-D and MOVi-E extraction.

The adapter normalizes the public TFDS TFRecord schemas without importing
TensorFlow.  It deliberately keeps simulator object identity and semantic
metadata separate from inference-available camera/depth geometry.
"""

from __future__ import annotations

import hashlib
import io
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
from PIL import Image


CV_TO_KUBRIC_AXES = np.asarray([1.0, -1.0, -1.0], dtype=np.float64)
COMMON_REQUIRED_KEYS = frozenset(
    {
        "metadata/video_name",
        "metadata/depth_range",
        "metadata/num_frames",
        "metadata/num_instances",
        "metadata/height",
        "metadata/width",
        "camera/focal_length",
        "camera/sensor_width",
        "camera/positions",
        "camera/quaternions",
        "instances/asset_id",
        "instances/category",
        "instances/scale",
        "instances/is_dynamic",
        "instances/positions",
        "instances/velocities",
        "instances/bboxes_3d",
        "instances/visibility",
        "video",
        "segmentations",
        "depth",
        "background",
    }
)


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    name: str
    display_name: str
    camera_regime: str
    expected_motion: str

    @property
    def shard_prefix(self) -> str:
        return self.name


DATASET_SPECS = {
    "movi_d": DatasetSpec(
        name="movi_d",
        display_name="MOVi-D",
        camera_regime="fixed_random",
        expected_motion="fixed within each video",
    ),
    "movi_e": DatasetSpec(
        name="movi_e",
        display_name="MOVi-E",
        camera_regime="linear_movement",
        expected_motion="linear translation while looking toward the scene origin",
    ),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    normalized = name.lower().replace("-", "_")
    try:
        return DATASET_SPECS[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported dataset {name!r}; expected one of {sorted(DATASET_SPECS)}"
        ) from exc


def scalar(record: dict[str, Any], key: str, cast: type = float) -> Any:
    value = np.asarray(record[key]).reshape(-1)
    if value.size != 1:
        raise ValueError(f"Expected scalar feature {key}, got shape {value.shape}")
    return cast(value[0])


def text_scalar(value: Any) -> str:
    flattened = np.asarray(value).reshape(-1)
    if flattened.size != 1:
        raise ValueError(f"Expected one text value, got shape {flattened.shape}")
    item = flattened[0]
    return bytes(item).decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)


def text_at(values: Any, index: int) -> str:
    item = np.asarray(values).reshape(-1)[index]
    return bytes(item).decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)


def record_video_id(record: dict[str, Any]) -> str:
    return text_scalar(record["metadata/video_name"])


def validate_record_schema(record: dict[str, Any], dataset: str | DatasetSpec) -> None:
    spec = get_dataset_spec(dataset) if isinstance(dataset, str) else dataset
    missing = sorted(COMMON_REQUIRED_KEYS - set(record))
    if missing:
        raise ValueError(f"{spec.display_name} record is missing required features: {missing}")
    num_frames = scalar(record, "metadata/num_frames", int)
    num_instances = scalar(record, "metadata/num_instances", int)
    expected_shapes = {
        "camera/positions": (num_frames, 3),
        "camera/quaternions": (num_frames, 4),
        "instances/visibility": (num_instances, num_frames),
        "instances/positions": (num_instances, num_frames, 3),
        "instances/velocities": (num_instances, num_frames, 3),
        "instances/bboxes_3d": (num_instances, num_frames, 8, 3),
    }
    for key, shape in expected_shapes.items():
        if np.asarray(record[key]).size != math.prod(shape):
            raise ValueError(
                f"{spec.display_name} feature {key} cannot reshape to {shape}; "
                f"found {np.asarray(record[key]).shape}"
            )
    for key in ("instances/asset_id", "instances/category", "instances/scale", "instances/is_dynamic"):
        if np.asarray(record[key]).size != num_instances:
            raise ValueError(
                f"{spec.display_name} feature {key} has {np.asarray(record[key]).size} "
                f"values for {num_instances} instances"
            )
    for key in ("video", "segmentations", "depth"):
        if len(record[key]) != num_frames:
            raise ValueError(
                f"{spec.display_name} feature {key} has {len(record[key])} frames; expected {num_frames}"
            )


def discover_shards(
    directory: Path,
    dataset: str,
    split: str = "validation",
    require_complete: bool = True,
) -> list[Path]:
    """Return ordered shards and validate their shared declared total.

    Confirmatory runs require the complete set. Pilot runs may explicitly set
    ``require_complete=False`` when a download manifest records the chosen
    subset of shard indices.
    """
    spec = get_dataset_spec(dataset)
    pattern = re.compile(
        rf"^{re.escape(spec.shard_prefix)}-{re.escape(split)}\.tfrecord-(\d+)-of-(\d+)$"
    )
    candidates = sorted(directory.glob(f"{spec.shard_prefix}-{split}.tfrecord-*-of-*"))
    parsed: list[tuple[int, int, Path]] = []
    for path in candidates:
        match = pattern.match(path.name)
        if not match:
            continue
        parsed.append((int(match.group(1)), int(match.group(2)), path))
    if not parsed:
        raise FileNotFoundError(
            f"No {spec.display_name} {split} TFRecord shards found in {directory}"
        )
    declared_counts = {total for _, total, _ in parsed}
    if len(declared_counts) != 1:
        raise ValueError(f"Inconsistent declared shard totals: {sorted(declared_counts)}")
    total = declared_counts.pop()
    indices = [index for index, _, _ in parsed]
    if len(indices) != len(set(indices)):
        raise ValueError(f"Duplicate {spec.display_name} shard index in {directory}")
    expected = list(range(total))
    if require_complete and sorted(indices) != expected:
        missing = sorted(set(expected) - set(indices))
        extra = sorted(set(indices) - set(expected))
        raise FileNotFoundError(
            f"Incomplete {spec.display_name} {split} shard set: found {len(indices)}/{total}; "
            f"missing={missing}, extra={extra}"
        )
    return [path for _, _, path in sorted(parsed)]


def iter_tfrecords(shards: Iterable[Path]) -> Iterator[tuple[Path, dict[str, Any]]]:
    try:
        from tfrecord.reader import tfrecord_loader
    except ImportError as exc:
        raise RuntimeError(
            "The lightweight 'tfrecord' package is required to read MOVi TFRecords."
        ) from exc
    for shard in shards:
        for record in tfrecord_loader(str(shard), None):
            yield shard, record


def decode_png(blob: bytes | np.bytes_) -> np.ndarray:
    with Image.open(io.BytesIO(bytes(blob))) as image:
        return np.asarray(image).copy()


def decode_depth(blob: bytes | np.bytes_, depth_range: np.ndarray) -> np.ndarray:
    """Decode TFDS uint16 depth into radial camera-center distance."""
    encoded = decode_png(blob).astype(np.float64)
    if encoded.ndim == 3:
        encoded = encoded[..., 0]
    minimum, maximum = (float(depth_range[0]), float(depth_range[1]))
    return encoded / 65535.0 * (maximum - minimum) + minimum


def padded_bbox(mask: np.ndarray, padding: float) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        raise ValueError("Cannot compute a bounding box for an empty mask")
    x0, x1 = int(cols.min()), int(cols.max()) + 1
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    width, height = x1 - x0, y1 - y0
    pad_x = int(math.ceil(width * padding))
    pad_y = int(math.ceil(height * padding))
    frame_height, frame_width = mask.shape
    return (x0, y0, x1, y1), (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(frame_width, x1 + pad_x),
        min(frame_height, y1 + pad_y),
    )


def masked_camera_points(
    mask: np.ndarray,
    depth: np.ndarray,
    focal_x_px: float,
    focal_y_px: float,
) -> np.ndarray:
    """Back-project masked radial depth into CV axes (right, down, forward)."""
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


def project_camera_points(
    points_cv: np.ndarray,
    focal_x_px: float,
    focal_y_px: float,
    width: int,
    height: int,
) -> np.ndarray:
    points = np.asarray(points_cv, dtype=np.float64).reshape(-1, 3)
    if np.any(points[:, 2] <= 0):
        raise ValueError("Cannot project camera points with non-positive forward coordinate")
    cols = focal_x_px * points[:, 0] / points[:, 2] + width / 2.0 - 0.5
    rows = focal_y_px * points[:, 1] / points[:, 2] + height / 2.0 - 0.5
    return np.column_stack((cols, rows))


def quaternion_to_rotation_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Convert a Kubric wxyz quaternion to a camera-to-world rotation."""
    q = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
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


def camera_cv_to_world(
    points_cv: np.ndarray,
    camera_position_world: np.ndarray,
    quaternion_wxyz: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points_cv, dtype=np.float64).reshape(-1, 3)
    rotation = quaternion_to_rotation_matrix(quaternion_wxyz)
    points_kubric = points * CV_TO_KUBRIC_AXES
    return np.asarray(camera_position_world, dtype=np.float64).reshape(1, 3) + points_kubric @ rotation.T


def world_to_camera_cv(
    points_world: np.ndarray,
    camera_position_world: np.ndarray,
    quaternion_wxyz: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    rotation = quaternion_to_rotation_matrix(quaternion_wxyz)
    points_kubric = (points - np.asarray(camera_position_world, dtype=np.float64)) @ rotation
    return points_kubric * CV_TO_KUBRIC_AXES


def relative_rotation_degrees(left_wxyz: np.ndarray, right_wxyz: np.ndarray) -> float:
    left = quaternion_to_rotation_matrix(left_wxyz)
    right = quaternion_to_rotation_matrix(right_wxyz)
    relative = left.T @ right
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def rotation_orthonormality_error(quaternion_wxyz: np.ndarray) -> float:
    rotation = quaternion_to_rotation_matrix(quaternion_wxyz)
    return float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))


def camera_motion_summary(positions: np.ndarray, quaternions: np.ndarray) -> dict[str, float]:
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    quaternions = np.asarray(quaternions, dtype=np.float64).reshape(-1, 4)
    step_translation = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    step_rotation = np.asarray(
        [relative_rotation_degrees(quaternions[i], quaternions[i + 1]) for i in range(len(quaternions) - 1)]
    )
    distance_from_origin = np.linalg.norm(positions, axis=1)
    total_displacement = float(np.linalg.norm(positions[-1] - positions[0]))
    return {
        "translation_start_to_end_scene_units": total_displacement,
        "translation_path_length_scene_units": float(step_translation.sum()),
        "translation_step_median_scene_units": float(np.median(step_translation)),
        "translation_step_max_scene_units": float(np.max(step_translation)),
        "rotation_start_to_end_degrees": relative_rotation_degrees(quaternions[0], quaternions[-1]),
        "rotation_step_median_degrees": float(np.median(step_rotation)),
        "rotation_step_max_degrees": float(np.max(step_rotation)),
        "mean_camera_to_origin_distance_scene_units": float(np.mean(distance_from_origin)),
        "normalized_start_to_end_translation": total_displacement / float(np.mean(distance_from_origin)),
    }


def stable_observation_id(dataset: str, video_id: str, frame_index: int, instance_index: int) -> str:
    spec = get_dataset_spec(dataset)
    source = f"{spec.name}|{video_id}|{frame_index}|{instance_index}".encode("utf-8")
    return hashlib.sha256(source).hexdigest()[:20]


def normalize_instance_metadata(
    record: dict[str, Any],
    dataset: str,
    video_id: str,
    split: str,
) -> list[dict[str, Any]]:
    spec = get_dataset_spec(dataset)
    num_instances = scalar(record, "metadata/num_instances", int)
    background = text_scalar(record["background"])
    scales = np.asarray(record["instances/scale"], dtype=np.float64).reshape(-1)
    dynamics = np.asarray(record["instances/is_dynamic"]).reshape(-1)
    rows = []
    for instance_index in range(num_instances):
        rows.append(
            {
                "dataset": spec.name,
                "video_id": video_id,
                "split": split,
                "instance_index": instance_index,
                "segmentation_id": instance_index + 1,
                "asset_id": text_at(record["instances/asset_id"], instance_index),
                "category": text_at(record["instances/category"], instance_index),
                "scale": float(scales[instance_index]),
                "is_dynamic": bool(dynamics[instance_index]),
                "background": background,
            }
        )
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
