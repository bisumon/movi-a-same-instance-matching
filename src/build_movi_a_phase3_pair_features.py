#!/usr/bin/env python3
"""Build leakage-safe pair features for the four Phase 3 configurations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


VERSION = "1.0.0"
RGB_FEATURES = ["rgb_cosine_similarity"]
TWO_D_FEATURES = [
    "temporal_gap",
    "mask_center_dx_abs",
    "mask_center_dy_abs",
    "mask_center_distance",
    "log_mask_area_ratio_abs",
    "mean_log_mask_area",
    "log_crop_area_ratio_abs",
    "mean_log_crop_area",
    "log_bbox_aspect_ratio_diff_abs",
    "log_visibility_ratio_abs",
    "min_log_visibility",
]
THREE_D_FEATURES = [
    "camera_centroid_dx_abs",
    "camera_centroid_dy_abs",
    "camera_centroid_dz_abs",
    "camera_centroid_distance",
    "camera_centroid_distance_per_frame",
    "depth_q05_diff_abs",
    "depth_q25_diff_abs",
    "depth_median_diff_abs",
    "depth_q75_diff_abs",
    "depth_q95_diff_abs",
    "depth_iqr_diff_abs",
    "depth_median_diff_per_frame",
    "extent_x_diff_abs",
    "extent_y_diff_abs",
    "extent_z_diff_abs",
    "extent_l2_diff",
    "extent_x_log_ratio_abs",
    "extent_y_log_ratio_abs",
    "extent_z_log_ratio_abs",
]
FEATURE_NAMES = RGB_FEATURES + TWO_D_FEATURES + THREE_D_FEATURES
FORBIDDEN_FEATURE_FRAGMENTS = (
    "video_id",
    "instance_index",
    "shape",
    "color",
    "material",
    "world",
    "velocity",
    "split",
    "label",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-inputs", type=Path, required=True)
    parser.add_argument("--observation-index", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def safe_log_ratio(left: float, right: float, epsilon: float = 1e-8) -> float:
    return abs(math.log(max(left, epsilon) / max(right, epsilon)))


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0:
        raise ValueError("Zero-norm RGB embedding")
    return float(np.dot(left, right) / denominator)


def pair_features(
    left: dict[str, Any],
    right: dict[str, Any],
    left_embedding: np.ndarray,
    right_embedding: np.ndarray,
    temporal_gap: int,
) -> list[float]:
    center_dx = abs(float(left["mask_center_x_normalized"]) - float(right["mask_center_x_normalized"]))
    center_dy = abs(float(left["mask_center_y_normalized"]) - float(right["mask_center_y_normalized"]))
    left_crop_area = float(left["padded_crop_width"] * left["padded_crop_height"])
    right_crop_area = float(right["padded_crop_width"] * right["padded_crop_height"])
    centroid_left = np.asarray(left["camera_space_visible_surface_centroid_xyz"], dtype=np.float64)
    centroid_right = np.asarray(right["camera_space_visible_surface_centroid_xyz"], dtype=np.float64)
    centroid_delta = np.abs(centroid_left - centroid_right)
    centroid_distance = float(np.linalg.norm(centroid_left - centroid_right))
    depth_left, depth_right = left["depth"], right["depth"]
    extent_left = np.asarray(left["camera_space_visible_surface_extent_q05_q95_xyz"], dtype=np.float64)
    extent_right = np.asarray(right["camera_space_visible_surface_extent_q05_q95_xyz"], dtype=np.float64)
    extent_delta = np.abs(extent_left - extent_right)
    depth_differences = [
        abs(float(depth_left[name]) - float(depth_right[name]))
        for name in ("q05", "q25", "median", "q75", "q95")
    ]
    left_iqr = float(depth_left["q75"]) - float(depth_left["q25"])
    right_iqr = float(depth_right["q75"]) - float(depth_right["q25"])
    values = [
        cosine_similarity(left_embedding, right_embedding),
        float(temporal_gap),
        center_dx,
        center_dy,
        math.hypot(center_dx, center_dy),
        safe_log_ratio(float(left["mask_area"]), float(right["mask_area"])),
        0.5 * (math.log1p(float(left["mask_area"])) + math.log1p(float(right["mask_area"]))),
        safe_log_ratio(left_crop_area, right_crop_area),
        0.5 * (math.log1p(left_crop_area) + math.log1p(right_crop_area)),
        safe_log_ratio(float(left["bbox_aspect_ratio"]), float(right["bbox_aspect_ratio"])),
        safe_log_ratio(float(left["visibility"]), float(right["visibility"])),
        min(math.log1p(float(left["visibility"])), math.log1p(float(right["visibility"]))),
        float(centroid_delta[0]),
        float(centroid_delta[1]),
        float(centroid_delta[2]),
        centroid_distance,
        centroid_distance / temporal_gap,
        *depth_differences,
        abs(left_iqr - right_iqr),
        depth_differences[2] / temporal_gap,
        float(extent_delta[0]),
        float(extent_delta[1]),
        float(extent_delta[2]),
        float(np.linalg.norm(extent_left - extent_right)),
        safe_log_ratio(float(extent_left[0]), float(extent_right[0])),
        safe_log_ratio(float(extent_left[1]), float(extent_right[1])),
        safe_log_ratio(float(extent_left[2]), float(extent_right[2])),
    ]
    if len(values) != len(FEATURE_NAMES):
        raise RuntimeError(f"Feature width mismatch: {len(values)} vs {len(FEATURE_NAMES)}")
    return values


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in FEATURE_NAMES:
        if any(fragment in name.lower() for fragment in FORBIDDEN_FEATURE_FRAGMENTS):
            raise ValueError(f"Forbidden leakage-sensitive feature name: {name}")

    model_rows = load_jsonl(args.model_inputs)
    model_by_id = {str(row["observation_id"]): row for row in model_rows}
    index_rows = load_jsonl(args.observation_index)
    index_by_id = {str(row["observation_id"]): row for row in index_rows}
    if set(model_by_id) != set(index_by_id):
        raise ValueError("Model-input and observation-index ID sets differ")
    pair_rows = load_jsonl(args.pairs)
    embedding_data = np.load(args.embeddings, allow_pickle=False)
    embedding_ids = embedding_data["observation_ids"].astype(str)
    embedding_matrix = embedding_data["embeddings"].astype(np.float32, copy=False)
    embedding_by_id = {observation_id: embedding_matrix[index] for index, observation_id in enumerate(embedding_ids)}
    if set(embedding_by_id) != set(model_by_id):
        raise ValueError("Embedding and Phase 1 observation ID sets differ")

    matrix_rows = []
    pair_ids = []
    splits = []
    labels = []
    difficulties = []
    for pair in pair_rows:
        left_id, right_id = str(pair["observation_id_a"]), str(pair["observation_id_b"])
        left_index, right_index = index_by_id[left_id], index_by_id[right_id]
        if left_index["video_id"] != right_index["video_id"] or left_index["split"] != right_index["split"]:
            raise ValueError(f"Cross-video or cross-split pair encountered: {pair['pair_id']}")
        if str(pair["split"]) != str(left_index["split"]):
            raise ValueError(f"Pair split disagrees with observation index: {pair['pair_id']}")
        matrix_rows.append(
            pair_features(
                model_by_id[left_id],
                model_by_id[right_id],
                embedding_by_id[left_id],
                embedding_by_id[right_id],
                int(pair["temporal_gap"]),
            )
        )
        pair_ids.append(str(pair["pair_id"]))
        splits.append(str(pair["split"]))
        labels.append(int(pair["label"]))
        difficulties.append(str(pair["negative_difficulty"] or "positive"))

    matrix = np.asarray(matrix_rows, dtype=np.float32)
    if matrix.shape != (len(pair_rows), len(FEATURE_NAMES)) or not np.isfinite(matrix).all():
        raise RuntimeError(f"Unexpected or non-finite feature matrix: {matrix.shape}")
    output_path = args.output_dir / "phase3_pair_features.npz"
    np.savez_compressed(
        output_path,
        pair_ids=np.asarray(pair_ids, dtype="U24"),
        splits=np.asarray(splits, dtype="U5"),
        labels=np.asarray(labels, dtype=np.int8),
        negative_difficulty=np.asarray(difficulties, dtype="U8"),
        feature_names=np.asarray(FEATURE_NAMES, dtype="U48"),
        features=matrix,
    )
    feature_groups = {
        "rgb": RGB_FEATURES,
        "2d_controls": TWO_D_FEATURES,
        "depth_3d": THREE_D_FEATURES,
        "configurations": {
            "A_rgb_only": RGB_FEATURES,
            "B_rgb_2d": RGB_FEATURES + TWO_D_FEATURES,
            "C_rgb_2d_3d": RGB_FEATURES + TWO_D_FEATURES + THREE_D_FEATURES,
            "geometry_only": TWO_D_FEATURES + THREE_D_FEATURES,
        },
    }
    manifest = {
        "pipeline": "MOVi-A Phase 3 leakage-safe pair feature construction",
        "version": VERSION,
        "inputs": {
            "model_inputs_sha256": sha256(args.model_inputs),
            "observation_index_sha256": sha256(args.observation_index),
            "pairs_sha256": sha256(args.pairs),
            "embeddings_sha256": sha256(args.embeddings),
        },
        "counts": {
            "pairs": len(pair_rows),
            "pairs_by_split": dict(sorted(Counter(splits).items())),
            "labels": dict(sorted(Counter(labels).items())),
            "feature_count": len(FEATURE_NAMES),
        },
        "feature_groups": feature_groups,
        "leakage_guard": {
            "forbidden_feature_name_fragments": list(FORBIDDEN_FEATURE_FRAGMENTS),
            "ground_truth_diagnostics_loaded": False,
            "instance_attribute_labels_loaded": False,
            "video_and_instance_ids_used_only_for_pair-integrity_checks": True,
        },
        "quality_checks": {
            "shape": list(matrix.shape),
            "all_finite": bool(np.isfinite(matrix).all()),
            "unique_pair_ids": len(set(pair_ids)),
            "feature_min": {name: float(matrix[:, index].min()) for index, name in enumerate(FEATURE_NAMES)},
            "feature_max": {name: float(matrix[:, index].max()) for index, name in enumerate(FEATURE_NAMES)},
        },
        "output": {"filename": output_path.name, "sha256": sha256(output_path)},
    }
    write_json(args.output_dir / "phase3_feature_manifest.json", manifest)
    print(f"Complete: {matrix.shape[0]} pairs x {matrix.shape[1]} features", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
