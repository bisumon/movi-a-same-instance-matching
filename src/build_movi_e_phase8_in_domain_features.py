#!/usr/bin/env python3
"""Build leakage-safe Phase 8 regime-1 feature matrices for locked MOVi-E pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from movi_de_dataset_adapter import quaternion_to_rotation_matrix
from run_movi_de_phase7_pose_noise import geometry_features
from validate_movi_de_phase6_systems import deranged_pose_assignment, resolve_features, validate_config


VERSION = "1.0.0"
SYSTEM_ORDER = (
    "A_rgb", "B_rgb_2d", "C_camera_geometry", "D_pose_aligned_geometry",
    "G_camera_geometry_only", "G_pose_aligned_geometry_only", "P_pose_only", "S_shuffled_pose",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_key(meta: dict[str, Any]) -> tuple[str, str, str, int]:
    return (str(meta["dataset"]), str(meta["split"]), str(meta["video_id"]), int(meta["frame_index"]))


def pose(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    value = row["camera_pose"]
    return (
        np.asarray(value["position_world_xyz"], dtype=np.float64),
        quaternion_to_rotation_matrix(np.asarray(value["camera_to_world_quaternion_wxyz"], dtype=np.float64)),
    )


def shuffled_summary(
    row: dict[str, Any], donor_position: np.ndarray, donor_rotation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    source_position, source_rotation = pose(row)
    center = np.asarray(row["pose_aligned_world_visible_surface_centroid_xyz"], dtype=np.float64)
    extent = np.asarray(row["pose_aligned_world_visible_surface_extent_q05_q95_xyz"], dtype=np.float64)
    delta = donor_rotation @ source_rotation.T
    return donor_position + delta @ (center - source_position), np.abs(delta) @ extent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase7-features", type=Path, required=True)
    parser.add_argument("--model-inputs", type=Path, required=True)
    parser.add_argument("--observation-index", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--system-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.system_config.read_text(encoding="utf-8"))
    if not all(validate_config(config).values()):
        raise ValueError("Invalid frozen Phase 6 system configuration")
    seed = int(config["seed"])
    pairs = read_jsonl(args.pairs)
    model_rows = read_jsonl(args.model_inputs)
    index_rows = read_jsonl(args.observation_index)
    model = {str(row["observation_id"]): row for row in model_rows}
    index = {str(row["observation_id"]): row for row in index_rows}
    required = {str(pair[key]) for pair in pairs for key in ("observation_id_a", "observation_id_b")}
    if len(pairs) != 10000 or len({row["pair_id"] for row in pairs}) != 10000 or set(model) != required or set(index) != required:
        raise ValueError("Inputs do not match the exact locked MOVi-E pair endpoints")
    if any(str(row["dataset"]) != "movi_e" for row in index_rows):
        raise ValueError("Regime 1 accepts MOVi-E observations only")

    phase7 = np.load(args.phase7_features, allow_pickle=False)
    pair_ids = phase7["pair_ids"].astype(str)
    if pair_ids.tolist() != [str(row["pair_id"]) for row in pairs]:
        raise ValueError("Phase 7 feature order differs from locked pair order")
    common_names = phase7["common_feature_names"].astype(str).tolist()
    common = phase7["common_features"].astype(np.float32, copy=False)
    condition_ids = phase7["condition_ids"].astype(str).tolist()
    zero_index = condition_ids.index("N_t0_r0")
    world = phase7["geometry_features"][zero_index].astype(np.float32, copy=False)
    if common.shape != (10000, 19) or world.shape != (10000, 12):
        raise ValueError("Unexpected Phase 7 feature shapes")

    # Build one deterministic derangement independently in train/dev/test.
    frame_rows: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for observation_id, meta in index.items():
        key = frame_key(meta)
        current = model[observation_id]
        if key in frame_rows and current["camera_pose"] != frame_rows[key]["camera_pose"]:
            raise ValueError(f"Camera pose differs within frame {key}")
        frame_rows[key] = current
    donor_by_frame = {}
    for split in ("train", "dev", "test"):
        keys = [key for key in frame_rows if key[1] == split]
        donor_by_frame.update(deranged_pose_assignment(keys, seed))
    if set(donor_by_frame) != set(frame_rows) or any(source == donor for source, donor in donor_by_frame.items()):
        raise RuntimeError("Seeded shuffled-pose assignment is not a complete derangement")
    shuffled_observations = {}
    for observation_id, meta in index.items():
        donor = donor_by_frame[frame_key(meta)]
        donor_position, donor_rotation = pose(frame_rows[donor])
        shuffled_observations[observation_id] = shuffled_summary(model[observation_id], donor_position, donor_rotation)

    camera = np.empty((10000, 12), dtype=np.float32)
    shuffled = np.empty((10000, 12), dtype=np.float32)
    pose_only = np.empty((10000, 3), dtype=np.float32)
    for row_number, pair in enumerate(pairs):
        a, b = str(pair["observation_id_a"]), str(pair["observation_id_b"])
        gap = int(pair["temporal_gap"])
        camera[row_number] = geometry_features(
            np.asarray(model[a]["camera_space_visible_surface_centroid_xyz"]),
            np.asarray(model[a]["camera_space_visible_surface_extent_q05_q95_xyz"]),
            np.asarray(model[b]["camera_space_visible_surface_centroid_xyz"]),
            np.asarray(model[b]["camera_space_visible_surface_extent_q05_q95_xyz"]), gap,
        )
        center_a, extent_a = shuffled_observations[a]
        center_b, extent_b = shuffled_observations[b]
        shuffled[row_number] = geometry_features(center_a, extent_a, center_b, extent_b, gap)
        controls = pair["controls"]
        pose_only[row_number] = [
            controls["camera_displacement_scene_units"], controls["relative_camera_rotation_degrees"],
            controls["normalized_camera_displacement"],
        ]
    if not all(np.isfinite(value).all() for value in (common, world, camera, shuffled, pose_only)):
        raise RuntimeError("Non-finite Phase 8 feature")

    # Phase 7 common order is RGB + 2D + shared depth.
    groups = config["feature_groups"]
    expected_common = groups["rgb"] + groups["two_d"] + groups["shared_radial_depth"]
    if common_names != expected_common:
        raise RuntimeError("Phase 7 common feature order differs from Phase 6")
    rgb = common[:, :1]
    two_d = common[:, 1:12]
    depth = common[:, 12:19]
    matrices = {
        "A_rgb": rgb,
        "B_rgb_2d": np.concatenate((rgb, two_d), axis=1),
        "C_camera_geometry": np.concatenate((rgb, two_d, depth, camera), axis=1),
        "D_pose_aligned_geometry": np.concatenate((rgb, two_d, depth, world), axis=1),
        "G_camera_geometry_only": np.concatenate((depth, camera), axis=1),
        "G_pose_aligned_geometry_only": np.concatenate((depth, world), axis=1),
        "P_pose_only": pose_only,
        "S_shuffled_pose": np.concatenate((rgb, two_d, depth, shuffled), axis=1),
    }
    checks = {}
    for system_id, matrix in matrices.items():
        expected = resolve_features(config, system_id)
        checks[f"{system_id}_shape_and_width"] = matrix.shape == (10000, len(expected))
        checks[f"{system_id}_all_finite"] = bool(np.isfinite(matrix).all())
    checks.update({
        "C_D_equal_width": matrices["C_camera_geometry"].shape[1] == matrices["D_pose_aligned_geometry"].shape[1] == 31,
        "G_variants_equal_width": matrices["G_camera_geometry_only"].shape[1] == matrices["G_pose_aligned_geometry_only"].shape[1] == 19,
        "D_byte_identical_to_phase7_clean_D": np.array_equal(matrices["D_pose_aligned_geometry"], np.concatenate((common, world), axis=1)),
        "S_differs_from_D_on_MOVi_E": not np.array_equal(matrices["S_shuffled_pose"], matrices["D_pose_aligned_geometry"]),
        "shuffle_has_no_fixed_points": all(source != donor for source, donor in donor_by_frame.items()),
        "shuffle_stays_within_dataset_and_split": all(source[:2] == donor[:2] for source, donor in donor_by_frame.items()),
        "P_contains_only_three_pose_controls": matrices["P_pose_only"].shape[1] == 3,
        "pair_split_counts_locked": Counter(row["split"] for row in pairs) == Counter({"train": 6000, "dev": 2000, "test": 2000}),
    })
    if not all(checks.values()):
        raise RuntimeError(f"Feature audit failed: {[key for key, value in checks.items() if not value]}")
    output_path = args.output_dir / "movi_e_phase8_in_domain_features.npz"
    np.savez_compressed(
        output_path,
        pair_ids=np.asarray(pair_ids, dtype="U24"),
        splits=np.asarray([row["split"] for row in pairs], dtype="U5"),
        labels=np.asarray([row["label"] for row in pairs], dtype=np.int8),
        system_ids=np.asarray(SYSTEM_ORDER, dtype="U32"),
        **{system_id: matrix.astype(np.float32, copy=False) for system_id, matrix in matrices.items()},
    )
    shuffle_path = args.output_dir / "movi_e_phase8_shuffled_pose_assignment.jsonl"
    with shuffle_path.open("w", encoding="utf-8") as handle:
        for source in sorted(donor_by_frame):
            donor = donor_by_frame[source]
            handle.write(json.dumps({
                "source": {"dataset": source[0], "split": source[1], "video_id": source[2], "frame_index": source[3]},
                "donor": {"dataset": donor[0], "split": donor[1], "video_id": donor[2], "frame_index": donor[3]},
            }, sort_keys=True, separators=(",", ":")) + "\n")
    manifest_path = args.output_dir / "movi_e_phase8_feature_manifest.json"
    manifest_path.write_text(json.dumps({
        "pipeline": "MOVi-E Phase 8 regime 1 leakage-safe feature construction", "version": VERSION,
        "dataset": "movi_e", "seed": seed,
        "systems": {system_id: {"feature_names": resolve_features(config, system_id), "feature_count": matrices[system_id].shape[1]} for system_id in SYSTEM_ORDER},
        "counts": {"pairs": len(pairs), "pairs_by_split": dict(Counter(row["split"] for row in pairs)), "unique_frames": len(frame_rows)},
        "shuffled_pose": {
            "scope": "within MOVi-E and locked split", "assignment_rows": len(donor_by_frame),
            "sufficient_statistic_transform": "donor_position + donor_rotation * source_rotation^T * (clean_world_summary - source_position); extents use abs(delta_rotation) * clean_extent",
        },
        "leakage_boundary": {
            "identity_fields_used_only_for_joins_and_shuffle_assignment": True,
            "labels_used_in_feature_construction": False,
            "evaluation_only_object_metadata_loaded": False,
            "test_derived_normalization_loaded": False,
        },
        "checks": checks,
        "inputs": {name: sha256(path) for name, path in {
            "phase7_features": args.phase7_features, "model_inputs": args.model_inputs,
            "observation_index": args.observation_index, "pairs": args.pairs, "system_config": args.system_config,
        }.items()},
        "outputs": {output_path.name: sha256(output_path), shuffle_path.name: sha256(shuffle_path)},
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Complete: {len(SYSTEM_ORDER)} MOVi-E system matrices over 10,000 locked pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
