#!/usr/bin/env python3
"""Build Phase 8 regime-2 feature matrices for locked fixed-camera MOVi-D pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from build_movi_e_phase8_in_domain_features import SYSTEM_ORDER, frame_key, pose, shuffled_summary
from run_movi_de_phase7_pose_noise import geometry_features
from validate_movi_de_phase6_systems import deranged_pose_assignment, resolve_features, validate_config


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    if not all(validate_config(config).values()): raise ValueError("Invalid Phase 6 configuration")
    seed = int(config["seed"])
    pairs, model_rows, index_rows = read_jsonl(args.pairs), read_jsonl(args.model_inputs), read_jsonl(args.observation_index)
    model = {str(row["observation_id"]): row for row in model_rows}
    index = {str(row["observation_id"]): row for row in index_rows}
    required = {str(pair[key]) for pair in pairs for key in ("observation_id_a", "observation_id_b")}
    if len(pairs) != 10000 or set(model) != required or set(index) != required or any(str(row["dataset"]) != "movi_d" for row in index_rows):
        raise ValueError("Inputs do not exactly match locked MOVi-D pair endpoints")
    phase7 = np.load(args.phase7_features, allow_pickle=False)
    pair_ids = phase7["pair_ids"].astype(str)
    if pair_ids.tolist() != [str(row["pair_id"]) for row in pairs]: raise ValueError("Pair order mismatch")
    common = phase7["common_features"].astype(np.float32, copy=False)
    common_names = phase7["common_feature_names"].astype(str).tolist()
    condition_ids = phase7["condition_ids"].astype(str).tolist()
    world = phase7["geometry_features"][condition_ids.index("N_t0_r0")].astype(np.float32, copy=False)
    groups = config["feature_groups"]
    if common_names != groups["rgb"] + groups["two_d"] + groups["shared_radial_depth"]:
        raise RuntimeError("Phase 7 common feature order mismatch")

    frame_rows = {}
    for observation_id, meta in index.items():
        key = frame_key(meta); row = model[observation_id]
        if key in frame_rows and row["camera_pose"] != frame_rows[key]["camera_pose"]: raise ValueError(f"Pose differs within frame {key}")
        frame_rows[key] = row
    donor_by_frame = {}
    for split in ("train", "dev", "test"):
        donor_by_frame.update(deranged_pose_assignment([key for key in frame_rows if key[1] == split], seed))
    shuffled_observations = {}
    for observation_id, meta in index.items():
        donor = donor_by_frame[frame_key(meta)]
        donor_position, donor_rotation = pose(frame_rows[donor])
        shuffled_observations[observation_id] = shuffled_summary(model[observation_id], donor_position, donor_rotation)

    camera = np.empty((10000, 12), dtype=np.float32)
    shuffled = np.empty((10000, 12), dtype=np.float32)
    pose_only = np.empty((10000, 3), dtype=np.float32)
    for row_number, pair in enumerate(pairs):
        a, b = str(pair["observation_id_a"]), str(pair["observation_id_b"]); gap = int(pair["temporal_gap"])
        camera[row_number] = geometry_features(
            np.asarray(model[a]["camera_space_visible_surface_centroid_xyz"]), np.asarray(model[a]["camera_space_visible_surface_extent_q05_q95_xyz"]),
            np.asarray(model[b]["camera_space_visible_surface_centroid_xyz"]), np.asarray(model[b]["camera_space_visible_surface_extent_q05_q95_xyz"]), gap,
        )
        ca, ea = shuffled_observations[a]; cb, eb = shuffled_observations[b]
        shuffled[row_number] = geometry_features(ca, ea, cb, eb, gap)
        controls = pair["controls"]
        pose_only[row_number] = [controls["camera_displacement_scene_units"], controls["relative_camera_rotation_degrees"], controls["normalized_camera_displacement"]]
    rgb, two_d, depth = common[:, :1], common[:, 1:12], common[:, 12:19]
    matrices = {
        "A_rgb": rgb, "B_rgb_2d": np.concatenate((rgb, two_d), axis=1),
        "C_camera_geometry": np.concatenate((rgb, two_d, depth, camera), axis=1),
        "D_pose_aligned_geometry": np.concatenate((rgb, two_d, depth, world), axis=1),
        "G_camera_geometry_only": np.concatenate((depth, camera), axis=1),
        "G_pose_aligned_geometry_only": np.concatenate((depth, world), axis=1),
        "P_pose_only": pose_only, "S_shuffled_pose": np.concatenate((rgb, two_d, depth, shuffled), axis=1),
    }
    unique_poses = {
        (tuple(row["camera_pose"]["position_world_xyz"]), tuple(row["camera_pose"]["camera_to_world_quaternion_wxyz"]))
        for row in frame_rows.values()
    }
    checks = {
        "all_system_shapes_match_allowlists": all(matrix.shape == (10000, len(resolve_features(config, system_id))) for system_id, matrix in matrices.items()),
        "all_features_finite": all(np.isfinite(matrix).all() for matrix in matrices.values()),
        "C_D_equal_width": matrices["C_camera_geometry"].shape[1] == matrices["D_pose_aligned_geometry"].shape[1] == 31,
        "D_byte_identical_to_phase7_clean_D": np.array_equal(matrices["D_pose_aligned_geometry"], np.concatenate((common, world), axis=1)),
        "all_pose_only_controls_structural_zero": np.array_equal(pose_only, np.zeros_like(pose_only)),
        "fixed_pose_within_every_video": all(
            len({(tuple(model[oid]["camera_pose"]["position_world_xyz"]), tuple(model[oid]["camera_pose"]["camera_to_world_quaternion_wxyz"])) for oid, meta in index.items() if str(meta["video_id"]) == video}) == 1
            for video in {str(meta["video_id"]) for meta in index.values()}
        ),
        "static_pose_varies_across_videos": len(unique_poses) > 1,
        "shuffle_has_no_frame_fixed_points": all(source != donor for source, donor in donor_by_frame.items()),
        "shuffle_stays_within_dataset_and_split": all(source[:2] == donor[:2] for source, donor in donor_by_frame.items()),
        "S_differs_from_D_due_to_cross_video_static_pose_variation": not np.array_equal(matrices["S_shuffled_pose"], matrices["D_pose_aligned_geometry"]),
        "locked_pair_split_counts": Counter(row["split"] for row in pairs) == Counter({"train": 6000, "dev": 2000, "test": 2000}),
    }
    if not all(checks.values()): raise RuntimeError(f"Feature audit failed: {[key for key, value in checks.items() if not value]}")
    feature_path = args.output_dir / "movi_d_phase8_in_domain_features.npz"
    np.savez_compressed(feature_path, pair_ids=np.asarray(pair_ids, dtype="U24"), splits=np.asarray([row["split"] for row in pairs], dtype="U5"), labels=np.asarray([row["label"] for row in pairs], dtype=np.int8), system_ids=np.asarray(SYSTEM_ORDER, dtype="U32"), **matrices)
    shuffle_path = args.output_dir / "movi_d_phase8_shuffled_pose_assignment.jsonl"
    with shuffle_path.open("w", encoding="utf-8") as handle:
        for source in sorted(donor_by_frame):
            donor = donor_by_frame[source]
            handle.write(json.dumps({"source": {"dataset": source[0], "split": source[1], "video_id": source[2], "frame_index": source[3]}, "donor": {"dataset": donor[0], "split": donor[1], "video_id": donor[2], "frame_index": donor[3]}}, sort_keys=True, separators=(",", ":")) + "\n")
    manifest_path = args.output_dir / "movi_d_phase8_feature_manifest.json"
    manifest_path.write_text(json.dumps({
        "pipeline": "MOVi-D Phase 8 regime 2 feature construction", "version": "1.0.0", "dataset": "movi_d", "seed": seed,
        "systems": {system_id: {"feature_names": resolve_features(config, system_id), "feature_count": matrices[system_id].shape[1]} for system_id in SYSTEM_ORDER},
        "fixed_camera_audit": {"unique_static_poses_across_selected_videos": len(unique_poses), "note": "Pose is fixed within video but differs across videos; the frozen split-wide shuffle is therefore nontrivial."},
        "leakage_boundary": {"labels_used_in_feature_construction": False, "evaluation_only_object_metadata_loaded": False, "identity_fields_used_only_for_integrity_and_shuffle": True},
        "checks": checks,
        "inputs": {name: sha256(path) for name, path in {"phase7_features": args.phase7_features, "model_inputs": args.model_inputs, "observation_index": args.observation_index, "pairs": args.pairs, "system_config": args.system_config}.items()},
        "outputs": {feature_path.name: sha256(feature_path), shuffle_path.name: sha256(shuffle_path)},
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Complete: 8 MOVi-D system matrices; {len(unique_poses)} static per-video poses")
    return 0


if __name__ == "__main__": raise SystemExit(main())
