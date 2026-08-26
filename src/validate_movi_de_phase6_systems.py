#!/usr/bin/env python3
"""Validate, resolve, and checksum-lock MOVi-D/E Phase 6 system definitions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


EXPECTED_SYSTEMS = {
    "A_rgb", "B_rgb_2d", "C_camera_geometry", "D_pose_aligned_geometry",
    "G_camera_geometry_only", "G_pose_aligned_geometry_only", "P_pose_only",
    "S_shuffled_pose", "N_noisy_pose",
}
EXPECTED_C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
EXPECTED_TRANSLATION_GRID = [0, 0.01, 0.05, 0.1, 0.25, 0.5]
EXPECTED_ROTATION_GRID = [0, 0.1, 0.5, 1, 2, 5]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_rank(seed: int, parts: Iterable[object]) -> str:
    return hashlib.sha256("|".join(map(str, (seed, *parts))).encode()).hexdigest()


def deranged_pose_assignment(frame_keys: list[tuple[str, str, str, int]], seed: int) -> dict[tuple[str, str, str, int], tuple[str, str, str, int]]:
    """Return the Phase 6 deterministic one-step cyclic pose assignment."""
    if len(set(frame_keys)) != len(frame_keys):
        raise ValueError("Frame keys must be unique")
    if not frame_keys:
        return {}
    ordered = sorted(frame_keys, key=lambda key: (stable_rank(seed, key), key))
    if len(ordered) == 1:
        return {ordered[0]: ordered[0]}
    donors = ordered[1:] + ordered[:1]
    return dict(zip(ordered, donors, strict=True))


def level_token(value: float | int) -> str:
    return format(float(value), "g").replace(".", "p")


def noise_conditions(config: dict[str, Any]) -> list[dict[str, Any]]:
    noisy = config["noisy_pose"]
    return [
        {
            "condition_id": f"N_t{level_token(translation)}_r{level_token(rotation)}",
            "translation_std_scene_units": float(translation),
            "rotation_std_degrees": float(rotation),
        }
        for translation in noisy["translation_standard_deviation_scene_units"]
        for rotation in noisy["rotation_standard_deviation_degrees"]
    ]


def resolve_features(config: dict[str, Any], system_id: str) -> list[str]:
    groups = config["feature_groups"]
    features = []
    for group in config["systems"][system_id]["feature_groups"]:
        features.extend(groups[group])
    return features


def validate_config(config: dict[str, Any]) -> dict[str, bool]:
    groups = config["feature_groups"]
    systems = config["systems"]
    all_features = [feature for values in groups.values() for feature in values]
    evaluation_fields = config["information_boundary"]["evaluation_or_sampling_only_fields"]
    resolved = {system_id: resolve_features(config, system_id) for system_id in systems}
    camera = groups["camera_space_surface_geometry"]
    world = groups["pose_aligned_surface_geometry"]
    corresponding = config["corresponding_geometry_features"]
    pose_users = {system_id for system_id, features in resolved.items() if any(feature in groups["pose_only"] for feature in features)}
    transform_users = set(config["information_boundary"]["camera_pose_use_for_geometry_transform_allowed_for"])
    conditions = noise_conditions(config)
    checks = {
        "protocol_and_version": config["protocol_id"] == "MOVI-DE-POSE-001" and config["protocol_version"] == "0.2",
        "seed": int(config["seed"]) == 20260825,
        "exact_system_set": set(systems) == EXPECTED_SYSTEMS,
        "all_system_groups_exist": all(group in groups for system in systems.values() for group in system["feature_groups"]),
        "feature_names_unique_across_catalog": len(all_features) == len(set(all_features)),
        "resolved_system_features_unique": all(len(values) == len(set(values)) for values in resolved.values()),
        "no_evaluation_only_fields_in_feature_catalog": not any(
            forbidden == feature or feature.startswith(forbidden + "_")
            for feature in all_features for forbidden in evaluation_fields
        ),
        "regularization_grid_frozen": config["shared_training"]["regularization_grid_C"] == EXPECTED_C_GRID,
        "camera_world_geometry_dimensions_equal": len(camera) == len(world),
        "camera_world_correspondence_complete": len(corresponding) == len(camera)
        and {pair[0] for pair in corresponding} == set(camera)
        and {pair[1] for pair in corresponding} == set(world),
        "primary_C_D_dimensions_equal": len(resolved["C_camera_geometry"]) == len(resolved["D_pose_aligned_geometry"]),
        "primary_C_D_shared_noncoordinate_features": (
            set(resolved["C_camera_geometry"]) - set(camera)
            == set(resolved["D_pose_aligned_geometry"]) - set(world)
        ),
        "geometry_only_variants_equal_width": len(resolved["G_camera_geometry_only"]) == len(resolved["G_pose_aligned_geometry_only"]),
        "geometry_only_variants_exclude_rgb_and_2d": not (
            (set(groups["rgb"]) | set(groups["two_d"]))
            & (set(resolved["G_camera_geometry_only"]) | set(resolved["G_pose_aligned_geometry_only"]))
        ),
        "pose_only_contains_only_pose_features": resolved["P_pose_only"] == groups["pose_only"],
        "direct_pose_scalars_restricted_to_P": pose_users == {"P_pose_only"},
        "transform_pose_users_exact": transform_users == {
            "D_pose_aligned_geometry", "G_pose_aligned_geometry_only", "S_shuffled_pose", "N_noisy_pose"
        },
        "D_S_N_feature_dimensions_and_names_equal": resolved["D_pose_aligned_geometry"] == resolved["S_shuffled_pose"] == resolved["N_noisy_pose"],
        "translation_grid_frozen": config["noisy_pose"]["translation_standard_deviation_scene_units"] == EXPECTED_TRANSLATION_GRID,
        "rotation_grid_frozen": config["noisy_pose"]["rotation_standard_deviation_degrees"] == EXPECTED_ROTATION_GRID,
        "exact_36_unique_noise_conditions": len(conditions) == 36 and len({row["condition_id"] for row in conditions}) == 36,
        "zero_noise_condition_present": any(row["translation_std_scene_units"] == 0 and row["rotation_std_degrees"] == 0 for row in conditions),
        "noisy_pose_uses_clean_D_without_refitting": systems["N_noisy_pose"]["fit_policy"].startswith("apply the clean fitted D model"),
        "shuffled_pose_is_independently_fit": systems["S_shuffled_pose"]["fit_policy"].startswith("construct shuffled train/dev/test features and fit an independent model"),
        "transfer_has_no_E_refit": config["transfer"]["refit_on_movi_e"] is False,
    }
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--resolved-output", type=Path, required=True)
    parser.add_argument("--lock-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repository_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    checks = validate_config(config)
    if not all(checks.values()):
        failures = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Phase 6 system validation failed: {failures}")
    resolved = {
        "protocol_id": config["protocol_id"], "protocol_version": config["protocol_version"],
        "configuration_version": config["configuration_version"], "seed": config["seed"],
        "systems": {
            system_id: {
                **system,
                "resolved_features": resolve_features(config, system_id),
                "resolved_feature_count": len(resolve_features(config, system_id)),
            }
            for system_id, system in config["systems"].items()
        },
        "noisy_pose_conditions": noise_conditions(config),
        "validation_checks": checks,
    }
    args.resolved_output.parent.mkdir(parents=True, exist_ok=True)
    args.resolved_output.write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifact_paths = [
        args.config.resolve(), args.resolved_output.resolve(),
        root / "docs/MOVI_DE_CAMERA_POSE_PROTOCOL_v0.2.md",
        root / "docs/MOVI_DE_PHASE6_SYSTEM_CONFIGURATIONS.md",
        root / "src/validate_movi_de_phase6_systems.py",
        root / "tests/test_movi_de_phase6_systems.py",
        root / "manifests/movi_de/protocol_freeze_v0.2.json",
        root / "manifests/movi_de/phase5_pair_manifest_freeze.json",
    ]
    lock = {
        "lock_id": "MOVI-DE-POSE-001-PHASE6-SYSTEMS-v1.0.0",
        "lock_date": "2026-08-25", "status": "locked", "seed": config["seed"],
        "systems": sorted(config["systems"]), "noise_condition_count": len(noise_conditions(config)),
        "checks": checks,
        "artifacts": {
            str(path.relative_to(root)): {"sha256": sha256(path)} for path in artifact_paths
        },
        "change_control": "Any change to feature membership, pose treatment, perturbation grid, fit policy, tuning, or thresholds requires a prospective amendment and replacement lock before confirmatory model fitting.",
    }
    args.lock_output.parent.mkdir(parents=True, exist_ok=True)
    args.lock_output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Validated and locked {len(config['systems'])} Phase 6 system definitions and {len(noise_conditions(config))} noise conditions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
