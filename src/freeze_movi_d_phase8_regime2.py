#!/usr/bin/env python3
"""Independently verify and checksum-lock Phase 8 MOVi-D regime 2."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


SYSTEMS = {"A_rgb", "B_rgb_2d", "C_camera_geometry", "D_pose_aligned_geometry", "G_camera_geometry_only", "G_pose_aligned_geometry_only", "P_pose_only", "S_shuffled_pose"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--repository-root", type=Path, default=Path(".")); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    root = args.repository_root.resolve(); run = root / "runs/movi_de_confirmatory/phase8_regime2"; features = run / "features"; output = run / "in_domain_movi_d"
    pair_rows = rows(root / "manifests/pairs/movi_de/movi_d_all_pairs.jsonl"); pair_by_id = {row["pair_id"]: row for row in pair_rows}
    predictions = rows(output / "movi_d_phase8_regime2_predictions.jsonl")
    result = json.loads((output / "movi_d_phase8_regime2_results.json").read_text()); locked = json.loads((output / "movi_d_phase8_regime2_locked_config.json").read_text())
    feature_manifest = json.loads((features / "movi_d_phase8_feature_manifest.json").read_text()); output_manifest = json.loads((output / "movi_d_phase8_regime2_manifest.json").read_text())
    selected = rows(root / "manifests/movi_de/confirmatory_movi_d_150.jsonl"); split_by_video = {str(row["video_id"]): row["split"] for row in selected}
    falsification = result["fixed_camera_falsification"]
    checks = {
        "exact_90_30_30_video_split": len(split_by_video) == 150 and Counter(split_by_video.values()) == Counter({"train": 90, "dev": 30, "test": 30}),
        "exact_6000_2000_2000_pair_split": Counter(row["split"] for row in pair_rows) == Counter({"train": 6000, "dev": 2000, "test": 2000}),
        "pair_videos_match_locked_split": all(split_by_video.get(str(row["video_id"])) == row["split"] for row in pair_rows),
        "exact_10000_unique_predictions": len(predictions) == 10000 and len({row["pair_id"] for row in predictions}) == 10000,
        "prediction_pair_ids_exact": {row["pair_id"] for row in predictions} == set(pair_by_id),
        "prediction_identity_fields_unchanged": all(row["label"] == pair_by_id[row["pair_id"]]["label"] and str(row["video_id"]) == str(pair_by_id[row["pair_id"]]["video_id"]) and row["split"] == pair_by_id[row["pair_id"]]["split"] for row in predictions),
        "exact_system_scores": all(set(row["scores"]) == SYSTEMS for row in predictions),
        "all_pair_camera_motion_controls_zero": all(row["controls"]["camera_displacement_scene_units"] == 0 and row["controls"]["relative_camera_rotation_degrees"] == 0 and row["controls"]["normalized_camera_displacement"] == 0 for row in predictions),
        "feature_checks_pass": all(feature_manifest["checks"].values()), "output_checks_pass": all(output_manifest["checks"].values()), "result_checks_pass": all(bool(value) for value in result["checks"].values()),
        "locked_system_set": set(locked["systems"]) == SYSTEMS,
        "all_models_train_only": all(row["fit_scope"] == "MOVi-D training pairs only" and row["standardizer_scope"] == "MOVi-D training pairs only" for row in locked["systems"].values()),
        "fixed_camera_D_minus_C_interval_spans_zero": falsification["paired_video_cluster_ci_low"] <= 0 <= falsification["paired_video_cluster_ci_high"],
        "pose_only_exact_chance": result["aggregate"]["P_pose_only"]["test"]["auroc"] == 0.5,
        "clean_D_phase7_reproduced": all(result["clean_D_phase7_reproduction"].values()),
    }
    paths = [
        root / "configs/movi_de_phase6_systems.json", root / "docs/MOVI_DE_PHASE8_REGIME2_IN_DOMAIN_MOVI_D.md",
        root / "src/build_movi_d_phase8_in_domain_features.py", root / "src/run_movi_d_phase8_in_domain.py", root / "src/freeze_movi_d_phase8_regime2.py", root / "tests/test_movi_d_phase8_in_domain.py",
        root / "manifests/movi_de/phase5_pair_manifest_freeze.json", root / "manifests/movi_de/phase6_system_configuration_freeze.json", root / "manifests/movi_de/phase7_pose_noise_study_freeze.json", root / "manifests/movi_de/phase8_regime2_reproducibility_audit.json",
        features / "movi_d_phase8_in_domain_features.npz", features / "movi_d_phase8_shuffled_pose_assignment.jsonl", features / "movi_d_phase8_feature_manifest.json",
        output / "movi_d_phase8_regime2_locked_config.json", output / "movi_d_phase8_regime2_predictions.jsonl", output / "movi_d_phase8_regime2_models.joblib", output / "movi_d_phase8_regime2_results.json", output / "movi_d_phase8_regime2_results.csv", output / "movi_d_phase8_regime2_manifest.json",
        root / "results/movi_de_phase8_regime2/movi_d_in_domain_results.json", root / "results/movi_de_phase8_regime2/movi_d_in_domain_results.csv", root / "results/movi_de_phase8_regime2/movi_d_in_domain_locked_config.json",
    ]
    artifacts = {str(path.relative_to(root)): {"sha256": sha256(path), "size_bytes": path.stat().st_size} for path in paths}
    status = "locked" if all(checks.values()) else "failed"
    lock = {"lock_id": "MOVI-DE-POSE-001-PHASE8-REGIME2-v1.0.0", "lock_date": "2026-08-25", "status": status, "seed": 20260825, "dataset": "movi_d", "regime": "in_domain_video_disjoint_fixed_camera", "systems": sorted(SYSTEMS), "bootstrap_replicates": 10000, "fixed_camera_falsification": falsification, "checks": checks, "artifacts": dict(sorted(artifacts.items())), "change_control": "Regime 2 is frozen. Post-test changes to pools, features, shuffle, fitting, thresholds, scoring, or inference are exploratory and require a replacement lock."}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if status != "locked": raise RuntimeError(f"Freeze failed: {[key for key, value in checks.items() if not value]}")
    print(f"Locked Phase 8 regime 2 with {len(artifacts)} checksummed artifacts")
    return 0


if __name__ == "__main__": raise SystemExit(main())
