#!/usr/bin/env python3
"""Verify and checksum-lock the completed MOVi-D/E Phase 7 pose-noise study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    run = root / "runs/movi_de_confirmatory/phase7"
    checks: dict[str, bool] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    for dataset in ("movi_d", "movi_e"):
        pair_path = root / f"manifests/pairs/movi_de/{dataset}_all_pairs.jsonl"
        pair_rows = rows(pair_path)
        pair_by_id = {str(row["pair_id"]): row for row in pair_rows}
        output_dir = run / f"{dataset}_pose_noise"
        result_path = output_dir / f"{dataset}_phase7_pose_noise_results.json"
        prediction_path = output_dir / f"{dataset}_phase7_pose_noise_predictions.jsonl"
        manifest_path = output_dir / f"{dataset}_phase7_pose_noise_manifest.json"
        result = json.loads(result_path.read_text())
        predictions = rows(prediction_path)
        condition_ids = [row["condition_id"] for row in result["conditions"]]
        prefix = dataset
        checks[f"{prefix}_exact_36_conditions"] = len(condition_ids) == 36 and len(set(condition_ids)) == 36
        checks[f"{prefix}_all_internal_result_checks_pass"] = all(result["checks"].values())
        checks[f"{prefix}_exact_10000_predictions"] = len(predictions) == 10000 and len({row["pair_id"] for row in predictions}) == 10000
        checks[f"{prefix}_prediction_pair_ids_exact"] = {row["pair_id"] for row in predictions} == set(pair_by_id)
        checks[f"{prefix}_pair_membership_labels_and_videos_unchanged"] = all(
            row["label"] == pair_by_id[row["pair_id"]]["label"]
            and str(row["video_id"]) == str(pair_by_id[row["pair_id"]]["video_id"])
            and row["split"] == pair_by_id[row["pair_id"]]["split"]
            for row in predictions
        )
        checks[f"{prefix}_all_condition_scores_present"] = all(set(row["scores"]) == set(condition_ids) for row in predictions)
        checks[f"{prefix}_zero_noise_checks"] = result["checks"]["zero_noise_features_byte_identical_to_clean_D"] and result["checks"]["zero_noise_scores_byte_identical_to_clean_D"]
        checks[f"{prefix}_clean_model_not_refit"] = result["checks"]["clean_model_not_refit_for_noise"]
        checks[f"{prefix}_output_manifest_checks_pass"] = all(json.loads(manifest_path.read_text())["checks"].values())
        paths = [
            run / f"{dataset}_phase1/phase1_adapter_manifest.json",
            run / f"{dataset}_pair_observations/phase7_observation_filter_manifest.json",
            run / f"{dataset}_rgb_embeddings/rgb_embedding_manifest.json",
            output_dir / f"{dataset}_phase7_clean_D_locked_config.json",
            output_dir / f"{dataset}_phase7_clean_D_model.joblib",
            output_dir / f"{dataset}_phase7_pose_noise_features.npz",
            prediction_path, result_path,
            output_dir / f"{dataset}_phase7_pose_noise_results.csv", manifest_path,
        ]
        for path in paths:
            artifacts[str(path.relative_to(root))] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
    summary_dir = root / "results/movi_de_phase7_pose_noise"
    summary_manifest = json.loads((summary_dir / "phase7_pose_noise_summary_manifest.json").read_text())
    checks["summary_checks_pass"] = all(summary_manifest["checks"].values())
    support = [
        root / "configs/movi_de_phase6_systems.json",
        root / "docs/MOVI_DE_PHASE7_POSE_NOISE_STUDY.md",
        root / "src/prepare_movi_de_phase7_observations.py",
        root / "src/run_movi_de_phase7_pose_noise.py",
        root / "src/summarize_movi_de_phase7_pose_noise.py",
        root / "src/freeze_movi_de_phase7_pose_noise.py",
        root / "tests/test_movi_de_phase7_pose_noise.py",
        root / "manifests/movi_de/phase5_pair_manifest_freeze.json",
        root / "manifests/movi_de/phase6_system_configuration_freeze.json",
        root / "manifests/movi_de/phase7_reproducibility_audit.json",
        summary_dir / "phase7_pose_noise_combined_results.json",
        summary_dir / "phase7_pose_noise_results_table.csv",
        summary_dir / "phase7_pose_noise_summary_manifest.json",
    ]
    for path in support:
        artifacts[str(path.relative_to(root))] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
    status = "locked" if all(checks.values()) else "failed"
    lock = {
        "lock_id": "MOVI-DE-POSE-001-PHASE7-NOISE-v1.0.0", "lock_date": "2026-08-25",
        "status": status, "seed": 20260825, "datasets": ["movi_d", "movi_e"],
        "conditions_per_dataset": 36, "bootstrap_replicates": 10000,
        "checks": checks, "artifacts": dict(sorted(artifacts.items())),
        "change_control": "Phase 7 is frozen. Any change to perturbations, sufficient-statistic propagation, pair membership, clean model fitting, thresholds, scoring, or bootstrap analysis requires a prospective amendment and replacement lock.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if status != "locked":
        raise RuntimeError(f"Phase 7 freeze failed: {[name for name, passed in checks.items() if not passed]}")
    print(f"Locked Phase 7 with {len(artifacts)} checksummed artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
