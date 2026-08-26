#!/usr/bin/env python3
"""Independently verify and checksum-lock Phase 8 regime 3 D-to-E transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np


SYSTEM_ID = "D_pose_aligned_geometry"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    output = root / "runs/movi_de_confirmatory/phase8_regime3/d_to_e_transfer"

    config = json.loads((root / "configs/movi_de_phase6_systems.json").read_text(encoding="utf-8"))
    source_lock_path = root / "runs/movi_de_confirmatory/phase8_regime2/in_domain_movi_d/movi_d_phase8_regime2_locked_config.json"
    source_model_path = root / "runs/movi_de_confirmatory/phase8_regime2/in_domain_movi_d/movi_d_phase8_regime2_models.joblib"
    source_freeze = json.loads((root / "manifests/movi_de/phase8_regime2_in_domain_movi_d_freeze.json").read_text(encoding="utf-8"))
    source_lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    source_bundle = joblib.load(source_model_path)
    transfer_lock = json.loads((output / "movi_d_to_e_phase8_regime3_locked_config.json").read_text(encoding="utf-8"))
    results = json.loads((output / "movi_d_to_e_phase8_regime3_results.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "movi_d_to_e_phase8_regime3_manifest.json").read_text(encoding="utf-8"))
    predictions = read_jsonl(output / "movi_d_to_e_phase8_regime3_predictions.jsonl")
    pairs = [row for row in read_jsonl(root / "manifests/pairs/movi_de/movi_e_all_pairs.jsonl") if row["split"] == "test"]
    features = np.load(root / "runs/movi_de_confirmatory/phase8_regime1/features/movi_e_phase8_in_domain_features.npz", allow_pickle=False)
    test_mask = features["splits"].astype(str) == "test"
    recomputed = source_bundle["models"][SYSTEM_ID].predict_proba(
        features[SYSTEM_ID][test_mask].astype(np.float64)
    )[:, 1]
    recorded = np.asarray([row["transfer_score"] for row in predictions], dtype=np.float64)
    source_system = source_lock["systems"][SYSTEM_ID]
    in_domain_gap = results["paired_transfer_differences"]["in_domain_D"]["auroc"]
    c_gap = results["paired_transfer_differences"]["in_domain_C"]["auroc"]

    checks = {
        "source_regime_status_locked": source_freeze["status"] == "locked" and all(source_freeze["checks"].values()),
        "protocol_declares_D_to_E_without_refit": config["transfer"]["refit_on_movi_e"] is False and "MOVi-D clean system D" in config["transfer"]["source"],
        "source_model_bundle_matches_source_lock": source_bundle["locked"][SYSTEM_ID] == source_system,
        "transfer_uses_clean_D_only": transfer_lock["system"] == SYSTEM_ID and results["system"] == SYSTEM_ID,
        "source_fit_and_scaler_scopes_are_MOVi_D_train_only": source_system["fit_scope"] == "MOVi-D training pairs only" and source_system["standardizer_scope"] == "MOVi-D training pairs only",
        "source_regularization_unchanged": transfer_lock["selected_C"] == source_system["selected_C"],
        "source_thresholds_unchanged": transfer_lock["recall_90_threshold"] == source_system["recall_90_threshold"] and transfer_lock["max_f1_threshold"] == source_system["max_f1_threshold"],
        "exact_2000_unique_test_predictions": len(predictions) == 2000 and len({row["pair_id"] for row in predictions}) == 2000,
        "prediction_pair_order_exact": [row["pair_id"] for row in predictions] == [row["pair_id"] for row in pairs],
        "prediction_identity_fields_exact": all(pred["label"] == pair["label"] and pred["video_id"] == str(pair["video_id"]) and pred["split"] == "test" for pred, pair in zip(predictions, pairs, strict=True)),
        "all_recorded_scores_finite": bool(np.isfinite(recorded).all()),
        "all_2000_scores_recompute_exactly": bool(np.array_equal(recorded, recomputed)),
        "all_result_checks_pass": all(bool(value) for value in results["checks"].values()),
        "all_manifest_checks_pass": all(bool(value) for value in manifest["checks"].values()),
        "paired_in_domain_D_AUROC_interval_spans_zero": in_domain_gap["paired_video_cluster_ci_low"] <= 0 <= in_domain_gap["paired_video_cluster_ci_high"],
        "paired_in_domain_C_AUROC_interval_above_zero": c_gap["paired_video_cluster_ci_low"] > 0,
        "exact_10000_bootstrap_replicates": results["checks"]["bootstrap_replicates"] == 10000,
    }

    paths = [
        root / "configs/movi_de_phase6_systems.json",
        root / "docs/MOVI_DE_PHASE8_REGIME3_D_TO_E_TRANSFER.md",
        root / "src/run_movi_d_to_e_phase8_transfer.py",
        root / "src/freeze_movi_d_to_e_phase8_regime3.py",
        root / "tests/test_movi_d_to_e_phase8_transfer.py",
        root / "manifests/movi_de/phase8_regime1_in_domain_movi_e_freeze.json",
        root / "manifests/movi_de/phase8_regime2_in_domain_movi_d_freeze.json",
        root / "manifests/movi_de/phase8_regime3_reproducibility_audit.json",
        source_model_path,
        output / "movi_d_to_e_phase8_regime3_locked_config.json",
        output / "movi_d_to_e_phase8_regime3_predictions.jsonl",
        output / "movi_d_to_e_phase8_regime3_results.json",
        output / "movi_d_to_e_phase8_regime3_results.csv",
        output / "movi_d_to_e_phase8_regime3_manifest.json",
        root / "results/movi_de_phase8_regime3/movi_d_to_e_transfer_results.json",
        root / "results/movi_de_phase8_regime3/movi_d_to_e_transfer_results.csv",
        root / "results/movi_de_phase8_regime3/movi_d_to_e_transfer_locked_config.json",
    ]
    artifacts = {
        str(path.relative_to(root)): {"sha256": sha256(path), "size_bytes": path.stat().st_size}
        for path in paths
    }
    status = "locked" if all(checks.values()) else "failed"
    lock = {
        "lock_id": "MOVI-DE-POSE-001-PHASE8-REGIME3-v1.0.0",
        "lock_date": "2026-08-25",
        "status": status,
        "seed": 20260825,
        "source_dataset": "movi_d",
        "target_dataset": "movi_e",
        "regime": "clean_D_transfer_without_refit",
        "target_test_pairs": 2000,
        "bootstrap_replicates": 10000,
        "headline": {
            "transfer_AUROC": results["transfer_test"]["auroc"],
            "transfer_PR_AUC": results["transfer_test"]["pr_auc"],
            "transfer_minus_in_domain_D_AUROC": in_domain_gap["transfer_minus_reference"],
            "paired_video_cluster_ci_low": in_domain_gap["paired_video_cluster_ci_low"],
            "paired_video_cluster_ci_high": in_domain_gap["paired_video_cluster_ci_high"],
        },
        "checks": checks,
        "artifacts": dict(sorted(artifacts.items())),
        "change_control": "Regime 3 is frozen. Any source refit, target normalization, target tuning, threshold adjustment, pair change, or post-test feature change requires a replacement lock and is exploratory.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if status != "locked":
        raise RuntimeError(f"Freeze failed: {[name for name, passed in checks.items() if not passed]}")
    print(f"Locked Phase 8 regime 3 with {len(artifacts)} checksummed artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
