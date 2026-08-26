#!/usr/bin/env python3
"""Apply the frozen MOVi-D clean-D system to the locked MOVi-E test pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn

from run_movi_e_phase8_in_domain import evaluate, paired_rate_interval, read_jsonl, write_json
from summarize_movi_de_phase7_pose_noise import paired_cluster_interval
from validate_movi_de_phase6_systems import resolve_features, validate_config


VERSION = "1.0.0"
SYSTEM_ID = "D_pose_aligned_geometry"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def comparison(
    labels: np.ndarray,
    transfer_scores: np.ndarray,
    transfer_threshold: float,
    reference_scores: np.ndarray,
    reference_threshold: float,
    video_ids: np.ndarray,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    auc = paired_cluster_interval(labels, reference_scores, transfer_scores, video_ids, seed, replicates)
    ref_predictions = reference_scores >= reference_threshold
    transfer_predictions = transfer_scores >= transfer_threshold
    return {
        "auroc": {
            "transfer_minus_reference": auc["noise_minus_clean_auroc"],
            "paired_video_cluster_ci_low": auc["paired_video_cluster_ci_low"],
            "paired_video_cluster_ci_high": auc["paired_video_cluster_ci_high"],
            "bootstrap_replicates": replicates,
        },
        "false_match_rate": paired_rate_interval(
            labels, ref_predictions, transfer_predictions, video_ids, 0, seed + 1, replicates
        ),
        "recall": paired_rate_interval(
            labels, ref_predictions, transfer_predictions, video_ids, 1, seed + 2, replicates
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-models", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--source-freeze", type=Path, required=True)
    parser.add_argument("--target-features", type=Path, required=True)
    parser.add_argument("--target-feature-manifest", type=Path, required=True)
    parser.add_argument("--target-pairs", type=Path, required=True)
    parser.add_argument("--system-config", type=Path, required=True)
    parser.add_argument("--target-in-domain-predictions", type=Path, required=True)
    parser.add_argument("--target-in-domain-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(args.system_config.read_text(encoding="utf-8"))
    if not all(validate_config(config).values()):
        raise ValueError("Invalid Phase 6 configuration")
    transfer_spec = config["transfer"]
    if transfer_spec["refit_on_movi_e"] is not False or "MOVi-D clean system D" not in transfer_spec["source"]:
        raise ValueError("Phase 6 transfer rule is not the locked D-to-E no-refit rule")
    seed = int(config["seed"])

    source_lock = json.loads(args.source_lock.read_text(encoding="utf-8"))
    source_freeze = json.loads(args.source_freeze.read_text(encoding="utf-8"))
    source_bundle = joblib.load(args.source_models)
    source_system = source_lock["systems"][SYSTEM_ID]
    expected_features = resolve_features(config, SYSTEM_ID)
    if source_freeze["status"] != "locked" or not all(source_freeze["checks"].values()):
        raise ValueError("MOVi-D source regime is not authoritatively locked")
    if source_system["features"] != expected_features or source_bundle["locked"][SYSTEM_ID] != source_system:
        raise ValueError("Source model feature or lock mismatch")
    model = source_bundle["models"][SYSTEM_ID]
    if int(model.n_features_in_) != len(expected_features):
        raise ValueError("Source model input width mismatch")

    target_manifest = json.loads(args.target_feature_manifest.read_text(encoding="utf-8"))
    declared_target_features = target_manifest["systems"][SYSTEM_ID]["feature_names"]
    if declared_target_features != expected_features:
        raise ValueError("MOVi-E D features do not match the frozen MOVi-D model features")
    target = np.load(args.target_features, allow_pickle=False)
    pair_ids = target["pair_ids"].astype(str)
    splits = target["splits"].astype(str)
    labels = target["labels"].astype(np.int8)
    matrix = target[SYSTEM_ID].astype(np.float64)
    pairs = read_jsonl(args.target_pairs)
    if pair_ids.tolist() != [str(row["pair_id"]) for row in pairs]:
        raise ValueError("Target feature/pair order mismatch")
    if matrix.shape != (10000, len(expected_features)) or not np.isfinite(matrix).all():
        raise ValueError("Invalid MOVi-E clean-D feature matrix")
    test_mask = splits == "test"
    if int(test_mask.sum()) != 2000 or np.bincount(labels[test_mask], minlength=2).tolist() != [1000, 1000]:
        raise ValueError("Locked MOVi-E test count or balance mismatch")

    in_domain_lock = json.loads(args.target_in_domain_lock.read_text(encoding="utf-8"))
    in_domain_rows = [row for row in read_jsonl(args.target_in_domain_predictions) if row["split"] == "test"]
    test_pairs = [row for row, selected in zip(pairs, test_mask, strict=True) if selected]
    test_ids = pair_ids[test_mask]
    test_labels = labels[test_mask]
    test_videos = np.asarray([str(row["video_id"]) for row in test_pairs])
    if [row["pair_id"] for row in in_domain_rows] != test_ids.tolist():
        raise ValueError("MOVi-E in-domain benchmark/test feature order mismatch")
    if [int(row["label"]) for row in in_domain_rows] != test_labels.tolist():
        raise ValueError("MOVi-E benchmark label mismatch")

    # This transfer lock is written before the frozen source model is applied to target test features.
    transfer_lock_path = args.output_dir / "movi_d_to_e_phase8_regime3_locked_config.json"
    transfer_lock = {
        "pipeline": "Phase 8 regime 3 D-to-E transfer lock written before target test scoring",
        "version": VERSION,
        "seed": seed,
        "source_dataset": "movi_d",
        "target_dataset": "movi_e",
        "system": SYSTEM_ID,
        "features": expected_features,
        "selected_C": source_system["selected_C"],
        "recall_90_threshold": source_system["recall_90_threshold"],
        "max_f1_threshold": source_system["max_f1_threshold"],
        "normalization_scope": source_system["standardizer_scope"],
        "fit_scope": source_system["fit_scope"],
        "target_policy": "score locked MOVi-E test pairs only; no MOVi-E refit, renormalization, tuning, or threshold adjustment",
        "inputs": {
            "source_models_sha256": sha256(args.source_models),
            "source_lock_sha256": sha256(args.source_lock),
            "source_freeze_sha256": sha256(args.source_freeze),
            "target_features_sha256": sha256(args.target_features),
            "target_feature_manifest_sha256": sha256(args.target_feature_manifest),
            "target_pairs_sha256": sha256(args.target_pairs),
            "system_config_sha256": sha256(args.system_config),
        },
    }
    write_json(transfer_lock_path, transfer_lock)

    started = time.perf_counter_ns()
    transfer_scores = model.predict_proba(matrix[test_mask])[:, 1]
    elapsed_us_per_pair = (time.perf_counter_ns() - started) / len(transfer_scores) / 1000.0
    if not np.isfinite(transfer_scores).all():
        raise ValueError("Non-finite transfer scores")
    transfer_metrics = evaluate(
        test_labels,
        transfer_scores,
        float(source_system["recall_90_threshold"]),
        float(source_system["max_f1_threshold"]),
    )

    benchmark_scores = {
        "in_domain_D": np.asarray([row["scores"][SYSTEM_ID] for row in in_domain_rows], dtype=np.float64),
        "in_domain_C": np.asarray([row["scores"]["C_camera_geometry"] for row in in_domain_rows], dtype=np.float64),
    }
    benchmark_thresholds = {
        "in_domain_D": float(in_domain_lock["systems"][SYSTEM_ID]["recall_90_threshold"]),
        "in_domain_C": float(in_domain_lock["systems"]["C_camera_geometry"]["recall_90_threshold"]),
    }
    benchmark_metrics = {
        "in_domain_D": evaluate(
            test_labels, benchmark_scores["in_domain_D"], benchmark_thresholds["in_domain_D"],
            float(in_domain_lock["systems"][SYSTEM_ID]["max_f1_threshold"]),
        ),
        "in_domain_C": evaluate(
            test_labels, benchmark_scores["in_domain_C"], benchmark_thresholds["in_domain_C"],
            float(in_domain_lock["systems"]["C_camera_geometry"]["max_f1_threshold"]),
        ),
    }
    comparisons = {
        name: comparison(
            test_labels, transfer_scores, float(source_system["recall_90_threshold"]), scores,
            benchmark_thresholds[name], test_videos, seed + 100 * number, args.bootstrap_replicates,
        )
        for number, (name, scores) in enumerate(benchmark_scores.items(), start=1)
    }

    predictions_path = args.output_dir / "movi_d_to_e_phase8_regime3_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for number, pair in enumerate(test_pairs):
            handle.write(json.dumps({
                "dataset": "movi_e", "source_dataset": "movi_d", "regime": "D_to_E_no_refit",
                "pair_id": str(test_ids[number]), "split": "test", "label": int(test_labels[number]),
                "video_id": str(pair["video_id"]), "negative_difficulty": pair["negative_difficulty"],
                "temporal_gap": pair["temporal_gap"], "temporal_gap_bin": pair["temporal_gap_bin"],
                "controls": pair["controls"], "transfer_score": float(transfer_scores[number]),
                "source_recall_90_threshold": float(source_system["recall_90_threshold"]),
                "source_max_f1_threshold": float(source_system["max_f1_threshold"]),
            }, sort_keys=True, separators=(",", ":")) + "\n")

    results_path = args.output_dir / "movi_d_to_e_phase8_regime3_results.json"
    results = {
        "pipeline": "Phase 8 regime 3 MOVi-D-to-MOVi-E clean-D transfer",
        "version": VERSION,
        "source_dataset": "movi_d",
        "target_dataset": "movi_e",
        "system": SYSTEM_ID,
        "transfer_test": transfer_metrics,
        "in_domain_MOVi_E_benchmarks": benchmark_metrics,
        "paired_transfer_differences": comparisons,
        "threshold_provenance": {
            "selected_C": "MOVi-D development",
            "recall_90_threshold": "MOVi-D development",
            "max_f1_threshold": "MOVi-D development",
            "MOVi-E_adjustment": "none",
        },
        "scoring_latency": {
            "microseconds_per_pair_single_pass": elapsed_us_per_pair,
            "batch_size": len(transfer_scores),
        },
        "checks": {
            "source_regime_authoritatively_locked": True,
            "exact_feature_name_and_order_match": True,
            "source_standardizer_applied_unchanged": True,
            "source_model_applied_unchanged": True,
            "source_regularization_applied_unchanged": True,
            "source_thresholds_applied_unchanged": True,
            "no_MOVi_E_refit_or_renormalization": True,
            "only_locked_MOVi_E_test_pairs_scored": True,
            "exact_2000_unique_test_predictions": len(set(test_ids.tolist())) == 2000,
            "all_scores_finite": True,
            "bootstrap_replicates": args.bootstrap_replicates,
        },
    }
    write_json(results_path, results)

    table_path = args.output_dir / "movi_d_to_e_phase8_regime3_results.csv"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["evaluation", "auroc", "pr_auc", "false_match_rate", "recall", "precision_at_max_f1", "recall_at_max_f1", "f1", "brier_score", "log_loss", "ece_10_bins"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, value in [("D_to_E_transfer", transfer_metrics), *benchmark_metrics.items()]:
            calibration = value["calibration"]
            writer.writerow({
                "evaluation": name, "auroc": value["auroc"], "pr_auc": value["pr_auc"],
                "false_match_rate": value["false_match_rate_at_locked_90_recall_threshold"],
                "recall": value["recall_at_locked_90_recall_threshold"],
                "precision_at_max_f1": value["precision_at_locked_max_f1_threshold"],
                "recall_at_max_f1": value["recall_at_locked_max_f1_threshold"],
                "f1": value["f1_at_locked_max_f1_threshold"], "brier_score": calibration["brier_score"],
                "log_loss": calibration["log_loss"], "ece_10_bins": calibration["expected_calibration_error_10_bins"],
            })

    manifest_path = args.output_dir / "movi_d_to_e_phase8_regime3_manifest.json"
    output_paths = [transfer_lock_path, predictions_path, results_path, table_path]
    write_json(manifest_path, {
        "pipeline": "Phase 8 regime 3 output manifest", "version": VERSION, "status": "pass",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
        "counts": {"systems": 1, "target_test_pairs": 2000, "target_test_videos": len(set(test_videos.tolist()))},
        "checks": {"all_result_checks_pass": all(bool(value) for value in results["checks"].values()), "transfer_lock_written_before_scoring": True},
        "outputs": {path.name: sha256(path) for path in output_paths},
    })
    print(f"Complete: D-to-E transfer AUROC={transfer_metrics['auroc']:.6f}, PR-AUC={transfer_metrics['pr_auc']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
