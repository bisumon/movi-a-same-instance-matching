#!/usr/bin/env python3
"""Train, tune, lock, and test Phase 8 regime 2 on video-disjoint MOVi-D pools."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_movi_de_phase7_pose_noise import choose_f1_threshold, choose_recall_threshold, stable
from run_movi_e_phase8_in_domain import SYSTEM_ORDER, evaluate, paired_rate_interval, read_jsonl, write_json
from summarize_movi_de_phase7_pose_noise import paired_cluster_interval
from validate_movi_de_phase6_systems import resolve_features, validate_config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_comparisons(system_order, reference, scores, locked, labels, mask, videos, seed, replicates):
    result = {}
    ref_predictions = scores[reference][mask] >= float(locked[reference]["recall_90_threshold"])
    for number, system_id in enumerate(system_order):
        auc = paired_cluster_interval(labels[mask], scores[reference][mask], scores[system_id][mask], videos, seed, replicates)
        predictions = scores[system_id][mask] >= float(locked[system_id]["recall_90_threshold"])
        result[system_id] = {
            "reference": reference,
            "auroc": {
                "system_minus_reference": auc["noise_minus_clean_auroc"],
                "paired_video_cluster_ci_low": auc["paired_video_cluster_ci_low"],
                "paired_video_cluster_ci_high": auc["paired_video_cluster_ci_high"],
                "bootstrap_replicates": replicates,
            },
            "false_match_rate": paired_rate_interval(labels[mask], ref_predictions, predictions, videos, 0, seed + 100 + number, replicates),
            "recall": paired_rate_interval(labels[mask], ref_predictions, predictions, videos, 1, seed + 200 + number, replicates),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--system-config", type=Path, required=True)
    parser.add_argument("--phase7-d-lock", type=Path, required=True)
    parser.add_argument("--phase7-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.system_config.read_text(encoding="utf-8"))
    if not all(validate_config(config).values()): raise ValueError("Invalid Phase 6 configuration")
    seed = int(config["seed"])
    data = np.load(args.features, allow_pickle=False)
    pair_ids, splits, labels = data["pair_ids"].astype(str), data["splits"].astype(str), data["labels"].astype(np.int8)
    pairs = read_jsonl(args.pairs)
    if pair_ids.tolist() != [str(row["pair_id"]) for row in pairs]: raise ValueError("Feature/pair order mismatch")
    matrices = {system_id: data[system_id].astype(np.float64) for system_id in SYSTEM_ORDER}
    masks = {split: splits == split for split in ("train", "dev", "test")}
    if {split: int(value.sum()) for split, value in masks.items()} != {"train": 6000, "dev": 2000, "test": 2000}: raise ValueError("Locked split count mismatch")

    fitted, locked = {}, {}
    c_grid = [float(value) for value in config["shared_training"]["regularization_grid_C"]]
    for system_id in SYSTEM_ORDER:
        matrix = matrices[system_id]
        if matrix.shape != (10000, len(resolve_features(config, system_id))) or not np.isfinite(matrix).all(): raise ValueError(f"Invalid {system_id} matrix")
        train_x, train_y = matrix[masks["train"]], labels[masks["train"]]
        dev_x, dev_y = matrix[masks["dev"]], labels[masks["dev"]]
        best, tuning = None, []
        for c_value in c_grid:
            model = Pipeline([("standardizer", StandardScaler()), ("logistic_regression", LogisticRegression(C=c_value, solver="lbfgs", max_iter=5000, random_state=seed))])
            model.fit(train_x, train_y); dev_scores = model.predict_proba(dev_x)[:, 1]
            auroc = float(roc_auc_score(dev_y, dev_scores)); tuning.append({"C": c_value, "dev_auroc": stable(auroc)})
            candidate = (auroc, -c_value, model, dev_scores)
            if best is None or candidate[:2] > best[:2]: best = candidate
        assert best is not None
        model, dev_scores = best[2], best[3]
        recall_threshold, achieved_recall, dev_fmr = choose_recall_threshold(dev_y, dev_scores)
        f1_threshold, dev_f1 = choose_f1_threshold(dev_y, dev_scores)
        fitted[system_id] = model
        locked[system_id] = {
            "features": resolve_features(config, system_id), "selected_C": float(model.named_steps["logistic_regression"].C),
            "C_tuning": tuning, "selection_metric": "development AUROC; ties choose smaller C",
            "recall_90_threshold": recall_threshold, "development_achieved_recall": stable(achieved_recall),
            "development_false_match_rate": stable(dev_fmr), "max_f1_threshold": f1_threshold,
            "development_max_f1": stable(dev_f1), "fit_scope": "MOVi-D training pairs only", "standardizer_scope": "MOVi-D training pairs only",
        }
        print(f"locked {system_id}: C={locked[system_id]['selected_C']:g} dev_AUROC={best[0]:.6f}", flush=True)
    lock_path = args.output_dir / "movi_d_phase8_regime2_locked_config.json"
    write_json(lock_path, {
        "pipeline": "MOVi-D Phase 8 regime 2 lock written before test scoring", "version": "1.0.0",
        "dataset": "movi_d", "regime": "in_domain_video_disjoint_fixed_camera", "seed": seed, "systems": locked,
        "decision_protocol": {"normalization_and_fit": "train only", "regularization_and_thresholds": "development only", "test_policy": "test evaluated only after this lock was written"},
        "inputs": {"features_sha256": sha256(args.features), "pairs_sha256": sha256(args.pairs), "system_config_sha256": sha256(args.system_config)},
    })

    scores, aggregate, latency = {}, {}, {}
    for system_id in SYSTEM_ORDER:
        matrix, model = matrices[system_id], fitted[system_id]
        scores[system_id] = model.predict_proba(matrix)[:, 1]
        aggregate[system_id] = {split: evaluate(labels[masks[split]], scores[system_id][masks[split]], float(locked[system_id]["recall_90_threshold"]), float(locked[system_id]["max_f1_threshold"])) for split in ("dev", "test")}
        repeats, test_x = [], matrix[masks["test"]]
        for _ in range(50):
            started = time.perf_counter_ns(); model.predict_proba(test_x); repeats.append((time.perf_counter_ns() - started) / len(test_x) / 1000.0)
        latency[system_id] = {"microseconds_per_pair_p50": stable(np.quantile(repeats, 0.5)), "microseconds_per_pair_p95": stable(np.quantile(repeats, 0.95)), "batch_size": len(test_x), "repetitions": 50}
    test_videos = np.asarray([str(row["video_id"]) for row in pairs])[masks["test"]]
    paired_vs_c = paired_comparisons(SYSTEM_ORDER, "C_camera_geometry", scores, locked, labels, masks["test"], test_videos, seed, args.bootstrap_replicates)
    paired_vs_d = paired_comparisons(SYSTEM_ORDER, "D_pose_aligned_geometry", scores, locked, labels, masks["test"], test_videos, seed, args.bootstrap_replicates)
    falsification = paired_vs_c["D_pose_aligned_geometry"]["auroc"]

    phase7_lock = json.loads(args.phase7_d_lock.read_text())
    phase7_zero = {row["pair_id"]: row["scores"]["N_t0_r0"] for row in read_jsonl(args.phase7_predictions)}
    d_scores = {pair_id: stable(score) for pair_id, score in zip(pair_ids, scores["D_pose_aligned_geometry"], strict=True)}
    reproduction = {
        "selected_C_matches_phase7": locked["D_pose_aligned_geometry"]["selected_C"] == phase7_lock["selected_C"],
        "recall_threshold_matches_phase7": locked["D_pose_aligned_geometry"]["recall_90_threshold"] == phase7_lock["clean_D_90_recall_threshold"],
        "f1_threshold_matches_phase7": locked["D_pose_aligned_geometry"]["max_f1_threshold"] == phase7_lock["clean_D_max_f1_threshold"],
        "all_10000_scores_match_phase7_zero_noise": d_scores == phase7_zero,
    }
    if not all(reproduction.values()): raise RuntimeError(f"Phase 7 D reproduction failed: {reproduction}")

    prediction_path = args.output_dir / "movi_d_phase8_regime2_predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for number, pair in enumerate(pairs):
            handle.write(json.dumps({
                "dataset": "movi_d", "regime": "in_domain_video_disjoint_fixed_camera", "pair_id": str(pair_ids[number]),
                "split": str(splits[number]), "label": int(labels[number]), "video_id": str(pair["video_id"]),
                "negative_difficulty": pair["negative_difficulty"], "temporal_gap": pair["temporal_gap"], "temporal_gap_bin": pair["temporal_gap_bin"],
                "controls": pair["controls"], "scores": {system_id: stable(scores[system_id][number]) for system_id in SYSTEM_ORDER},
            }, sort_keys=True, separators=(",", ":")) + "\n")
    model_path = args.output_dir / "movi_d_phase8_regime2_models.joblib"; joblib.dump({"models": fitted, "system_order": SYSTEM_ORDER, "locked": locked}, model_path, compress=3)
    results_path = args.output_dir / "movi_d_phase8_regime2_results.json"
    write_json(results_path, {
        "pipeline": "MOVi-D Phase 8 regime 2 fixed-camera in-domain evaluation", "version": "1.0.0",
        "dataset": "movi_d", "regime": "train_dev_test_on_video_disjoint_MOVi-D_pools",
        "aggregate": aggregate, "paired_differences_vs_C": paired_vs_c, "paired_differences_vs_D": paired_vs_d,
        "fixed_camera_falsification": {
            "estimand": "AUROC_D_minus_C_on_identical_MOVi_D_test_pairs", "D_minus_C": falsification["system_minus_reference"],
            "paired_video_cluster_ci_low": falsification["paired_video_cluster_ci_low"], "paired_video_cluster_ci_high": falsification["paired_video_cluster_ci_high"],
            "interpretation_rule": "descriptive fixed-camera falsification; no equivalence margin was predeclared",
        },
        "scoring_latency": latency, "clean_D_phase7_reproduction": reproduction,
        "checks": {"exact_video_disjoint_pair_counts": True, "all_models_fit_train_only": True, "all_selection_and_thresholds_dev_only": True, "test_scored_after_lock": True, "all_scores_finite": all(np.isfinite(x).all() for x in scores.values()), "clean_D_reproduces_phase7": all(reproduction.values()), "all_pose_only_inputs_structural_zero": bool(np.array_equal(matrices["P_pose_only"], np.zeros_like(matrices["P_pose_only"]))), "bootstrap_replicates": args.bootstrap_replicates},
    })
    table_path = args.output_dir / "movi_d_phase8_regime2_results.csv"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["system", "split", "selected_C", "auroc", "pr_auc", "false_match_rate", "recall", "precision_at_max_f1", "recall_at_max_f1", "f1", "brier_score", "log_loss", "ece_10_bins"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for system_id in SYSTEM_ORDER:
            for split in ("dev", "test"):
                value, cal = aggregate[system_id][split], aggregate[system_id][split]["calibration"]
                writer.writerow({"system": system_id, "split": split, "selected_C": locked[system_id]["selected_C"], "auroc": value["auroc"], "pr_auc": value["pr_auc"], "false_match_rate": value["false_match_rate_at_locked_90_recall_threshold"], "recall": value["recall_at_locked_90_recall_threshold"], "precision_at_max_f1": value["precision_at_locked_max_f1_threshold"], "recall_at_max_f1": value["recall_at_locked_max_f1_threshold"], "f1": value["f1_at_locked_max_f1_threshold"], "brier_score": cal["brier_score"], "log_loss": cal["log_loss"], "ece_10_bins": cal["expected_calibration_error_10_bins"]})
    manifest_path = args.output_dir / "movi_d_phase8_regime2_manifest.json"
    output_paths = [lock_path, prediction_path, model_path, results_path, table_path]
    write_json(manifest_path, {
        "pipeline": "MOVi-D Phase 8 regime 2 output manifest", "version": "1.0.0", "status": "pass",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
        "inputs": {name: sha256(path) for name, path in {"features": args.features, "feature_manifest": args.feature_manifest, "pairs": args.pairs, "system_config": args.system_config, "phase7_D_lock": args.phase7_d_lock, "phase7_predictions": args.phase7_predictions}.items()},
        "counts": {"systems": len(SYSTEM_ORDER), "pairs": len(pairs), "train": 6000, "dev": 2000, "test": 2000},
        "checks": {"all_results_checks_pass": True, "fixed_camera_falsification_recorded": True, "clean_D_phase7_reproduction_passed": all(reproduction.values())},
        "outputs": {path.name: sha256(path) for path in output_paths},
    })
    print(f"Complete: MOVi-D regime 2, D-C={falsification['system_minus_reference']:+.6f}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
