#!/usr/bin/env python3
"""Train, tune, lock, and test Phase 8 regime 1 on video-disjoint MOVi-E pools."""

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score, log_loss, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_movi_de_phase7_pose_noise import choose_f1_threshold, choose_recall_threshold, stable
from summarize_movi_de_phase7_pose_noise import paired_cluster_interval
from validate_movi_de_phase6_systems import resolve_features, validate_config


VERSION = "1.0.0"
SYSTEM_ORDER = (
    "A_rgb", "B_rgb_2d", "C_camera_geometry", "D_pose_aligned_geometry",
    "G_camera_geometry_only", "G_pose_aligned_geometry_only", "P_pose_only", "S_shuffled_pose",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def calibration(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> dict[str, float]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        selected = (scores >= edges[index]) & (scores <= edges[index + 1] if index == bins - 1 else scores < edges[index + 1])
        if selected.any():
            ece += float(selected.mean()) * abs(float(scores[selected].mean()) - float(labels[selected].mean()))
    return {
        "brier_score": stable(brier_score_loss(labels, scores)),
        "log_loss": stable(log_loss(labels, scores, labels=[0, 1])),
        "expected_calibration_error_10_bins": stable(ece),
    }


def evaluate(labels: np.ndarray, scores: np.ndarray, recall_threshold: float, f1_threshold: float) -> dict[str, Any]:
    at_recall, at_f1 = scores >= recall_threshold, scores >= f1_threshold
    negatives, positives = labels == 0, labels == 1
    return {
        "auroc": stable(roc_auc_score(labels, scores)), "pr_auc": stable(average_precision_score(labels, scores)),
        "false_match_rate_at_locked_90_recall_threshold": stable(np.mean(at_recall[negatives])),
        "recall_at_locked_90_recall_threshold": stable(np.mean(at_recall[positives])),
        "precision_at_locked_max_f1_threshold": stable(precision_score(labels, at_f1, zero_division=0)),
        "recall_at_locked_max_f1_threshold": stable(recall_score(labels, at_f1, zero_division=0)),
        "f1_at_locked_max_f1_threshold": stable(f1_score(labels, at_f1, zero_division=0)),
        "calibration": calibration(labels, scores),
    }


def paired_rate_interval(
    labels: np.ndarray, reference_predictions: np.ndarray, comparison_predictions: np.ndarray,
    video_ids: np.ndarray, target_label: int, seed: int, replicates: int = 10000,
) -> dict[str, Any]:
    selected = labels == target_label
    labels_unused = labels[selected]
    del labels_unused
    videos = sorted(set(video_ids.tolist()), key=int)
    mapping = {video: index for index, video in enumerate(videos)}
    clusters = np.asarray([mapping[value] for value in video_ids[selected]], dtype=np.int64)
    reference = reference_predictions[selected].astype(np.float64)
    comparison = comparison_predictions[selected].astype(np.float64)
    counts = np.bincount(clusters, minlength=len(videos))
    ref_sums = np.bincount(clusters, weights=reference, minlength=len(videos))
    cmp_sums = np.bincount(clusters, weights=comparison, minlength=len(videos))
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(videos), np.full(len(videos), 1.0 / len(videos)), size=replicates)
    denominator = weights @ counts
    valid = denominator > 0
    delta = (weights @ (cmp_sums - ref_sums))[valid] / denominator[valid]
    observed = float(comparison.mean() - reference.mean())
    return {
        "comparison_minus_reference": stable(observed),
        "paired_video_cluster_ci_low": stable(np.quantile(delta, 0.025)),
        "paired_video_cluster_ci_high": stable(np.quantile(delta, 0.975)),
        "bootstrap_replicates": replicates, "valid_replicates": int(valid.sum()), "video_clusters": len(videos),
    }


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
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.system_config.read_text(encoding="utf-8"))
    if not all(validate_config(config).values()):
        raise ValueError("Invalid Phase 6 config")
    seed = int(config["seed"])
    data = np.load(args.features, allow_pickle=False)
    pair_ids = data["pair_ids"].astype(str)
    splits = data["splits"].astype(str)
    labels = data["labels"].astype(np.int8)
    pairs = read_jsonl(args.pairs)
    if pair_ids.tolist() != [str(row["pair_id"]) for row in pairs]:
        raise ValueError("Feature and pair order differ")
    matrices = {system_id: data[system_id].astype(np.float64) for system_id in SYSTEM_ORDER}
    masks = {split: splits == split for split in ("train", "dev", "test")}
    if {split: int(mask.sum()) for split, mask in masks.items()} != {"train": 6000, "dev": 2000, "test": 2000}:
        raise ValueError("Not the locked 6000/2000/2000 pair split")

    fitted: dict[str, Pipeline] = {}
    locked: dict[str, Any] = {}
    c_grid = [float(value) for value in config["shared_training"]["regularization_grid_C"]]
    for system_id in SYSTEM_ORDER:
        matrix = matrices[system_id]
        if matrix.shape != (10000, len(resolve_features(config, system_id))) or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid matrix for {system_id}")
        train_x, train_y = matrix[masks["train"]], labels[masks["train"]]
        dev_x, dev_y = matrix[masks["dev"]], labels[masks["dev"]]
        tuning = []
        best = None
        for c_value in c_grid:
            model = Pipeline([
                ("standardizer", StandardScaler()),
                ("logistic_regression", LogisticRegression(C=c_value, solver="lbfgs", max_iter=5000, random_state=seed)),
            ])
            model.fit(train_x, train_y)
            dev_scores = model.predict_proba(dev_x)[:, 1]
            auroc = float(roc_auc_score(dev_y, dev_scores))
            tuning.append({"C": c_value, "dev_auroc": stable(auroc)})
            candidate = (auroc, -c_value, model, dev_scores)
            if best is None or candidate[:2] > best[:2]: best = candidate
        assert best is not None
        selected_model, dev_scores = best[2], best[3]
        recall_threshold, achieved_recall, dev_fmr = choose_recall_threshold(dev_y, dev_scores)
        f1_threshold, dev_f1 = choose_f1_threshold(dev_y, dev_scores)
        fitted[system_id] = selected_model
        locked[system_id] = {
            "features": resolve_features(config, system_id), "selected_C": float(selected_model.named_steps["logistic_regression"].C),
            "C_tuning": tuning, "selection_metric": "development AUROC; ties choose smaller C",
            "recall_90_threshold": recall_threshold, "development_achieved_recall": stable(achieved_recall),
            "development_false_match_rate": stable(dev_fmr), "max_f1_threshold": f1_threshold,
            "development_max_f1": stable(dev_f1), "fit_scope": "MOVi-E training pairs only",
            "standardizer_scope": "MOVi-E training pairs only",
        }
        print(f"locked {system_id}: C={locked[system_id]['selected_C']:g} dev_AUROC={best[0]:.6f}", flush=True)
    lock_path = args.output_dir / "movi_e_phase8_regime1_locked_config.json"
    write_json(lock_path, {
        "pipeline": "MOVi-E Phase 8 regime 1 model and threshold lock written before test scoring",
        "version": VERSION, "dataset": "movi_e", "regime": "in_domain_video_disjoint",
        "seed": seed, "systems": locked,
        "decision_protocol": {
            "normalization_and_fit": "train only", "regularization_and_thresholds": "development only",
            "test_policy": "test labels and scores not evaluated until after this file was written",
        },
        "inputs": {"features_sha256": sha256(args.features), "pairs_sha256": sha256(args.pairs), "system_config_sha256": sha256(args.system_config)},
    })

    # Test scoring begins only after the complete configuration lock is on disk.
    scores = {}
    aggregate = {}
    latency = {}
    for system_id in SYSTEM_ORDER:
        matrix = matrices[system_id]
        model = fitted[system_id]
        scores[system_id] = model.predict_proba(matrix)[:, 1]
        aggregate[system_id] = {}
        for split in ("dev", "test"):
            aggregate[system_id][split] = evaluate(
                labels[masks[split]], scores[system_id][masks[split]],
                float(locked[system_id]["recall_90_threshold"]), float(locked[system_id]["max_f1_threshold"]),
            )
        repetitions = []
        test_x = matrix[masks["test"]]
        for _ in range(50):
            started = time.perf_counter_ns(); model.predict_proba(test_x)
            repetitions.append((time.perf_counter_ns() - started) / len(test_x) / 1000.0)
        latency[system_id] = {
            "microseconds_per_pair_p50": stable(np.quantile(repetitions, 0.5)),
            "microseconds_per_pair_p95": stable(np.quantile(repetitions, 0.95)), "batch_size": len(test_x), "repetitions": 50,
        }

    test_labels = labels[masks["test"]]
    test_videos = np.asarray([str(row["video_id"]) for row in pairs])[masks["test"]]
    reference = "C_camera_geometry"
    paired = {}
    reference_predictions = scores[reference][masks["test"]] >= float(locked[reference]["recall_90_threshold"])
    for number, system_id in enumerate(SYSTEM_ORDER):
        auc = paired_cluster_interval(
            test_labels, scores[reference][masks["test"]], scores[system_id][masks["test"]], test_videos,
            seed, args.bootstrap_replicates,
        )
        predictions = scores[system_id][masks["test"]] >= float(locked[system_id]["recall_90_threshold"])
        paired[system_id] = {
            "reference": reference,
            "auroc": {
                "system_minus_C": auc["noise_minus_clean_auroc"],
                "paired_video_cluster_ci_low": auc["paired_video_cluster_ci_low"],
                "paired_video_cluster_ci_high": auc["paired_video_cluster_ci_high"],
                "bootstrap_replicates": auc["bootstrap_replicates_requested"],
            },
            "false_match_rate": paired_rate_interval(test_labels, reference_predictions, predictions, test_videos, 0, seed + 100 + number, args.bootstrap_replicates),
            "recall": paired_rate_interval(test_labels, reference_predictions, predictions, test_videos, 1, seed + 200 + number, args.bootstrap_replicates),
        }
    primary = paired["D_pose_aligned_geometry"]["auroc"]
    primary_success = primary["paired_video_cluster_ci_low"] > 0
    paired_vs_d = {}
    d_reference = "D_pose_aligned_geometry"
    d_predictions = scores[d_reference][masks["test"]] >= float(locked[d_reference]["recall_90_threshold"])
    for number, system_id in enumerate(SYSTEM_ORDER):
        auc = paired_cluster_interval(
            test_labels, scores[d_reference][masks["test"]], scores[system_id][masks["test"]], test_videos,
            seed, args.bootstrap_replicates,
        )
        predictions = scores[system_id][masks["test"]] >= float(locked[system_id]["recall_90_threshold"])
        paired_vs_d[system_id] = {
            "reference": d_reference,
            "auroc": {
                "system_minus_D": auc["noise_minus_clean_auroc"],
                "paired_video_cluster_ci_low": auc["paired_video_cluster_ci_low"],
                "paired_video_cluster_ci_high": auc["paired_video_cluster_ci_high"],
                "bootstrap_replicates": auc["bootstrap_replicates_requested"],
            },
            "false_match_rate": paired_rate_interval(test_labels, d_predictions, predictions, test_videos, 0, seed + 300 + number, args.bootstrap_replicates),
            "recall": paired_rate_interval(test_labels, d_predictions, predictions, test_videos, 1, seed + 400 + number, args.bootstrap_replicates),
        }

    phase7_lock = json.loads(args.phase7_d_lock.read_text())
    phase7_rows = read_jsonl(args.phase7_predictions)
    phase7_zero = {row["pair_id"]: row["scores"]["N_t0_r0"] for row in phase7_rows}
    d_rounded = {pair_id: stable(score) for pair_id, score in zip(pair_ids, scores["D_pose_aligned_geometry"], strict=True)}
    d_reproduction = {
        "selected_C_matches_phase7": locked["D_pose_aligned_geometry"]["selected_C"] == phase7_lock["selected_C"],
        "recall_threshold_matches_phase7": locked["D_pose_aligned_geometry"]["recall_90_threshold"] == phase7_lock["clean_D_90_recall_threshold"],
        "f1_threshold_matches_phase7": locked["D_pose_aligned_geometry"]["max_f1_threshold"] == phase7_lock["clean_D_max_f1_threshold"],
        "all_10000_scores_match_phase7_zero_noise": d_rounded == phase7_zero,
    }
    if not all(d_reproduction.values()):
        raise RuntimeError(f"Clean D failed Phase 7 reproduction: {d_reproduction}")

    prediction_path = args.output_dir / "movi_e_phase8_regime1_predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for row_number, pair in enumerate(pairs):
            handle.write(json.dumps({
                "dataset": "movi_e", "regime": "in_domain_video_disjoint", "pair_id": str(pair_ids[row_number]),
                "split": str(splits[row_number]), "label": int(labels[row_number]), "video_id": str(pair["video_id"]),
                "negative_difficulty": pair["negative_difficulty"], "temporal_gap": pair["temporal_gap"],
                "temporal_gap_bin": pair["temporal_gap_bin"], "controls": pair["controls"],
                "scores": {system_id: stable(scores[system_id][row_number]) for system_id in SYSTEM_ORDER},
            }, sort_keys=True, separators=(",", ":")) + "\n")
    model_path = args.output_dir / "movi_e_phase8_regime1_models.joblib"
    joblib.dump({"models": fitted, "system_order": SYSTEM_ORDER, "locked": locked}, model_path, compress=3)
    results_path = args.output_dir / "movi_e_phase8_regime1_results.json"
    write_json(results_path, {
        "pipeline": "MOVi-E Phase 8 regime 1 in-domain evaluation", "version": VERSION,
        "dataset": "movi_e", "regime": "train_dev_test_on_video_disjoint_MOVi-E_pools",
        "aggregate": aggregate, "paired_differences_vs_C": paired, "paired_differences_vs_D": paired_vs_d,
        "primary": {
            "estimand": "AUROC_D_minus_C_on_identical_MOVi_E_test_pairs", **primary,
            "success_rule": "paired two-sided 95% video-cluster bootstrap CI lies entirely above zero",
            "success": primary_success,
        },
        "scoring_latency": latency, "clean_D_phase7_reproduction": d_reproduction,
        "checks": {
            "exact_video_disjoint_pair_counts": True, "all_models_fit_train_only": True,
            "all_selection_and_thresholds_dev_only": True, "test_scored_after_lock": True,
            "all_scores_finite": all(np.isfinite(value).all() for value in scores.values()),
            "clean_D_reproduces_phase7": all(d_reproduction.values()), "bootstrap_replicates": args.bootstrap_replicates,
        },
    })
    table_path = args.output_dir / "movi_e_phase8_regime1_results.csv"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["system", "split", "selected_C", "auroc", "pr_auc", "false_match_rate", "recall", "precision_at_max_f1", "recall_at_max_f1", "f1", "brier_score", "log_loss", "ece_10_bins"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for system_id in SYSTEM_ORDER:
            for split in ("dev", "test"):
                value = aggregate[system_id][split]; cal = value["calibration"]
                writer.writerow({
                    "system": system_id, "split": split, "selected_C": locked[system_id]["selected_C"],
                    "auroc": value["auroc"], "pr_auc": value["pr_auc"],
                    "false_match_rate": value["false_match_rate_at_locked_90_recall_threshold"],
                    "recall": value["recall_at_locked_90_recall_threshold"],
                    "precision_at_max_f1": value["precision_at_locked_max_f1_threshold"],
                    "recall_at_max_f1": value["recall_at_locked_max_f1_threshold"], "f1": value["f1_at_locked_max_f1_threshold"],
                    "brier_score": cal["brier_score"], "log_loss": cal["log_loss"], "ece_10_bins": cal["expected_calibration_error_10_bins"],
                })
    manifest_path = args.output_dir / "movi_e_phase8_regime1_manifest.json"
    output_paths = [lock_path, prediction_path, model_path, results_path, table_path]
    write_json(manifest_path, {
        "pipeline": "MOVi-E Phase 8 regime 1 output manifest", "version": VERSION, "status": "pass",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
        "inputs": {name: sha256(path) for name, path in {
            "features": args.features, "feature_manifest": args.feature_manifest, "pairs": args.pairs,
            "system_config": args.system_config, "phase7_D_lock": args.phase7_d_lock, "phase7_predictions": args.phase7_predictions,
        }.items()},
        "counts": {"systems": len(SYSTEM_ORDER), "pairs": len(pairs), "train": 6000, "dev": 2000, "test": 2000},
        "checks": {"all_results_checks_pass": True, "primary_success_decision_recorded": True, "clean_D_phase7_reproduction_passed": all(d_reproduction.values())},
        "outputs": {path.name: sha256(path) for path in output_paths},
    })
    print(f"Complete: MOVi-E regime 1, {len(SYSTEM_ORDER)} systems, primary_success={primary_success}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
