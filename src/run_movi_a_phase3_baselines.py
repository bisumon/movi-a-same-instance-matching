#!/usr/bin/env python3
"""Fit Phase 3 logistic baselines on train, tune on dev, and score test once."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


VERSION = "1.0.1"
CONFIG_ORDER = ("A_rgb_only", "B_rgb_2d", "C_rgb_2d_3d", "geometry_only")
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_float(value: float) -> float:
    return round(float(value), 12)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_boundary(scores: np.ndarray, selected_score: float) -> float:
    """Choose a threshold midpoint that preserves the selected classification set."""
    lower_scores = scores[scores < selected_score]
    if lower_scores.size:
        return float((selected_score + float(lower_scores.max())) / 2.0)
    return float(np.nextafter(selected_score, -np.inf))


def choose_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(scores)
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in candidates:
        value = f1_score(labels, scores >= threshold)
        if value > best_f1 + 1e-12 or (abs(value - best_f1) <= 1e-12 and threshold > best_threshold):
            best_threshold, best_f1 = float(threshold), float(value)
    return stable_boundary(scores, best_threshold), best_f1


def choose_recall_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    target_recall: float = 0.90,
) -> tuple[float, float, float]:
    positives = int(labels.sum())
    negatives = int((labels == 0).sum())
    for threshold in np.unique(scores)[::-1]:
        predictions = scores >= threshold
        recall = float(((predictions == 1) & (labels == 1)).sum() / positives)
        if recall >= target_recall:
            false_match_rate = float(((predictions == 1) & (labels == 0)).sum() / negatives)
            return stable_boundary(scores, float(threshold)), recall, false_match_rate
    return float("-inf"), 1.0, 1.0


def evaluate(
    labels: np.ndarray,
    scores: np.ndarray,
    f1_threshold: float,
    recall_threshold: float,
) -> dict[str, float]:
    f1_predictions = scores >= f1_threshold
    recall_predictions = scores >= recall_threshold
    positives = int(labels.sum())
    negatives = int((labels == 0).sum())
    return {
        "auroc": stable_float(roc_auc_score(labels, scores)),
        "pr_auc": stable_float(average_precision_score(labels, scores)),
        "f1_at_locked_threshold": stable_float(f1_score(labels, f1_predictions)),
        "locked_f1_threshold": float(f1_threshold),
        "false_match_rate_at_locked_90_recall_threshold": stable_float(
            ((recall_predictions == 1) & (labels == 0)).sum() / negatives
        ),
        "achieved_recall_at_locked_90_recall_threshold": stable_float(
            ((recall_predictions == 1) & (labels == 1)).sum() / positives
        ),
        "locked_90_recall_threshold": float(recall_threshold),
    }


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.features, allow_pickle=False)
    pair_ids = data["pair_ids"].astype(str)
    splits = data["splits"].astype(str)
    labels = data["labels"].astype(np.int8)
    feature_names = data["feature_names"].astype(str).tolist()
    matrix = data["features"].astype(np.float64, copy=False)
    if matrix.shape[0] != 10000 or len(set(pair_ids)) != 10000 or not np.isfinite(matrix).all():
        raise ValueError("Invalid Phase 3 feature artifact")
    manifest = json.loads(args.feature_manifest.read_text(encoding="utf-8"))
    configured_groups = manifest["feature_groups"]["configurations"]
    feature_indices = {
        configuration: [feature_names.index(name) for name in configured_groups[configuration]]
        for configuration in CONFIG_ORDER
    }
    masks = {split: splits == split for split in ("train", "dev", "test")}
    if {split: int(mask.sum()) for split, mask in masks.items()} != {"train": 6000, "dev": 2000, "test": 2000}:
        raise ValueError("Feature split counts are not 6000/2000/2000")

    pair_rows = [json.loads(line) for line in args.pairs.read_text(encoding="utf-8").splitlines() if line.strip()]
    pair_by_id = {str(row["pair_id"]): row for row in pair_rows}
    if set(pair_by_id) != set(pair_ids):
        raise ValueError("Pair manifest and feature pair IDs differ")

    fitted_models: dict[str, Pipeline] = {}
    dev_scores_by_configuration: dict[str, np.ndarray] = {}
    locked: dict[str, Any] = {}
    for configuration in CONFIG_ORDER:
        indices = feature_indices[configuration]
        train_x, train_y = matrix[masks["train"]][:, indices], labels[masks["train"]]
        dev_x, dev_y = matrix[masks["dev"]][:, indices], labels[masks["dev"]]
        tuning_rows = []
        best: tuple[float, float, Pipeline, np.ndarray] | None = None
        for c_value in C_GRID:
            model = Pipeline(
                [
                    ("standardizer", StandardScaler()),
                    (
                        "logistic_regression",
                        LogisticRegression(
                            C=c_value,
                            solver="lbfgs",
                            max_iter=5000,
                            random_state=args.seed,
                        ),
                    ),
                ]
            )
            model.fit(train_x, train_y)
            dev_scores = model.predict_proba(dev_x)[:, 1]
            dev_auroc = float(roc_auc_score(dev_y, dev_scores))
            tuning_rows.append({"C": c_value, "dev_auroc": stable_float(dev_auroc)})
            candidate = (dev_auroc, -c_value, model, dev_scores)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        assert best is not None
        selected_model, selected_dev_scores = best[2], best[3]
        selected_c = float(selected_model.named_steps["logistic_regression"].C)
        f1_threshold, dev_f1 = choose_f1_threshold(dev_y, selected_dev_scores)
        recall_threshold, achieved_recall, dev_fmr = choose_recall_threshold(dev_y, selected_dev_scores)
        fitted_models[configuration] = selected_model
        dev_scores_by_configuration[configuration] = selected_dev_scores
        locked[configuration] = {
            "features": configured_groups[configuration],
            "selected_C": selected_c,
            "selection_metric": "dev AUROC",
            "C_grid": list(C_GRID),
            "C_tuning": tuning_rows,
            "f1_threshold_selected_on_dev": float(f1_threshold),
            "dev_f1_at_selected_threshold": stable_float(dev_f1),
            "recall_90_threshold_selected_on_dev": float(recall_threshold),
            "dev_achieved_recall": stable_float(achieved_recall),
            "dev_false_match_rate": stable_float(dev_fmr),
            "fit_scope": "train only",
            "standardizer_fit_scope": "train only",
        }
        print(
            f"locked {configuration}: C={selected_c:g}, dev_AUROC={best[0]:.4f}, "
            f"F1_threshold={f1_threshold:.4f}",
            flush=True,
        )

    # This artifact is deliberately written before test scores are computed.
    locked_config = {
        "pipeline": "MOVi-A Phase 3 locked model and threshold configuration",
        "version": VERSION,
        "seed": args.seed,
        "decision_protocol": {
            "model_fit": "train only",
            "regularization_selection": "maximum dev AUROC; ties choose smaller C",
            "f1_threshold_selection": "maximum dev F1; ties choose higher threshold",
            "90_recall_threshold_selection": "highest dev threshold achieving at least 90% recall",
            "test_policy": "no test labels or scores used before this configuration was written",
        },
        "configurations": locked,
        "input_features_sha256": sha256(args.features),
    }
    locked_path = args.output_dir / "phase3_locked_config.json"
    write_json(locked_path, locked_config)

    # Test is accessed only after the complete configuration has been locked.
    scores_by_configuration: dict[str, np.ndarray] = {}
    results: dict[str, Any] = {}
    scoring_latency: dict[str, Any] = {}
    for configuration in CONFIG_ORDER:
        indices = feature_indices[configuration]
        model = fitted_models[configuration]
        all_scores = model.predict_proba(matrix[:, indices])[:, 1]
        scores_by_configuration[configuration] = all_scores
        f1_threshold = float(locked[configuration]["f1_threshold_selected_on_dev"])
        recall_threshold = float(locked[configuration]["recall_90_threshold_selected_on_dev"])
        results[configuration] = {
            "dev": evaluate(labels[masks["dev"]], all_scores[masks["dev"]], f1_threshold, recall_threshold),
            "test": evaluate(labels[masks["test"]], all_scores[masks["test"]], f1_threshold, recall_threshold),
        }
        test_x = matrix[masks["test"]][:, indices]
        repetitions = []
        for _ in range(50):
            started = time.perf_counter_ns()
            model.predict_proba(test_x)
            repetitions.append((time.perf_counter_ns() - started) / len(test_x) / 1000.0)
        scoring_latency[configuration] = {
            "microseconds_per_pair_p50": stable_float(np.quantile(repetitions, 0.50)),
            "microseconds_per_pair_p95": stable_float(np.quantile(repetitions, 0.95)),
            "batch_size": len(test_x),
            "repetitions": len(repetitions),
        }

    prediction_rows = []
    for row_index, pair_id in enumerate(pair_ids):
        pair = pair_by_id[pair_id]
        prediction_rows.append(
            {
                "pair_id": pair_id,
                "split": str(splits[row_index]),
                "label": int(labels[row_index]),
                "video_id": str(pair["video_id"]),
                "negative_difficulty": pair["negative_difficulty"],
                "temporal_gap": int(pair["temporal_gap"]),
                "temporal_gap_bin": str(pair["temporal_gap_bin"]),
                "scores": {
                    configuration: stable_float(scores_by_configuration[configuration][row_index])
                    for configuration in CONFIG_ORDER
                },
            }
        )
    predictions_path = args.output_dir / "phase3_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    models_path = args.output_dir / "phase3_models.joblib"
    joblib.dump(
        {
            "models": fitted_models,
            "feature_indices": feature_indices,
            "feature_names": feature_names,
            "locked_config": locked_config,
        },
        models_path,
        compress=3,
    )
    results_path = args.output_dir / "phase3_results.json"
    write_json(results_path, {"results": results, "pair_scoring_latency": scoring_latency})
    output_manifest = {
        "pipeline": "MOVi-A Phase 3 train/dev/test logistic baselines",
        "version": VERSION,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "platform": platform.platform(),
        },
        "inputs": {
            "features_sha256": sha256(args.features),
            "feature_manifest_sha256": sha256(args.feature_manifest),
            "pairs_sha256": sha256(args.pairs),
        },
        "checks": {
            "train_pairs": int(masks["train"].sum()),
            "dev_pairs": int(masks["dev"].sum()),
            "test_pairs": int(masks["test"].sum()),
            "test_scored_after_locked_config_written": True,
            "all_prediction_scores_finite": all(
                np.isfinite(scores).all() for scores in scores_by_configuration.values()
            ),
            "model_count": len(fitted_models),
        },
        "outputs": {
            path.name: sha256(path)
            for path in (locked_path, predictions_path, models_path, results_path)
        },
    }
    write_json(args.output_dir / "phase3_baseline_manifest.json", output_manifest)
    print("Complete: four locked Phase 3 configurations and 10,000 prediction rows", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
