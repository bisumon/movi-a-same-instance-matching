#!/usr/bin/env python3
"""Evaluate locked MOVi-A Phase 3 predictions with the Phase 4 protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


VERSION = "1.0.0"
CONFIG_ORDER = ("A_rgb_only", "B_rgb_2d", "C_rgb_2d_3d", "geometry_only")
METRIC_NAMES = (
    "auroc",
    "pr_auc",
    "f1_at_locked_threshold",
    "false_match_rate_at_locked_90_recall_threshold",
    "achieved_recall_at_locked_90_recall_threshold",
)
PRIMARY_COMPARISON = ("B_rgb_2d", "C_rgb_2d_3d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--locked-config", type=Path, required=True)
    parser.add_argument("--phase3-results", type=Path, required=True)
    parser.add_argument("--rgb-manifest", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--model-inputs", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def thresholds_for(locked: dict[str, Any], configuration: str) -> tuple[float, float]:
    row = locked["configurations"][configuration]
    return (
        float(row["f1_threshold_selected_on_dev"]),
        float(row["recall_90_threshold_selected_on_dev"]),
    )


def evaluate(
    labels: np.ndarray,
    scores: np.ndarray,
    f1_threshold: float,
    recall_threshold: float,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if not labels.size or not np.isfinite(scores).all():
        raise ValueError("Metric input is empty or non-finite")
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC/PR-AUC require both classes")
    f1_predictions = scores >= f1_threshold
    recall_predictions = scores >= recall_threshold
    return {
        "n_pairs": int(labels.size),
        "n_positive": positives,
        "n_negative": negatives,
        "auroc": stable_float(roc_auc_score(labels, scores)),
        "pr_auc": stable_float(average_precision_score(labels, scores)),
        "f1_at_locked_threshold": stable_float(f1_score(labels, f1_predictions)),
        "false_match_rate_at_locked_90_recall_threshold": stable_float(
            ((recall_predictions == 1) & (labels == 0)).sum() / negatives
        ),
        "achieved_recall_at_locked_90_recall_threshold": stable_float(
            ((recall_predictions == 1) & (labels == 1)).sum() / positives
        ),
        "locked_f1_threshold": float(f1_threshold),
        "locked_90_recall_threshold": float(recall_threshold),
    }


def train_tertile_cutpoints(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Training stratum values must be finite and non-empty")
    low, high = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    if not low < high:
        raise ValueError("Training tertile cutpoints are not distinct")
    return stable_float(low), stable_float(high)


def assign_tertiles(values: np.ndarray, cutpoints: tuple[float, float]) -> np.ndarray:
    low, high = cutpoints
    values = np.asarray(values, dtype=np.float64)
    return np.where(values <= low, "low", np.where(values <= high, "medium", "high"))


def hard_easy_mask(labels: np.ndarray, difficulties: np.ndarray, difficulty: str) -> np.ndarray:
    """Use the identical positive set in both hard- and easy-negative comparisons."""
    return (labels == 1) | ((labels == 0) & (difficulties == difficulty))


def metric_row(prefix: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    return {**prefix, **metrics}


def make_primary_svg(
    path: Path,
    aggregate_rows: list[dict[str, Any]],
    interval_rows: list[dict[str, Any]],
) -> None:
    """Write a dependency-free SVG of test point estimates and cluster-bootstrap CIs."""
    point = {
        (row["configuration"], metric): float(row[metric])
        for row in aggregate_rows
        if row["split"] == "test"
        for metric in METRIC_NAMES[:4]
    }
    intervals = {
        (row["configuration"], row["metric"]): (float(row["ci_lower"]), float(row["ci_upper"]))
        for row in interval_rows
    }
    labels = {
        "auroc": "AUROC",
        "pr_auc": "PR-AUC",
        "f1_at_locked_threshold": "F1 (locked)",
        "false_match_rate_at_locked_90_recall_threshold": "FMR (locked 90% dev recall)",
    }
    colors = {"A_rgb_only": "#8b9aab", "B_rgb_2d": "#4c78a8", "C_rgb_2d_3d": "#e45756", "geometry_only": "#72b7b2"}
    width, height = 1120, 720
    panel_w, panel_h = 510, 270
    origins = [(70, 70), (610, 70), (70, 405), (610, 405)]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#243447}.title{font-size:18px;font-weight:600}.axis{font-size:12px}.value{font-size:11px}</style>',
    ]
    for (metric, title), (ox, oy) in zip(labels.items(), origins):
        values = [point[(cfg, metric)] for cfg in CONFIG_ORDER]
        bounds = [intervals[(cfg, metric)] for cfg in CONFIG_ORDER]
        minimum = max(0.0, min(lo for lo, _ in bounds) - 0.04)
        maximum = min(1.0, max(hi for _, hi in bounds) + 0.04)
        if maximum - minimum < 0.15:
            pad = (0.15 - (maximum - minimum)) / 2
            minimum, maximum = max(0.0, minimum - pad), min(1.0, maximum + pad)
        plot_left, plot_right = ox + 160, ox + panel_w - 30
        parts.append(f'<text x="{ox}" y="{oy}" class="title">{title}</text>')
        parts.append(f'<line x1="{plot_left}" y1="{oy+25}" x2="{plot_left}" y2="{oy+235}" stroke="#aab5c0"/>')
        for tick_index in range(6):
            tick = minimum + tick_index * (maximum - minimum) / 5
            x = plot_left + tick_index * (plot_right - plot_left) / 5
            parts.append(f'<line x1="{x:.1f}" y1="{oy+25}" x2="{x:.1f}" y2="{oy+235}" stroke="#edf0f2"/>')
            parts.append(f'<text x="{x:.1f}" y="{oy+253}" text-anchor="middle" class="axis">{tick:.2f}</text>')
        for idx, cfg in enumerate(CONFIG_ORDER):
            y = oy + 55 + idx * 50
            val, (lo, hi) = values[idx], bounds[idx]
            scale = lambda v: plot_left + (v - minimum) / (maximum - minimum) * (plot_right - plot_left)
            parts.append(f'<text x="{ox}" y="{y+4}" class="axis">{cfg}</text>')
            parts.append(f'<line x1="{scale(lo):.1f}" y1="{y}" x2="{scale(hi):.1f}" y2="{y}" stroke="{colors[cfg]}" stroke-width="4"/>')
            parts.append(f'<line x1="{scale(lo):.1f}" y1="{y-6}" x2="{scale(lo):.1f}" y2="{y+6}" stroke="{colors[cfg]}"/>')
            parts.append(f'<line x1="{scale(hi):.1f}" y1="{y-6}" x2="{scale(hi):.1f}" y2="{y+6}" stroke="{colors[cfg]}"/>')
            parts.append(f'<circle cx="{scale(val):.1f}" cy="{y}" r="6" fill="{colors[cfg]}"/>')
            parts.append(f'<text x="{min(plot_right-2, scale(val)+9):.1f}" y="{y-9}" class="value">{val:.3f}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = read_jsonl(args.predictions)
    pairs = read_jsonl(args.pairs)
    model_inputs = read_jsonl(args.model_inputs)
    diagnostics = read_jsonl(args.diagnostics)
    locked = json.loads(args.locked_config.read_text(encoding="utf-8"))
    phase3_results = json.loads(args.phase3_results.read_text(encoding="utf-8"))
    rgb_manifest = json.loads(args.rgb_manifest.read_text(encoding="utf-8"))
    if tuple(locked["configurations"].keys()) != CONFIG_ORDER:
        raise ValueError("Unexpected Phase 3 configurations or order")

    prediction_by_id = {str(row["pair_id"]): row for row in predictions}
    pair_by_id = {str(row["pair_id"]): row for row in pairs}
    model_by_id = {str(row["observation_id"]): row for row in model_inputs}
    diagnostics_by_id = {str(row["observation_id"]): row for row in diagnostics}
    if len(predictions) != 10_000 or len(prediction_by_id) != 10_000:
        raise ValueError("Expected 10,000 unique prediction rows")
    if len(pairs) != 10_000 or set(pair_by_id) != set(prediction_by_id):
        raise ValueError("Prediction and pair IDs differ")
    if len(model_by_id) != len(model_inputs) or len(diagnostics_by_id) != len(diagnostics):
        raise ValueError("Duplicate observation IDs")

    enriched: list[dict[str, Any]] = []
    for prediction in predictions:
        pair_id = str(prediction["pair_id"])
        pair = pair_by_id[pair_id]
        for field in ("split", "label", "video_id", "negative_difficulty", "temporal_gap", "temporal_gap_bin"):
            if prediction[field] != pair[field]:
                raise ValueError(f"Prediction/pair mismatch for {pair_id}: {field}")
        endpoint_ids = (str(pair["observation_id_a"]), str(pair["observation_id_b"]))
        if any(obs_id not in model_by_id or obs_id not in diagnostics_by_id for obs_id in endpoint_ids):
            raise ValueError(f"Missing Phase 1 endpoint for {pair_id}")
        pair_visibility = min(float(model_by_id[obs_id]["visibility"]) for obs_id in endpoint_ids)
        speeds = []
        for obs_id in endpoint_ids:
            velocity = np.asarray(diagnostics_by_id[obs_id]["gt_world_velocity_xyz"], dtype=np.float64)
            if velocity.shape != (3,) or not np.isfinite(velocity).all():
                raise ValueError(f"Invalid diagnostic velocity for {obs_id}")
            speeds.append(float(np.linalg.norm(velocity)))
        scores = {configuration: float(prediction["scores"][configuration]) for configuration in CONFIG_ORDER}
        if not np.isfinite(list(scores.values())).all():
            raise ValueError(f"Non-finite prediction score for {pair_id}")
        enriched.append(
            {
                **prediction,
                "scores": scores,
                "pair_visibility": pair_visibility,
                "pair_motion_speed": float(np.mean(speeds)),
            }
        )

    split_counts = {split: sum(row["split"] == split for row in enriched) for split in ("train", "dev", "test")}
    if split_counts != {"train": 6000, "dev": 2000, "test": 2000}:
        raise ValueError(f"Unexpected split counts: {split_counts}")
    train_rows = [row for row in enriched if row["split"] == "train"]
    visibility_cutpoints = train_tertile_cutpoints(np.asarray([row["pair_visibility"] for row in train_rows]))
    motion_cutpoints = train_tertile_cutpoints(np.asarray([row["pair_motion_speed"] for row in train_rows]))
    for row in enriched:
        row["visibility_stratum"] = str(assign_tertiles(np.asarray([row["pair_visibility"]]), visibility_cutpoints)[0])
        row["motion_stratum"] = str(assign_tertiles(np.asarray([row["pair_motion_speed"]]), motion_cutpoints)[0])

    aggregate_rows: list[dict[str, Any]] = []
    aggregate_json: dict[str, Any] = {}
    for configuration in CONFIG_ORDER:
        f1_threshold, recall_threshold = thresholds_for(locked, configuration)
        aggregate_json[configuration] = {}
        for split in ("dev", "test"):
            subset = [row for row in enriched if row["split"] == split]
            values = evaluate(
                np.asarray([row["label"] for row in subset]),
                np.asarray([row["scores"][configuration] for row in subset]),
                f1_threshold,
                recall_threshold,
            )
            aggregate_json[configuration][split] = values
            aggregate_rows.append(metric_row({"configuration": configuration, "split": split}, values))
            reference = phase3_results["results"][configuration][split]
            for metric in METRIC_NAMES:
                if abs(float(values[metric]) - float(reference[metric])) > 1e-11:
                    raise ValueError(f"Phase 4 failed to reproduce Phase 3 {configuration}/{split}/{metric}")

    test_rows = [row for row in enriched if row["split"] == "test"]
    test_videos = sorted({str(row["video_id"]) for row in test_rows}, key=lambda value: int(value))
    if len(test_videos) != 10:
        raise ValueError(f"Expected 10 test videos, found {len(test_videos)}")

    per_video_rows: list[dict[str, Any]] = []
    for configuration in CONFIG_ORDER:
        f1_threshold, recall_threshold = thresholds_for(locked, configuration)
        for video_id in test_videos:
            subset = [row for row in test_rows if str(row["video_id"]) == video_id]
            values = evaluate(
                np.asarray([row["label"] for row in subset]),
                np.asarray([row["scores"][configuration] for row in subset]),
                f1_threshold,
                recall_threshold,
            )
            per_video_rows.append(metric_row({"configuration": configuration, "video_id": video_id}, values))

    macro_rows: list[dict[str, Any]] = []
    for configuration in CONFIG_ORDER:
        subset = [row for row in per_video_rows if row["configuration"] == configuration]
        macro_row: dict[str, Any] = {"configuration": configuration, "n_videos": len(subset)}
        for metric in METRIC_NAMES:
            metric_values = np.asarray([float(row[metric]) for row in subset])
            macro_row[f"macro_{metric}"] = stable_float(metric_values.mean())
            macro_row[f"between_video_sd_{metric}"] = stable_float(metric_values.std(ddof=1))
        macro_rows.append(macro_row)

    stratified_rows: list[dict[str, Any]] = []
    labels = np.asarray([row["label"] for row in test_rows], dtype=np.int8)
    difficulties = np.asarray([row["negative_difficulty"] or "positive" for row in test_rows])
    strata: list[tuple[str, str, np.ndarray]] = []
    for difficulty in ("hard", "easy"):
        strata.append(("negative_comparison", f"positive_vs_{difficulty}", hard_easy_mask(labels, difficulties, difficulty)))
    for dimension, values, groups in (
        ("temporal_gap", np.asarray([row["temporal_gap_bin"] for row in test_rows]), ("short", "medium", "long")),
        ("visibility", np.asarray([row["visibility_stratum"] for row in test_rows]), ("low", "medium", "high")),
        ("motion", np.asarray([row["motion_stratum"] for row in test_rows]), ("low", "medium", "high")),
    ):
        for group in groups:
            strata.append((dimension, group, values == group))
    for dimension, group, mask in strata:
        if not (labels[mask] == 1).any() or not (labels[mask] == 0).any():
            raise ValueError(f"Stratum lacks both classes: {dimension}/{group}")
        for configuration in CONFIG_ORDER:
            f1_threshold, recall_threshold = thresholds_for(locked, configuration)
            scores = np.asarray([row["scores"][configuration] for row in test_rows])
            values = evaluate(labels[mask], scores[mask], f1_threshold, recall_threshold)
            stratified_rows.append(metric_row({"dimension": dimension, "stratum": group, "configuration": configuration}, values))

    # Resample whole videos with replacement. The same sampled video multiplicities
    # are reused for every configuration, preserving the paired B-versus-C design.
    video_indices = {
        video_id: np.flatnonzero(np.asarray([str(row["video_id"]) == video_id for row in test_rows]))
        for video_id in test_videos
    }
    rng = np.random.default_rng(args.seed)
    bootstrap_rows: list[dict[str, Any]] = []
    bootstrap_values = {
        configuration: {metric: np.empty(args.bootstrap_replicates, dtype=np.float64) for metric in METRIC_NAMES}
        for configuration in CONFIG_ORDER
    }
    all_scores = {
        configuration: np.asarray([row["scores"][configuration] for row in test_rows], dtype=np.float64)
        for configuration in CONFIG_ORDER
    }
    for replicate in range(args.bootstrap_replicates):
        sampled = rng.integers(0, len(test_videos), size=len(test_videos))
        indices = np.concatenate([video_indices[test_videos[index]] for index in sampled])
        replicate_labels = labels[indices]
        row: dict[str, Any] = {"replicate": replicate + 1}
        for configuration in CONFIG_ORDER:
            f1_threshold, recall_threshold = thresholds_for(locked, configuration)
            values = evaluate(replicate_labels, all_scores[configuration][indices], f1_threshold, recall_threshold)
            for metric in METRIC_NAMES:
                value = float(values[metric])
                bootstrap_values[configuration][metric][replicate] = value
                row[f"{configuration}__{metric}"] = stable_float(value)
        bootstrap_rows.append(row)
        if (replicate + 1) % 1000 == 0:
            print(f"bootstrap {replicate + 1}/{args.bootstrap_replicates}", flush=True)

    interval_rows: list[dict[str, Any]] = []
    for configuration in CONFIG_ORDER:
        point_row = next(row for row in aggregate_rows if row["configuration"] == configuration and row["split"] == "test")
        for metric in METRIC_NAMES:
            values = bootstrap_values[configuration][metric]
            interval_rows.append(
                {
                    "configuration": configuration,
                    "metric": metric,
                    "point_estimate": point_row[metric],
                    "bootstrap_mean": stable_float(values.mean()),
                    "bootstrap_standard_error": stable_float(values.std(ddof=1)),
                    "ci_lower": stable_float(np.quantile(values, 0.025)),
                    "ci_upper": stable_float(np.quantile(values, 0.975)),
                    "confidence_level": 0.95,
                    "method": "percentile video-cluster bootstrap",
                    "replicates": args.bootstrap_replicates,
                }
            )

    paired_rows: list[dict[str, Any]] = []
    baseline, augmented = PRIMARY_COMPARISON
    baseline_point = next(row for row in aggregate_rows if row["configuration"] == baseline and row["split"] == "test")
    augmented_point = next(row for row in aggregate_rows if row["configuration"] == augmented and row["split"] == "test")
    for metric in METRIC_NAMES:
        differences = bootstrap_values[augmented][metric] - bootstrap_values[baseline][metric]
        favorable = "lower" if metric == "false_match_rate_at_locked_90_recall_threshold" else "higher"
        paired_rows.append(
            {
                "baseline_configuration": baseline,
                "augmented_configuration": augmented,
                "metric": metric,
                "favorable_direction": favorable,
                "delta_definition": "augmented_minus_baseline",
                "point_delta": stable_float(float(augmented_point[metric]) - float(baseline_point[metric])),
                "bootstrap_mean_delta": stable_float(differences.mean()),
                "ci_lower": stable_float(np.quantile(differences, 0.025)),
                "ci_upper": stable_float(np.quantile(differences, 0.975)),
                "confidence_level": 0.95,
                "method": "paired percentile video-cluster bootstrap",
                "replicates": args.bootstrap_replicates,
                "ci_excludes_zero": bool(np.quantile(differences, 0.025) > 0 or np.quantile(differences, 0.975) < 0),
            }
        )

    latency_rows: list[dict[str, Any]] = []
    rgb_latency = rgb_manifest["latency"]
    latency_rows.extend(
        [
            {"component": "frozen_rgb_encoder", "configuration": "shared_rgb", "statistic": "forward_ms_per_crop_p50", "value": rgb_latency["forward_ms_per_crop_p50"], "unit": "ms/crop"},
            {"component": "frozen_rgb_encoder", "configuration": "shared_rgb", "statistic": "forward_ms_per_crop_p95", "value": rgb_latency["forward_ms_per_crop_p95"], "unit": "ms/crop"},
            {"component": "end_to_end_embedding_extraction", "configuration": "shared_rgb", "statistic": "wall_ms_per_crop", "value": rgb_latency["wall_ms_per_crop"], "unit": "ms/crop"},
        ]
    )
    for configuration in CONFIG_ORDER:
        row = phase3_results["pair_scoring_latency"][configuration]
        latency_rows.extend(
            [
                {"component": "logistic_pair_scoring", "configuration": configuration, "statistic": "microseconds_per_pair_p50", "value": row["microseconds_per_pair_p50"], "unit": "microseconds/pair"},
                {"component": "logistic_pair_scoring", "configuration": configuration, "statistic": "microseconds_per_pair_p95", "value": row["microseconds_per_pair_p95"], "unit": "microseconds/pair"},
            ]
        )

    metric_fields = [
        "n_pairs", "n_positive", "n_negative", *METRIC_NAMES,
        "locked_f1_threshold", "locked_90_recall_threshold",
    ]
    aggregate_path = args.output_dir / "phase4_aggregate_metrics.csv"
    per_video_path = args.output_dir / "phase4_per_video_metrics.csv"
    macro_path = args.output_dir / "phase4_macro_metrics.csv"
    stratified_path = args.output_dir / "phase4_stratified_metrics.csv"
    intervals_path = args.output_dir / "phase4_bootstrap_intervals.csv"
    paired_path = args.output_dir / "phase4_paired_differences.csv"
    replicates_path = args.output_dir / "phase4_bootstrap_replicates.csv"
    latency_path = args.output_dir / "phase4_latency.csv"
    figure_path = args.output_dir / "phase4_primary_comparison.svg"
    results_path = args.output_dir / "phase4_results.json"
    write_csv(aggregate_path, aggregate_rows, ["configuration", "split", *metric_fields])
    write_csv(per_video_path, per_video_rows, ["configuration", "video_id", *metric_fields])
    write_csv(macro_path, macro_rows, list(macro_rows[0]))
    write_csv(stratified_path, stratified_rows, ["dimension", "stratum", "configuration", *metric_fields])
    write_csv(intervals_path, interval_rows, list(interval_rows[0]))
    write_csv(paired_path, paired_rows, list(paired_rows[0]))
    write_csv(replicates_path, bootstrap_rows, list(bootstrap_rows[0]))
    write_csv(latency_path, latency_rows, ["component", "configuration", "statistic", "value", "unit"])
    make_primary_svg(figure_path, aggregate_rows, interval_rows)

    results = {
        "aggregate": aggregate_json,
        "train_defined_strata": {
            "visibility": {
                "pair_definition": "minimum endpoint visibility in pixels",
                "tertile_cutpoints": list(visibility_cutpoints),
            },
            "motion": {
                "pair_definition": "mean endpoint Euclidean norm of ground-truth world velocity",
                "tertile_cutpoints": list(motion_cutpoints),
                "evaluation_only": True,
            },
        },
        "test_video_macro": {row["configuration"]: row for row in macro_rows},
        "primary_paired_comparison": paired_rows,
        "bootstrap": {
            "unit": "video",
            "test_video_count": len(test_videos),
            "resample_size_in_videos": len(test_videos),
            "sampling": "with replacement; sampled clusters retain all within-video pairs",
            "paired_across_configurations": True,
            "replicates": args.bootstrap_replicates,
            "seed": args.seed,
            "confidence_interval": "two-sided 95% percentile",
        },
    }
    write_json(results_path, results)

    manifest_path = args.output_dir / "phase4_evaluation_manifest.json"
    output_paths = [
        aggregate_path, per_video_path, macro_path, stratified_path, intervals_path,
        paired_path, replicates_path, latency_path, figure_path, results_path,
    ]
    manifest = {
        "pipeline": "MOVi-A Phase 4 locked evaluation",
        "version": VERSION,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
        },
        "protocol": {
            "no_refitting_or_retuning": True,
            "model_thresholds": "locked Phase 3 dev-selected thresholds",
            "primary_comparison": {"baseline": baseline, "augmented": augmented},
            "hard_easy_rule": "all test positives are reused in each comparison; negatives are filtered by difficulty",
            "stratum_cutpoints": "training-pair tertiles fixed before application to dev/test",
            "motion_diagnostics_use": "evaluation-only; never joined into model features",
            "bootstrap": results["bootstrap"],
        },
        "inputs": {
            "predictions_sha256": sha256(args.predictions),
            "locked_config_sha256": sha256(args.locked_config),
            "phase3_results_sha256": sha256(args.phase3_results),
            "rgb_manifest_sha256": sha256(args.rgb_manifest),
            "pairs_sha256": sha256(args.pairs),
            "model_inputs_sha256": sha256(args.model_inputs),
            "diagnostics_sha256": sha256(args.diagnostics),
        },
        "checks": {
            "prediction_pair_join_exact": True,
            "phase3_aggregate_metrics_reproduced": True,
            "all_scores_finite": True,
            "split_counts": split_counts,
            "test_video_count": len(test_videos),
            "all_test_strata_have_both_classes": True,
            "bootstrap_replicates": args.bootstrap_replicates,
        },
        "outputs": {path.name: sha256(path) for path in output_paths},
    }
    write_json(manifest_path, manifest)
    print(f"Complete: Phase 4 evaluation written to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
