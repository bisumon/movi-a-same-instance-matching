#!/usr/bin/env python3
"""Derive presentation-only paired PR-AUC and locked-F1 intervals from frozen predictions."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


SEED = 20260825
REPLICATES = 10_000
REFERENCE = "C_camera_geometry"
COMPARISON = "D_pose_aligned_geometry"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def interval(values: list[float]) -> dict[str, float | int]:
    return {
        "paired_video_cluster_ci_low": quantile(values, 0.025),
        "paired_video_cluster_ci_high": quantile(values, 0.975),
        "bootstrap_replicates": len(values),
    }


def average_precision(labels: list[int], scores: list[float], clusters: list[int], weights: list[int]) -> float:
    order = sorted(range(len(labels)), key=lambda index: scores[index], reverse=True)
    total_positive = sum(weights[clusters[index]] for index, label in enumerate(labels) if label == 1)
    if total_positive <= 0:
        raise ValueError("Bootstrap replicate has no positive weight")
    true_positive = 0
    false_positive = 0
    prior_recall = 0.0
    result = 0.0
    position = 0
    while position < len(order):
        score = scores[order[position]]
        end = position
        while end < len(order) and scores[order[end]] == score:
            index = order[end]
            weight = weights[clusters[index]]
            if labels[index] == 1:
                true_positive += weight
            else:
                false_positive += weight
            end += 1
        recall = true_positive / total_positive
        denominator = true_positive + false_positive
        precision = 0.0 if denominator == 0 else true_positive / denominator
        result += (recall - prior_recall) * precision
        prior_recall = recall
        position = end
    return result


def weighted_f1(labels: list[int], predictions: list[bool], clusters: list[int], weights: list[int]) -> float:
    true_positive = false_positive = false_negative = 0
    for label, prediction, cluster in zip(labels, predictions, clusters):
        weight = weights[cluster]
        if label == 1 and prediction:
            true_positive += weight
        elif label == 0 and prediction:
            false_positive += weight
        elif label == 1 and not prediction:
            false_negative += weight
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


def derive(predictions_path: Path, lock_path: Path, seed_offset: int) -> dict:
    rows = [row for row in read_jsonl(predictions_path) if row["split"] == "test"]
    if len(rows) != 2_000:
        raise ValueError(f"Expected 2,000 test rows in {predictions_path}, found {len(rows)}")
    labels = [int(row["label"]) for row in rows]
    videos = [str(row["video_id"]) for row in rows]
    reference_scores = [float(row["scores"][REFERENCE]) for row in rows]
    comparison_scores = [float(row["scores"][COMPARISON]) for row in rows]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))["systems"]
    reference_predictions = [score >= float(lock[REFERENCE]["max_f1_threshold"]) for score in reference_scores]
    comparison_predictions = [score >= float(lock[COMPARISON]["max_f1_threshold"]) for score in comparison_scores]

    unique_videos = sorted(set(videos), key=int)
    video_to_cluster = {video: index for index, video in enumerate(unique_videos)}
    clusters = [video_to_cluster[video] for video in videos]
    rng = random.Random(SEED + seed_offset)
    pr_deltas: list[float] = []
    f1_deltas: list[float] = []
    for _ in range(REPLICATES):
        cluster_weights = [0] * len(unique_videos)
        for _ in range(len(unique_videos)):
            cluster_weights[rng.randrange(len(unique_videos))] += 1
        pr_deltas.append(
            average_precision(labels, comparison_scores, clusters, cluster_weights)
            - average_precision(labels, reference_scores, clusters, cluster_weights)
        )
        f1_deltas.append(
            weighted_f1(labels, comparison_predictions, clusters, cluster_weights)
            - weighted_f1(labels, reference_predictions, clusters, cluster_weights)
        )
    unit_weights = [1] * len(unique_videos)
    observed_pr = average_precision(labels, comparison_scores, clusters, unit_weights) - average_precision(
        labels, reference_scores, clusters, unit_weights
    )
    observed_f1 = weighted_f1(labels, comparison_predictions, clusters, unit_weights) - weighted_f1(
        labels, reference_predictions, clusters, unit_weights
    )
    return {
        "video_clusters": len(unique_videos),
        "test_pairs": len(rows),
        "pr_auc": {"comparison_minus_reference": observed_pr, **interval(pr_deltas)},
        "f1_at_locked_max_f1_threshold": {"comparison_minus_reference": observed_f1, **interval(f1_deltas)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    repo = args.repo.resolve()
    result = {
        "version": "1.0.0",
        "purpose": "presentation-only descriptive intervals; frozen Phase 9 decisions are unchanged",
        "seed": SEED,
        "reference": REFERENCE,
        "comparison": COMPARISON,
        "datasets": {
            "movi_e": derive(
                repo / "predictions/movi_de/movi_e_phase8_regime1_predictions.jsonl",
                repo / "results/movi_de_phase8_regime1/movi_e_in_domain_locked_config.json",
                9100,
            ),
            "movi_d": derive(
                repo / "predictions/movi_de/movi_d_phase8_regime2_predictions.jsonl",
                repo / "results/movi_de_phase8_regime2/movi_d_in_domain_locked_config.json",
                9200,
            ),
        },
    }
    output = repo / "results/movi_de_final/presentation_table_metrics.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "datasets": result["datasets"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
