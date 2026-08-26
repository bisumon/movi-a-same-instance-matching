#!/usr/bin/env python3
"""Combine Phase 7 results and compute paired video-cluster AUROC intervals."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stable(value: float) -> float:
    return round(float(value), 12)


def cluster_auc_matrix(labels: np.ndarray, scores: np.ndarray, clusters: np.ndarray, cluster_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positive = labels == 1
    negative = labels == 0
    p_scores, n_scores = scores[positive], scores[negative]
    p_clusters, n_clusters = clusters[positive], clusters[negative]
    comparisons = (p_scores[:, None] > n_scores[None, :]).astype(np.float64)
    comparisons += 0.5 * (p_scores[:, None] == n_scores[None, :])
    group = (p_clusters[:, None] * cluster_count + n_clusters[None, :]).reshape(-1)
    matrix = np.bincount(group, weights=comparisons.reshape(-1), minlength=cluster_count * cluster_count).reshape(cluster_count, cluster_count)
    return matrix, np.bincount(p_clusters, minlength=cluster_count), np.bincount(n_clusters, minlength=cluster_count)


def paired_cluster_interval(labels: np.ndarray, clean: np.ndarray, noisy: np.ndarray, video_ids: np.ndarray, seed: int, replicates: int = 10000) -> dict[str, float | int]:
    videos = sorted(set(video_ids.tolist()), key=int)
    mapping = {video: index for index, video in enumerate(videos)}
    clusters = np.asarray([mapping[value] for value in video_ids], dtype=np.int64)
    clean_matrix, pos_counts, neg_counts = cluster_auc_matrix(labels, clean, clusters, len(videos))
    noisy_matrix, _, _ = cluster_auc_matrix(labels, noisy, clusters, len(videos))
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(videos), np.full(len(videos), 1.0 / len(videos)), size=replicates)
    denominator = (weights @ pos_counts) * (weights @ neg_counts)
    valid = denominator > 0
    numerator = np.einsum("bi,ij,bj->b", weights, noisy_matrix - clean_matrix, weights, optimize=True)
    delta = numerator[valid] / denominator[valid]
    observed_clean = clean_matrix.sum() / (pos_counts.sum() * neg_counts.sum())
    observed_noisy = noisy_matrix.sum() / (pos_counts.sum() * neg_counts.sum())
    return {
        "noise_minus_clean_auroc": stable(observed_noisy - observed_clean),
        "paired_video_cluster_ci_low": stable(np.quantile(delta, 0.025)),
        "paired_video_cluster_ci_high": stable(np.quantile(delta, 0.975)),
        "bootstrap_replicates_requested": replicates,
        "bootstrap_replicates_valid": int(valid.sum()),
        "test_video_clusters": len(videos),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movi-d-dir", type=Path, required=True)
    parser.add_argument("--movi-e-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined: dict[str, Any] = {}
    table_rows = []
    for dataset, directory in (("movi_d", args.movi_d_dir), ("movi_e", args.movi_e_dir)):
        results_path = directory / f"{dataset}_phase7_pose_noise_results.json"
        predictions_path = directory / f"{dataset}_phase7_pose_noise_predictions.jsonl"
        results = json.loads(results_path.read_text(encoding="utf-8"))
        predictions = [row for row in read_jsonl(predictions_path) if row["split"] == "test"]
        if len(predictions) != 2000:
            raise ValueError(f"Expected 2,000 {dataset} test predictions")
        labels = np.asarray([row["label"] for row in predictions], dtype=np.int8)
        videos = np.asarray([str(row["video_id"]) for row in predictions])
        zero_id = results["clean_condition_id"]
        clean = np.asarray([row["scores"][zero_id] for row in predictions], dtype=np.float64)
        conditions = []
        for condition_number, condition in enumerate(results["conditions"]):
            condition_id = condition["condition_id"]
            scores = np.asarray([row["scores"][condition_id] for row in predictions], dtype=np.float64)
            interval = paired_cluster_interval(
                labels, clean, scores, videos,
                args.seed + (0 if dataset == "movi_d" else 1), args.bootstrap_replicates,
            )
            enriched = {**condition, "test_paired_auroc_difference": interval}
            conditions.append(enriched)
            for split in ("dev", "test"):
                metric = condition["metrics"][split]
                table_rows.append({
                    "dataset": dataset, "condition_id": condition_id,
                    "translation_std_scene_units": condition["translation_std_scene_units"],
                    "rotation_std_degrees": condition["rotation_std_degrees"], "split": split,
                    "auroc": metric["auroc"], "pr_auc": metric["pr_auc"],
                    "false_match_rate": metric["false_match_rate_at_clean_D_locked_90_recall_threshold"],
                    "recall": metric["recall_at_clean_D_locked_90_recall_threshold"],
                    "f1": metric["f1_at_clean_D_locked_max_f1_threshold"],
                    "test_noise_minus_clean_auroc": interval["noise_minus_clean_auroc"] if split == "test" else "",
                    "test_paired_ci_low": interval["paired_video_cluster_ci_low"] if split == "test" else "",
                    "test_paired_ci_high": interval["paired_video_cluster_ci_high"] if split == "test" else "",
                })
        clean_condition = next(row for row in conditions if row["condition_id"] == zero_id)
        nonzero = [row for row in conditions if row["condition_id"] != zero_id]
        worst = min(nonzero, key=lambda row: row["metrics"]["test"]["auroc"])
        combined[dataset] = {
            "clean_condition": clean_condition, "conditions": conditions,
            "worst_test_auroc_condition": worst,
            "source_results_sha256": sha256(results_path), "source_predictions_sha256": sha256(predictions_path),
        }
        print(f"{dataset}: completed {len(conditions)} paired 10,000-replicate cluster intervals", flush=True)
    combined_path = args.output_dir / "phase7_pose_noise_combined_results.json"
    combined_path.write_text(json.dumps({
        "pipeline": "MOVi-D/E Phase 7 combined pose-noise analysis", "version": "1.0.0",
        "seed": args.seed, "bootstrap": {
            "method": "paired video-cluster bootstrap retaining all pairs from sampled videos",
            "replicates": args.bootstrap_replicates, "confidence_level": 0.95,
        }, "datasets": combined,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    table_path = args.output_dir / "phase7_pose_noise_results_table.csv"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader(); writer.writerows(table_rows)
    manifest_path = args.output_dir / "phase7_pose_noise_summary_manifest.json"
    manifest_path.write_text(json.dumps({
        "pipeline": "MOVi-D/E Phase 7 summary manifest", "status": "pass",
        "checks": {
            "both_datasets_present": set(combined) == {"movi_d", "movi_e"},
            "exact_36_conditions_per_dataset": all(len(value["conditions"]) == 36 for value in combined.values()),
            "exact_10000_bootstrap_replicates": args.bootstrap_replicates == 10000,
            "zero_condition_deltas_and_intervals_are_zero": all(
                value["clean_condition"]["test_paired_auroc_difference"][key] == 0
                for value in combined.values() for key in ("noise_minus_clean_auroc", "paired_video_cluster_ci_low", "paired_video_cluster_ci_high")
            ),
        },
        "outputs": {combined_path.name: sha256(combined_path), table_path.name: sha256(table_path)},
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Complete: {combined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
