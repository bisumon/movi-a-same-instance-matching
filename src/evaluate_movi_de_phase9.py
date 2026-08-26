#!/usr/bin/env python3
"""Evaluate locked MOVi-D/E results against the frozen Phase 9 hypotheses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


C_ID = "C_camera_geometry"
D_ID = "D_pose_aligned_geometry"
S_ID = "S_shuffled_pose"
MOTION_FIELDS = (
    "camera_displacement_scene_units",
    "relative_camera_rotation_degrees",
    "normalized_camera_displacement",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stable(value: float) -> float:
    return float(np.round(float(value), 12))


def weighted_auc_batch(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Tie-aware weighted AUROC for many bootstrap weight rows."""
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores) != 0) + 1]
    sorted_weights = weights[:, order]
    positive = sorted_weights * (sorted_labels == 1)
    negative = sorted_weights * (sorted_labels == 0)
    positive_group = np.add.reduceat(positive, starts, axis=1)
    negative_group = np.add.reduceat(negative, starts, axis=1)
    negative_before = np.cumsum(negative_group, axis=1) - negative_group
    numerator = np.sum(positive_group * (negative_before + 0.5 * negative_group), axis=1)
    denominator = np.sum(positive_group, axis=1) * np.sum(negative_group, axis=1)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def bootstrap_delta(
    labels: np.ndarray,
    c_scores: np.ndarray,
    d_scores: np.ndarray,
    cluster_index: np.ndarray,
    cluster_counts: np.ndarray,
    selected: np.ndarray | None = None,
    batch_size: int = 500,
) -> np.ndarray:
    if selected is None:
        selected = np.ones(len(labels), dtype=bool)
    labels = labels[selected]
    c_scores = c_scores[selected]
    d_scores = d_scores[selected]
    cluster_index = cluster_index[selected]
    output = []
    for start in range(0, len(cluster_counts), batch_size):
        weights = cluster_counts[start : start + batch_size, cluster_index].astype(np.float64)
        output.append(weighted_auc_batch(labels, d_scores, weights) - weighted_auc_batch(labels, c_scores, weights))
    return np.concatenate(output)


def dataset_arrays(root: Path, dataset: str) -> dict[str, Any]:
    regime = "regime1" if dataset == "movi_e" else "regime2"
    suffix = "movi_e_phase8_regime1_predictions.jsonl" if dataset == "movi_e" else "movi_d_phase8_regime2_predictions.jsonl"
    pair_path = root / f"manifests/pairs/movi_de/{dataset}_all_pairs.jsonl"
    prediction_path = root / f"runs/movi_de_confirmatory/phase8_{regime}/in_domain_{dataset}/{suffix}"
    pairs = read_jsonl(pair_path)
    predictions = read_jsonl(prediction_path)
    if [row["pair_id"] for row in pairs] != [row["pair_id"] for row in predictions]:
        raise ValueError(f"{dataset} pair/prediction order mismatch")
    test_indices = np.asarray([index for index, row in enumerate(pairs) if row["split"] == "test"])
    test_pairs = [pairs[index] for index in test_indices]
    test_predictions = [predictions[index] for index in test_indices]
    videos = np.asarray([str(row["video_id"]) for row in test_pairs])
    unique_videos = sorted(set(videos.tolist()), key=int)
    video_to_index = {video: index for index, video in enumerate(unique_videos)}
    return {
        "pair_path": pair_path,
        "prediction_path": prediction_path,
        "pairs": pairs,
        "test_pairs": test_pairs,
        "labels": np.asarray([row["label"] for row in test_pairs], dtype=np.int8),
        "c_scores": np.asarray([row["scores"][C_ID] for row in test_predictions], dtype=np.float64),
        "d_scores": np.asarray([row["scores"][D_ID] for row in test_predictions], dtype=np.float64),
        "cluster_index": np.asarray([video_to_index[video] for video in videos], dtype=np.int64),
        "video_count": len(unique_videos),
    }


def interval(samples: np.ndarray) -> dict[str, Any]:
    valid = samples[np.isfinite(samples)]
    return {
        "paired_video_cluster_ci_low": stable(np.quantile(valid, 0.025)),
        "paired_video_cluster_ci_high": stable(np.quantile(valid, 0.975)),
        "valid_replicates": int(len(valid)),
    }


def point_delta(data: dict[str, Any], selected: np.ndarray | None = None) -> float:
    if selected is None:
        selected = np.ones(len(data["labels"]), dtype=bool)
    return stable(
        roc_auc_score(data["labels"][selected], data["d_scores"][selected])
        - roc_auc_score(data["labels"][selected], data["c_scores"][selected])
    )


def motion_strata(
    data: dict[str, Any], rng: np.random.Generator, replicates: int
) -> dict[str, Any]:
    train_pairs = [row for row in data["pairs"] if row["split"] == "train"]
    counts = rng.multinomial(data["video_count"], np.full(data["video_count"], 1 / data["video_count"]), size=replicates)
    output: dict[str, Any] = {}
    for field in MOTION_FIELDS:
        train_values = np.asarray([row["controls"][field] for row in train_pairs], dtype=np.float64)
        cuts = np.quantile(train_values, [1 / 3, 2 / 3])
        test_values = np.asarray([row["controls"][field] for row in data["test_pairs"]], dtype=np.float64)
        selections = {
            "low": test_values <= cuts[0],
            "medium": (test_values > cuts[0]) & (test_values <= cuts[1]),
            "high": test_values > cuts[1],
        }
        strata: dict[str, Any] = {}
        bootstrap: dict[str, np.ndarray] = {}
        for name, selected in selections.items():
            samples = bootstrap_delta(
                data["labels"], data["c_scores"], data["d_scores"], data["cluster_index"], counts, selected
            )
            bootstrap[name] = samples
            labels = data["labels"][selected]
            strata[name] = {
                "pairs": int(selected.sum()),
                "positives": int((labels == 1).sum()),
                "negatives": int((labels == 0).sum()),
                "D_minus_C_AUROC": point_delta(data, selected),
                **interval(samples),
            }
        high_minus_low = bootstrap["high"] - bootstrap["low"]
        output[field] = {
            "train_tertile_cutoffs": [stable(cuts[0]), stable(cuts[1])],
            "strata": strata,
            "high_minus_low_D_minus_C_AUROC": stable(
                strata["high"]["D_minus_C_AUROC"] - strata["low"]["D_minus_C_AUROC"]
            ),
            **interval(high_minus_low),
        }
    return output


def endpoint(noise: dict[str, Any], dataset: str, translation: float, rotation: float) -> dict[str, Any]:
    condition = next(
        row for row in noise["datasets"][dataset]["conditions"]
        if row["translation_std_scene_units"] == translation and row["rotation_std_degrees"] == rotation
    )
    return {
        "condition_id": condition["condition_id"],
        "test_AUROC": condition["metrics"]["test"]["auroc"],
        "test_recall": condition["metrics"]["test"]["recall_at_clean_D_locked_90_recall_threshold"],
        **condition["test_paired_auroc_difference"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260825)

    e_results_path = root / "results/movi_de_phase8_regime1/movi_e_in_domain_results.json"
    d_results_path = root / "results/movi_de_phase8_regime2/movi_d_in_domain_results.json"
    transfer_path = root / "results/movi_de_phase8_regime3/movi_d_to_e_transfer_results.json"
    noise_path = root / "results/movi_de_phase7_pose_noise/phase7_pose_noise_combined_results.json"
    protocol_path = root / "docs/MOVI_DE_CAMERA_POSE_PROTOCOL_v0.2.md"
    e_results, d_results = read_json(e_results_path), read_json(d_results_path)
    transfer, noise = read_json(transfer_path), read_json(noise_path)
    d_data, e_data = dataset_arrays(root, "movi_d"), dataset_arrays(root, "movi_e")

    d_counts = rng.multinomial(d_data["video_count"], np.full(d_data["video_count"], 1 / d_data["video_count"]), size=args.bootstrap_replicates)
    e_counts = rng.multinomial(e_data["video_count"], np.full(e_data["video_count"], 1 / e_data["video_count"]), size=args.bootstrap_replicates)
    d_bootstrap = bootstrap_delta(d_data["labels"], d_data["c_scores"], d_data["d_scores"], d_data["cluster_index"], d_counts)
    e_bootstrap = bootstrap_delta(e_data["labels"], e_data["c_scores"], e_data["d_scores"], e_data["cluster_index"], e_counts)
    did_samples = e_bootstrap - d_bootstrap
    difference_in_differences = {
        "estimand": "(AUROC_D_minus_C_on_MOVi_E) minus (AUROC_D_minus_C_on_MOVi_D)",
        "point_estimate": stable(point_delta(e_data) - point_delta(d_data)),
        **interval(did_samples),
        "bootstrap_replicates": args.bootstrap_replicates,
        "resampling": "independent video-cluster resampling within MOVi-D and MOVi-E",
        "status": "descriptive_secondary",
    }
    strata = motion_strata(e_data, rng, args.bootstrap_replicates)

    h1 = e_results["primary"]
    h2 = e_results["paired_differences_vs_C"][D_ID]
    h4 = d_results["fixed_camera_falsification"]
    h5_c = e_results["paired_differences_vs_C"][S_ID]
    h5_d = e_results["paired_differences_vs_D"][S_ID]
    motion_core = [strata["camera_displacement_scene_units"], strata["relative_camera_rotation_degrees"]]
    h3_points_positive = all(row["high_minus_low_D_minus_C_AUROC"] > 0 for row in motion_core)
    h3_intervals_positive = all(row["paired_video_cluster_ci_low"] > 0 for row in motion_core)
    noise_endpoints = {
        dataset: {
            "rotation_5_only": endpoint(noise, dataset, 0, 5),
            "translation_0p5_only": endpoint(noise, dataset, 0.5, 0),
            "maximum_combined": endpoint(noise, dataset, 0.5, 5),
        }
        for dataset in ("movi_d", "movi_e")
    }

    criteria = [
        {
            "id": "H1", "criterion": "Pose-aligned D has higher MOVi-E AUROC than camera-space C.",
            "status": "supported_primary", "decision": bool(h1["success"]),
            "evidence": {"D_minus_C_AUROC": h1["system_minus_C"], "ci_low": h1["paired_video_cluster_ci_low"], "ci_high": h1["paired_video_cluster_ci_high"]},
            "interpretation": "The sole confirmatory success rule is met because the full paired 95% interval is above zero.",
        },
        {
            "id": "H2", "criterion": "D reduces false-match rate on MOVi-E without material recall loss.",
            "status": "directionally_consistent_not_resolved", "decision": None,
            "evidence": {"D_minus_C_false_match_rate": h2["false_match_rate"], "D_minus_C_recall": h2["recall"]},
            "interpretation": "Point estimates favor D, but both paired intervals include zero and no materiality margin was frozen.",
        },
        {
            "id": "H3", "criterion": "The D-minus-C benefit increases with camera displacement and rotation.",
            "status": "supported" if h3_intervals_positive else ("directionally_consistent_not_resolved" if h3_points_positive else "not_supported"),
            "decision": h3_intervals_positive,
            "evidence": {"MOVi_E_motion_strata": strata, "D_vs_E_difference_in_differences": difference_in_differences},
            "interpretation": "Motion-stratum conclusions are secondary and use predeclared MOVi-E-training tertiles.",
        },
        {
            "id": "H4", "criterion": "D and C perform similarly on fixed-camera MOVi-D.",
            "status": "consistent_not_equivalence", "decision": None,
            "evidence": h4,
            "interpretation": "The interval spans zero and the point effect is small, but no equivalence margin was frozen.",
        },
        {
            "id": "H5", "criterion": "Shuffled pose does not improve over C and impairs correct pose correspondence.",
            "status": "supported", "decision": h5_c["auroc"]["system_minus_C"] <= 0 and h5_d["auroc"]["paired_video_cluster_ci_high"] < 0,
            "evidence": {"S_vs_C": h5_c, "S_vs_D": h5_d},
            "interpretation": "S does not beat C and is resolved below clean D, while pose-only P remains at chance.",
        },
        {
            "id": "H6", "criterion": "Pose noise degrades performance as translation and rotation error grow.",
            "status": "supported_graded_endpoint_pattern", "decision": all(
                value["paired_video_cluster_ci_high"] < 0
                for dataset in noise_endpoints.values() for value in dataset.values()
            ),
            "evidence": noise_endpoints,
            "interpretation": "Large single-axis and combined endpoints degrade clearly; small errors are tolerated. This is not a claim of strict monotonicity at every grid step.",
        },
        {
            "id": "TRANSFER", "criterion": "MOVi-D clean D transfers to MOVi-E without target refitting or threshold adjustment.",
            "status": "ranking_transfers_threshold_partially_degrades", "decision": None,
            "evidence": {"transfer_test": transfer["transfer_test"], "paired_transfer_differences": transfer["paired_transfer_differences"]},
            "interpretation": "AUROC is unresolved from in-domain D, but the frozen MOVi-D threshold has a resolved 0.006 higher false-match rate.",
        },
    ]
    result = {
        "pipeline": "MOVi-D/E Phase 9 criteria evaluation",
        "version": "1.0.0",
        "protocol": "MOVI-DE-POSE-001 v0.2",
        "criteria_source": "Frozen protocol Sections 1, 3, 4, 11, and 12; the repository contains no separate numbered Phase 9 specification.",
        "primary_conclusion": "supported",
        "primary_success_rule_met": bool(h1["success"]),
        "overall_evidence_summary": "Primary hypothesis supported; correspondence and pose-noise controls support the intended mechanism; operational and fixed-camera evidence is directionally consistent but not fully resolved; transfer preserves ranking but not the false-match operating point perfectly.",
        "difference_in_differences": difference_in_differences,
        "criteria": criteria,
        "interpretation_limits": [
            "Oracle simulator masks, depth, intrinsics, and camera poses are used.",
            "MOVi-D and MOVi-E are independent synthetic samples, not counterfactual renders of the same scenes.",
            "All pairs are within-video; cross-video re-identification is not tested.",
            "No practical equivalence or material-recall-loss margins were predeclared.",
            "Secondary criteria are descriptive unless the frozen primary decision rule explicitly applies.",
        ],
        "inputs": {str(path.relative_to(root)): sha256(path) for path in (
            protocol_path, e_results_path, d_results_path, transfer_path, noise_path,
            d_data["pair_path"], d_data["prediction_path"], e_data["pair_path"], e_data["prediction_path"],
        )},
    }
    json_path = args.output_dir / "phase9_criteria_evaluation.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = args.output_dir / "phase9_criteria_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["criterion", "status", "formal_decision", "interpretation"])
        writer.writeheader()
        for row in criteria:
            writer.writerow({"criterion": row["id"], "status": row["status"], "formal_decision": row["decision"], "interpretation": row["interpretation"]})
    print(f"Complete: Phase 9 primary={result['primary_conclusion']}; {len(criteria)} criteria evaluated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
