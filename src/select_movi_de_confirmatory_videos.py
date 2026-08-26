#!/usr/bin/env python3
"""Download, inventory, select, split, audit, and lock MOVi-D/E videos."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import download_movi_de_pilot as source
from movi_de_dataset_adapter import (
    camera_motion_summary,
    decode_depth,
    decode_png,
    discover_shards,
    iter_tfrecords,
    record_video_id,
    scalar,
    sha256,
    validate_record_schema,
)


VERSION = "1.0.0"
DATASETS = ("movi_d", "movi_e")
FEATURES = (
    "eligible_instance_count",
    "eligible_dynamic_instance_count",
    "mean_eligible_visibility_pixels",
    "camera_translation_scene_units",
    "camera_rotation_degrees",
)
SPLIT_COUNTS = {"train": 90, "dev": 30, "test": 30}
SELECTION_COUNT = sum(SPLIT_COUNTS.values())
MIN_MASK_AREA = 32
MIN_VALID_DEPTH = 32
MIN_VISIBILITY = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pilot-data-root", type=Path, required=True)
    parser.add_argument("--pilot-manifests-dir", type=Path, required=True)
    parser.add_argument("--output-manifests-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--split-optimization-attempts", type=int, default=120000)
    parser.add_argument("--max-split-smd", type=float, default=0.20)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stable_rank(seed: int, dataset: str, purpose: str, video_id: str) -> str:
    return hashlib.sha256(f"{seed}|{dataset}|{purpose}|{video_id}".encode()).hexdigest()


def reuse_pilot_file(item: dict[str, Any], pilot_directory: Path, destination: Path) -> bool:
    source_path = pilot_directory / Path(item["name"]).name
    target = destination / source_path.name
    if target.exists() or not source_path.is_file():
        return False
    if source_path.stat().st_size != int(item["size"]):
        return False
    if item.get("md5Hash") and source.local_md5_base64(source_path) != item["md5Hash"]:
        return False
    destination.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source_path, target)
    except OSError:
        return False
    return True


def acquire_dataset(
    dataset: str,
    data_root: Path,
    pilot_data_root: Path,
    workers: int,
) -> tuple[Path, list[dict[str, Any]]]:
    shards, metadata = source.list_dataset_files(dataset)
    destination = data_root / f"{dataset}_validation"
    pilot_directory = pilot_data_root / f"{dataset}_validation_partial"
    items = metadata + shards
    reused = 0
    for item in items:
        reused += int(reuse_pilot_file(item, pilot_directory, destination))
    print(
        f"{dataset}: official files={len(items)}, reused={reused}, "
        f"total={sum(int(item['size']) for item in items) / 2**30:.2f} GiB",
        flush=True,
    )
    downloads: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(items))) as executor:
        futures = {executor.submit(source.download, item, destination): item for item in items}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            downloads.append(result)
            print(f"{dataset}: {result['status']} {result['filename']}", flush=True)
    paths = discover_shards(destination, dataset, require_complete=True)
    if len(paths) != 16:
        raise RuntimeError(f"{dataset}: expected 16 validation shards, found {len(paths)}")
    return destination, sorted(downloads, key=lambda row: row["filename"])


def inventory_dataset(
    dataset: str,
    shard_directory: Path,
    pilot_ids: set[str],
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    shard_record_counts: Counter[str] = Counter()
    shards = discover_shards(shard_directory, dataset, require_complete=True)
    for shard, record in iter_tfrecords(shards):
        validate_record_schema(record, dataset)
        video_id = record_video_id(record)
        if video_id in seen:
            raise RuntimeError(f"{dataset}: duplicate video ID {video_id}")
        seen.add(video_id)
        record_index = shard_record_counts[shard.name]
        shard_record_counts[shard.name] += 1
        num_frames = scalar(record, "metadata/num_frames", int)
        num_instances = scalar(record, "metadata/num_instances", int)
        height = scalar(record, "metadata/height", int)
        width = scalar(record, "metadata/width", int)
        visibility = np.asarray(record["instances/visibility"], dtype=np.int64).reshape(
            num_instances, num_frames
        )
        is_dynamic = np.asarray(record["instances/is_dynamic"], dtype=bool).reshape(num_instances)
        depth_range = np.asarray(record["metadata/depth_range"], dtype=np.float64).reshape(2)
        eligible_by_instance = np.zeros(num_instances, dtype=np.int64)
        included_visibility: list[int] = []
        technical_errors: list[str] = []
        try:
            for frame_index in range(num_frames):
                rgb = decode_png(record["video"][frame_index])
                segmentation = decode_png(record["segmentations"][frame_index])
                depth = decode_depth(record["depth"][frame_index], depth_range)
                if segmentation.ndim == 3:
                    segmentation = segmentation[..., 0]
                if rgb.shape != (height, width, 3):
                    raise ValueError(f"unexpected_rgb_shape:{rgb.shape}")
                if segmentation.shape != (height, width) or depth.shape != (height, width):
                    raise ValueError("unexpected_segmentation_or_depth_shape")
                for instance_index in range(num_instances):
                    mask = segmentation == instance_index + 1
                    mask_area = int(mask.sum())
                    if visibility[instance_index, frame_index] < MIN_VISIBILITY:
                        continue
                    if mask_area < MIN_MASK_AREA:
                        continue
                    valid_depth = int(np.sum(np.isfinite(depth[mask]) & (depth[mask] > 0)))
                    if valid_depth < MIN_VALID_DEPTH:
                        continue
                    eligible_by_instance[instance_index] += 1
                    included_visibility.append(int(visibility[instance_index, frame_index]))
        except Exception as exc:  # preserve the video-level failure in the inventory
            technical_errors.append(f"unreadable_or_invalid_required_data:{type(exc).__name__}:{exc}")

        eligible_instances = eligible_by_instance >= 2
        eligible_instance_count = int(eligible_instances.sum())
        reasons: list[str] = []
        if technical_errors:
            reasons.extend(technical_errors)
        if eligible_instance_count < 2:
            reasons.append("fewer_than_two_instances_with_two_eligible_observations")
        if video_id in pilot_ids:
            reasons.append("pilot_video_permanently_excluded")
        positions = np.asarray(record["camera/positions"], dtype=np.float64).reshape(num_frames, 3)
        quaternions = np.asarray(record["camera/quaternions"], dtype=np.float64).reshape(num_frames, 4)
        motion = camera_motion_summary(positions, quaternions)
        rows.append(
            {
                "dataset": dataset,
                "video_id": video_id,
                "source_shard": shard.name,
                "source_record_index": record_index,
                "num_frames": num_frames,
                "num_instances": num_instances,
                "eligible_instance_count": eligible_instance_count,
                "eligible_dynamic_instance_count": int(np.sum(is_dynamic & eligible_instances)),
                "eligible_static_instance_count": int(np.sum((~is_dynamic) & eligible_instances)),
                "eligible_observation_count": int(eligible_by_instance.sum()),
                "mean_eligible_visibility_pixels": (
                    float(np.mean(included_visibility)) if included_visibility else 0.0
                ),
                "camera_translation_scene_units": (
                    0.0 if dataset == "movi_d" else motion["translation_start_to_end_scene_units"]
                ),
                "camera_rotation_degrees": (
                    0.0 if dataset == "movi_d" else motion["rotation_start_to_end_degrees"]
                ),
                "normalized_camera_translation": (
                    0.0 if dataset == "movi_d" else motion["normalized_start_to_end_translation"]
                ),
                "pilot_video": video_id in pilot_ids,
                "confirmatory_eligible": not reasons,
                "exclusion_reasons": reasons,
                "seeded_tie_break": stable_rank(seed, dataset, "inventory", video_id),
            }
        )
        if len(rows) % 25 == 0:
            print(f"{dataset}: inventoried {len(rows)}/250 videos", flush=True)
    if len(rows) != 250:
        raise RuntimeError(f"{dataset}: expected 250 videos, inventoried {len(rows)}")
    if not pilot_ids <= seen:
        raise RuntimeError(f"{dataset}: pilot IDs missing from full inventory: {sorted(pilot_ids - seen)}")
    return sorted(rows, key=lambda row: row["video_id"])


def feature_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([[float(row[field]) for field in FEATURES] for row in rows], dtype=np.float64)


def active_feature_indices(matrix: np.ndarray) -> list[int]:
    return [index for index in range(matrix.shape[1]) if float(np.std(matrix[:, index])) > 1e-12]


def marginal_bins(matrix: np.ndarray, active: list[int]) -> dict[int, np.ndarray]:
    bins: dict[int, np.ndarray] = {}
    for index in active:
        edges = np.unique(np.quantile(matrix[:, index], [0.2, 0.4, 0.6, 0.8]))
        bins[index] = np.searchsorted(edges, matrix[:, index], side="right")
    return bins


def distribution_objective(
    subset_indices: np.ndarray,
    matrix: np.ndarray,
    active: list[int],
    bins: dict[int, np.ndarray],
) -> float:
    reference = matrix[:, active]
    selected = matrix[subset_indices][:, active]
    scale = np.std(reference, axis=0)
    mean_term = float(np.sum(((np.mean(selected, axis=0) - np.mean(reference, axis=0)) / scale) ** 2))
    std_term = float(np.sum(((np.std(selected, axis=0) - scale) / scale) ** 2))
    marginal_term = 0.0
    fraction = len(subset_indices) / len(matrix)
    for index in active:
        full_counts = np.bincount(bins[index])
        selected_counts = np.bincount(bins[index][subset_indices], minlength=len(full_counts))
        expected = full_counts * fraction
        marginal_term += float(np.sum(((selected_counts - expected) / np.maximum(expected, 1.0)) ** 2))
    return 20.0 * mean_term + 2.0 * std_term + marginal_term


def select_balanced(rows: list[dict[str, Any]], count: int) -> list[int]:
    matrix = feature_matrix(rows)
    active = active_feature_indices(matrix)
    bins = marginal_bins(matrix, active)
    selected = list(range(len(rows)))
    while len(selected) > count:
        best_position = None
        best_key = None
        for position, index in enumerate(selected):
            proposed = np.asarray(selected[:position] + selected[position + 1 :], dtype=np.int64)
            key = (
                distribution_objective(proposed, matrix, active, bins),
                rows[index]["seeded_tie_break"],
            )
            if best_key is None or key < best_key:
                best_position, best_key = position, key
        assert best_position is not None
        selected.pop(best_position)
    return selected


def standardized_mean_differences(
    reference_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
) -> dict[str, float | None]:
    reference = feature_matrix(reference_rows)
    comparison = feature_matrix(comparison_rows)
    result: dict[str, float | None] = {}
    for index, field in enumerate(FEATURES):
        pooled = math.sqrt(
            (float(np.var(reference[:, index], ddof=1)) + float(np.var(comparison[:, index], ddof=1))) / 2
        )
        result[field] = (
            abs(float(np.mean(comparison[:, index]) - np.mean(reference[:, index]))) / pooled
            if pooled > 1e-12
            else None
        )
    return result


def split_objective(assignments: list[str], rows: list[dict[str, Any]]) -> float:
    total = 0.0
    for split in SPLIT_COUNTS:
        subset = [row for row, assignment in zip(rows, assignments, strict=True) if assignment == split]
        smd = standardized_mean_differences(rows, subset)
        total += sum(value * value for value in smd.values() if value is not None)
    return total


def assign_balanced_splits(
    rows: list[dict[str, Any]], dataset: str, seed: int, attempts: int
) -> list[str]:
    order = sorted(range(len(rows)), key=lambda i: stable_rank(seed, dataset, "split", rows[i]["video_id"]))
    cycle = ["train", "train", "train", "dev", "test"]
    assignments = [""] * len(rows)
    for position, index in enumerate(order):
        assignments[index] = cycle[position % len(cycle)]
    if Counter(assignments) != Counter(SPLIT_COUNTS):
        raise RuntimeError(f"{dataset}: initial split allocation failed")
    rng = random.Random(int(stable_rank(seed, dataset, "split-optimization", "all")[:16], 16))
    current = split_objective(assignments, rows)
    for _ in range(attempts):
        left, right = rng.sample(range(len(rows)), 2)
        if assignments[left] == assignments[right]:
            continue
        assignments[left], assignments[right] = assignments[right], assignments[left]
        proposed = split_objective(assignments, rows)
        if proposed <= current:
            current = proposed
        else:
            assignments[left], assignments[right] = assignments[right], assignments[left]
    return assignments


def summarize_fields(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    matrix = feature_matrix(rows)
    return {
        field: {
            "minimum": float(np.min(matrix[:, index])),
            "mean": float(np.mean(matrix[:, index])),
            "median": float(np.median(matrix[:, index])),
            "maximum": float(np.max(matrix[:, index])),
        }
        for index, field in enumerate(FEATURES)
    }


def lock_dataset(
    dataset: str,
    inventory: list[dict[str, Any]],
    output_dir: Path,
    seed: int,
    attempts: int,
    max_split_smd: float,
) -> dict[str, Any]:
    eligible = [row for row in inventory if row["confirmatory_eligible"]]
    if len(eligible) < SELECTION_COUNT:
        raise RuntimeError(f"{dataset}: only {len(eligible)} eligible non-pilot videos")
    selected_indices = select_balanced(eligible, SELECTION_COUNT)
    selected = [eligible[index].copy() for index in selected_indices]
    assignments = assign_balanced_splits(selected, dataset, seed, attempts)
    for row, split in zip(selected, assignments, strict=True):
        row["split"] = split
        row["seed"] = seed
        row["selection_rule"] = (
            "deterministic distribution-balanced subsample and split assignment over eligible object count, "
            "dynamic-object count, mean eligible visibility, camera translation, and camera rotation"
        )
    selected.sort(key=lambda row: (row["split"], row["video_id"]))
    counts = Counter(row["split"] for row in selected)
    if counts != Counter(SPLIT_COUNTS):
        raise RuntimeError(f"{dataset}: wrong split counts {counts}")
    if any(row["pilot_video"] for row in selected):
        raise RuntimeError(f"{dataset}: pilot leakage into confirmatory selection")
    if len({row["video_id"] for row in selected}) != SELECTION_COUNT:
        raise RuntimeError(f"{dataset}: duplicate selected video ID")

    selection_smd = standardized_mean_differences(eligible, selected)
    split_smd: dict[str, dict[str, float | None]] = {}
    for split in SPLIT_COUNTS:
        pool = [row for row in selected if row["split"] == split]
        split_smd[split] = standardized_mean_differences(selected, pool)
    active_values = [
        value for values in split_smd.values() for value in values.values() if value is not None
    ]
    max_observed_smd = max(active_values, default=0.0)
    if max_observed_smd > max_split_smd:
        raise RuntimeError(
            f"{dataset}: split balance max SMD {max_observed_smd:.4f} exceeds {max_split_smd:.4f}"
        )

    inventory_path = output_dir / f"confirmatory_{dataset}_inventory_250.jsonl"
    combined_path = output_dir / f"confirmatory_{dataset}_150.jsonl"
    write_jsonl(inventory_path, inventory)
    write_jsonl(combined_path, selected)
    pool_paths: dict[str, Path] = {}
    for split in SPLIT_COUNTS:
        path = output_dir / f"confirmatory_{dataset}_{split}_{SPLIT_COUNTS[split]}.jsonl"
        write_jsonl(path, [row for row in selected if row["split"] == split])
        pool_paths[split] = path
    audit = {
        "dataset": dataset,
        "seed": seed,
        "inventory_videos": len(inventory),
        "pilot_videos_excluded": sum(row["pilot_video"] for row in inventory),
        "confirmatory_eligible_nonpilot_videos": len(eligible),
        "selected_videos": len(selected),
        "split_counts": dict(sorted(counts.items())),
        "selection_balance_smd_vs_all_eligible": selection_smd,
        "split_balance_smd_vs_selected": split_smd,
        "maximum_active_split_smd": max_observed_smd,
        "maximum_allowed_split_smd": max_split_smd,
        "feature_summaries": {
            "all_eligible_nonpilot": summarize_fields(eligible),
            "selected": summarize_fields(selected),
            **{
                split: summarize_fields([row for row in selected if row["split"] == split])
                for split in SPLIT_COUNTS
            },
        },
        "checks": {
            "exactly_250_inventory_videos": len(inventory) == 250,
            "exactly_20_pilot_videos_excluded": sum(row["pilot_video"] for row in inventory) == 20,
            "all_selected_confirmatory_eligible": all(row["confirmatory_eligible"] for row in selected),
            "no_pilot_selected": not any(row["pilot_video"] for row in selected),
            "unique_selected_video_ids": len({row["video_id"] for row in selected}) == 150,
            "exact_90_30_30_split": counts == Counter(SPLIT_COUNTS),
            "split_balance_gate_passed": max_observed_smd <= max_split_smd,
        },
        "outputs": {
            "inventory": {"path": str(inventory_path), "sha256": sha256(inventory_path)},
            "combined": {"path": str(combined_path), "sha256": sha256(combined_path)},
            "pools": {
                split: {"path": str(path), "sha256": sha256(path)}
                for split, path in pool_paths.items()
            },
        },
    }
    audit_path = output_dir / f"confirmatory_{dataset}_selection_audit.json"
    write_json(audit_path, audit)
    audit["outputs"]["audit"] = {"path": str(audit_path), "sha256": sha256(audit_path)}
    return audit


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.split_optimization_attempts < 1:
        raise ValueError("workers and split optimization attempts must be positive")
    args.data_root.mkdir(parents=True, exist_ok=True)
    args.output_manifests_dir.mkdir(parents=True, exist_ok=True)
    overall: dict[str, Any] = {
        "pipeline": "MOVi-D/E confirmatory video selection and split lock",
        "version": VERSION,
        "protocol_id": "MOVI-DE-POSE-001",
        "protocol_version": "0.2",
        "seed": args.seed,
        "selection_count_per_dataset": SELECTION_COUNT,
        "split_counts": SPLIT_COUNTS,
        "balance_features": list(FEATURES),
        "eligibility_thresholds": {
            "minimum_visibility_pixels": MIN_VISIBILITY,
            "minimum_mask_area_pixels": MIN_MASK_AREA,
            "minimum_valid_depth_pixels": MIN_VALID_DEPTH,
            "minimum_instances_with_two_eligible_observations": 2,
        },
        "datasets": {},
    }
    for dataset in DATASETS:
        pilot_path = args.pilot_manifests_dir / f"pilot_{dataset}_20.jsonl"
        pilot_ids = {str(row["video_id"]) for row in load_jsonl(pilot_path)}
        if len(pilot_ids) != 20:
            raise RuntimeError(f"{dataset}: expected 20 locked pilot IDs")
        shard_directory, downloads = acquire_dataset(
            dataset, args.data_root, args.pilot_data_root, args.workers
        )
        inventory = inventory_dataset(dataset, shard_directory, pilot_ids, args.seed)
        audit = lock_dataset(
            dataset,
            inventory,
            args.output_manifests_dir,
            args.seed,
            args.split_optimization_attempts,
            args.max_split_smd,
        )
        overall["datasets"][dataset] = {
            "source_directory": str(shard_directory.resolve()),
            "source_files": [
                {key: value for key, value in row.items() if key != "status"}
                for row in downloads
            ],
            "pilot_manifest": {"path": str(pilot_path), "sha256": sha256(pilot_path)},
            "selection_audit": audit,
        }
    d_ids = {
        row["video_id"]
        for row in load_jsonl(args.output_manifests_dir / "confirmatory_movi_d_150.jsonl")
    }
    e_ids = {
        row["video_id"]
        for row in load_jsonl(args.output_manifests_dir / "confirmatory_movi_e_150.jsonl")
    }
    overall["cross_dataset_note"] = (
        "MOVi-D and MOVi-E are independent samples; coincident numeric video IDs do not imply paired scenes."
    )
    overall["coincident_numeric_video_ids_across_datasets"] = len(d_ids & e_ids)
    lock_path = args.output_manifests_dir / "confirmatory_video_pool_lock.json"
    write_json(lock_path, overall)
    print(f"Complete: {lock_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
