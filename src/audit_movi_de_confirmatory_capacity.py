#!/usr/bin/env python3
"""Run the frozen post-split pair-capacity gate for MOVi-D/E pools."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from movi_de_dataset_adapter import (
    decode_depth,
    decode_png,
    discover_shards,
    iter_tfrecords,
    record_video_id,
    scalar,
    sha256,
    text_at,
    validate_record_schema,
)


TARGETS = {
    "train": {"positive": 3000, "hard": 1500, "easy": 1500},
    "dev": {"positive": 1000, "hard": 500, "easy": 500},
    "test": {"positive": 1000, "hard": 500, "easy": 500},
}


@dataclass(frozen=True, slots=True)
class Observation:
    frame: int
    instance: int
    mask_area: int
    category: str
    asset_id: str
    is_dynamic: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("movi_d", "movi_e"), required=True)
    parser.add_argument("--tfrecord-dir", type=Path, required=True)
    parser.add_argument("--video-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def eligible_observations(record: dict[str, Any]) -> list[Observation]:
    num_frames = scalar(record, "metadata/num_frames", int)
    num_instances = scalar(record, "metadata/num_instances", int)
    visibility = np.asarray(record["instances/visibility"], dtype=np.int64).reshape(
        num_instances, num_frames
    )
    depth_range = np.asarray(record["metadata/depth_range"], dtype=np.float64).reshape(2)
    categories = [text_at(record["instances/category"], index) for index in range(num_instances)]
    assets = [text_at(record["instances/asset_id"], index) for index in range(num_instances)]
    dynamic = np.asarray(record["instances/is_dynamic"], dtype=bool).reshape(num_instances)
    observations: list[Observation] = []
    for frame in range(num_frames):
        segmentation = decode_png(record["segmentations"][frame])
        depth = decode_depth(record["depth"][frame], depth_range)
        if segmentation.ndim == 3:
            segmentation = segmentation[..., 0]
        for instance in range(num_instances):
            mask = segmentation == instance + 1
            area = int(mask.sum())
            if visibility[instance, frame] < 32 or area < 32:
                continue
            if int(np.sum(np.isfinite(depth[mask]) & (depth[mask] > 0))) < 32:
                continue
            observations.append(
                Observation(frame, instance, area, categories[instance], assets[instance], bool(dynamic[instance]))
            )
    return observations


def pairs(observations: list[Observation]) -> Iterable[tuple[Observation, Observation]]:
    ordered = sorted(observations, key=lambda row: (row.frame, row.instance))
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            if left.frame != right.frame:
                yield left, right


def gap_bin(gap: int) -> str:
    if gap <= 5:
        return "short_1_5"
    if gap <= 11:
        return "medium_6_11"
    return "long_12_23"


def main() -> int:
    args = parse_args()
    manifest_rows = load_jsonl(args.video_manifest)
    selected = {str(row["video_id"]): str(row["split"]) for row in manifest_rows}
    if len(selected) != 150 or Counter(selected.values()) != Counter({"train": 90, "dev": 30, "test": 30}):
        raise ValueError("Video manifest is not an exact unique 90/30/30 split")
    observations: dict[str, list[Observation]] = {}
    for _, record in iter_tfrecords(discover_shards(args.tfrecord_dir, args.dataset, require_complete=True)):
        video_id = record_video_id(record)
        if video_id not in selected:
            continue
        validate_record_schema(record, args.dataset)
        observations[video_id] = eligible_observations(record)
        if len(observations) % 25 == 0:
            print(f"{args.dataset}: capacity observations {len(observations)}/150", flush=True)
    if observations.keys() != selected.keys():
        raise RuntimeError(f"Missing selected videos: {sorted(selected.keys() - observations.keys())}")

    train_ratios: list[float] = []
    for video_id, video_observations in observations.items():
        if selected[video_id] != "train":
            continue
        for left, right in pairs(video_observations):
            if left.instance != right.instance and left.category == right.category:
                train_ratios.append(abs(math.log(left.mask_area / right.mask_area)))
    if not train_ratios:
        raise RuntimeError("No train same-category negative candidates")
    cutoff = float(np.quantile(np.asarray(train_ratios), 0.25, method="higher"))

    counts = {split: Counter() for split in TARGETS}
    gap_counts = {split: {kind: Counter() for kind in ("positive", "hard", "easy")} for split in TARGETS}
    videos_with = {split: {kind: set() for kind in ("positive", "hard", "easy")} for split in TARGETS}
    motion_types = {split: {kind: Counter() for kind in ("positive", "hard", "easy")} for split in TARGETS}
    same_category_candidates = Counter()
    very_hard = Counter()
    for video_id, video_observations in observations.items():
        split = selected[video_id]
        for left, right in pairs(video_observations):
            if left.instance == right.instance:
                kind = "positive"
            elif left.category != right.category:
                kind = "easy"
            else:
                same_category_candidates[split] += 1
                if abs(math.log(left.mask_area / right.mask_area)) <= cutoff:
                    kind = "hard"
                    if left.asset_id == right.asset_id:
                        very_hard[split] += 1
                else:
                    continue
            counts[split][kind] += 1
            videos_with[split][kind].add(video_id)
            gap_counts[split][kind][gap_bin(abs(left.frame - right.frame))] += 1
            if kind == "positive":
                motion = "dynamic" if left.is_dynamic else "static"
            else:
                motion = "-".join(sorted(("dynamic" if left.is_dynamic else "static", "dynamic" if right.is_dynamic else "static")))
            motion_types[split][kind][motion] += 1

    capacity = {}
    failures = []
    for split, targets in TARGETS.items():
        capacity[split] = {}
        for kind, target in targets.items():
            available = counts[split][kind]
            passed = available >= target
            if not passed:
                failures.append(f"{split}:{kind}:{available}<{target}")
            capacity[split][kind] = {
                "target": target,
                "available_unique_candidates": available,
                "capacity_to_target_ratio": available / target,
                "videos_with_candidates": len(videos_with[split][kind]),
                "temporal_gap_capacity": dict(sorted(gap_counts[split][kind].items())),
                "object_motion_capacity": dict(sorted(motion_types[split][kind].items())),
                "passed": passed,
            }
            if kind == "hard":
                capacity[split][kind]["same_category_candidates_before_scale_filter"] = (
                    same_category_candidates[split]
                )
                capacity[split][kind]["retained_fraction_of_same_category_candidates"] = (
                    available / same_category_candidates[split]
                    if same_category_candidates[split]
                    else 0.0
                )
    train_retained_fraction = (
        counts["train"]["hard"] / same_category_candidates["train"]
        if same_category_candidates["train"]
        else 0.0
    )
    retention_gate_passed = train_retained_fraction >= 0.25
    if not retention_gate_passed:
        failures.append(f"train_hard_retention:{train_retained_fraction}<0.25")
    report = {
        "pipeline": "MOVi-D/E frozen post-split pair-capacity audit",
        "dataset": args.dataset,
        "protocol_id": "MOVI-DE-POSE-001",
        "protocol_version": "0.2",
        "video_manifest": {"path": str(args.video_manifest), "sha256": sha256(args.video_manifest)},
        "split_counts": dict(sorted(Counter(selected.values()).items())),
        "hard_negative_scale_cutoff_abs_log_area_ratio": cutoff,
        "hard_negative_scale_cutoff_equivalent_max_area_ratio": math.exp(cutoff),
        "hard_negative_cutoff_source": "locked training pool only; applied unchanged to development and test",
        "hard_negative_cutoff_quantile": 0.25,
        "hard_negative_cutoff_quantile_method": "higher (smallest observed cutoff retaining at least 25%)",
        "train_same_category_candidates_before_scale_filter": same_category_candidates["train"],
        "train_same_category_candidates_retained": counts["train"]["hard"],
        "train_same_category_retained_fraction": train_retained_fraction,
        "capacity": capacity,
        "very_hard_same_asset_candidates": dict(sorted(very_hard.items())),
        "checks": {
            "all_150_selected_videos_found": len(observations) == 150,
            "exact_90_30_30_split": Counter(selected.values()) == Counter({"train": 90, "dev": 30, "test": 30}),
            "train_cutoff_retains_at_least_25_percent": retention_gate_passed,
            "all_pool_kind_capacity_gates_passed": not failures,
        },
        "failures": failures,
        "status": "pass" if not failures else "fail",
    }
    write_json(args.output, report)
    if failures:
        raise RuntimeError(f"Capacity gate failed: {failures}")
    print(f"Complete: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
