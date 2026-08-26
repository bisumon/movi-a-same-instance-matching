#!/usr/bin/env python3
"""Generate and audit the frozen Phase 5 MOVi-D/E pair manifests."""

from __future__ import annotations

import argparse
import hashlib
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
    relative_rotation_degrees,
    scalar,
    sha256,
    stable_observation_id,
    text_at,
    validate_record_schema,
)


VERSION = "1.0.0"
TARGETS = {
    "train": {"positive": 3000, "hard": 1500, "easy": 1500},
    "dev": {"positive": 1000, "hard": 500, "easy": 500},
    "test": {"positive": 1000, "hard": 500, "easy": 500},
}
GAP_BINS = ("short_1_5", "medium_6_11", "long_12_23")


@dataclass(frozen=True, slots=True)
class Observation:
    observation_id: str
    frame: int
    instance: int
    visibility: int
    mask_area: int
    valid_depth_pixels: int
    category: str
    asset_id: str
    is_dynamic: bool


@dataclass(frozen=True, slots=True)
class Candidate:
    dataset: str
    split: str
    video_id: str
    kind: str
    left: Observation
    right: Observation
    camera_displacement: float
    camera_rotation: float
    normalized_camera_displacement: float

    @property
    def gap(self) -> int:
        return self.right.frame - self.left.frame

    @property
    def gap_bin(self) -> str:
        if self.gap <= 5:
            return GAP_BINS[0]
        if self.gap <= 11:
            return GAP_BINS[1]
        return GAP_BINS[2]

    @property
    def unordered_key(self) -> tuple[str, str]:
        return tuple(sorted((self.left.observation_id, self.right.observation_id)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("movi_d", "movi_e"), required=True)
    parser.add_argument("--tfrecord-dir", type=Path, required=True)
    parser.add_argument("--video-manifest", type=Path, required=True)
    parser.add_argument("--definition-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--max-pairs-per-video", type=int, default=250)
    parser.add_argument("--candidate-retention-per-video-kind-bin", type=int, default=600)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def allocation(total: int) -> dict[str, int]:
    base, remainder = divmod(total, 3)
    return {name: base + (index < remainder) for index, name in enumerate(GAP_BINS)}


def stable_rank(seed: int, *parts: object) -> str:
    return hashlib.sha256("|".join(map(str, (seed, *parts))).encode()).hexdigest()


def eligible_observations(record: dict[str, Any], dataset: str, video_id: str) -> list[Observation]:
    num_frames = scalar(record, "metadata/num_frames", int)
    num_instances = scalar(record, "metadata/num_instances", int)
    visibility = np.asarray(record["instances/visibility"], dtype=np.int64).reshape(num_instances, num_frames)
    depth_range = np.asarray(record["metadata/depth_range"], dtype=np.float64).reshape(2)
    categories = [text_at(record["instances/category"], index) for index in range(num_instances)]
    assets = [text_at(record["instances/asset_id"], index) for index in range(num_instances)]
    dynamic = np.asarray(record["instances/is_dynamic"], dtype=bool).reshape(num_instances)
    rows: list[Observation] = []
    for frame in range(num_frames):
        segmentation = decode_png(record["segmentations"][frame])
        depth = decode_depth(record["depth"][frame], depth_range)
        if segmentation.ndim == 3:
            segmentation = segmentation[..., 0]
        for instance in range(num_instances):
            mask = segmentation == instance + 1
            area = int(mask.sum())
            visible = int(visibility[instance, frame])
            if visible < 32 or area < 32:
                continue
            valid_depth = int(np.sum(np.isfinite(depth[mask]) & (depth[mask] > 0)))
            if valid_depth < 32:
                continue
            rows.append(Observation(
                stable_observation_id(dataset, video_id, frame, instance), frame, instance,
                visible, area, valid_depth, categories[instance], assets[instance], bool(dynamic[instance])
            ))
    return rows


def video_candidates(
    record: dict[str, Any], dataset: str, split: str, video_id: str, cutoff: float
) -> list[Candidate]:
    observations = sorted(eligible_observations(record, dataset, video_id), key=lambda x: (x.frame, x.instance))
    num_frames = scalar(record, "metadata/num_frames", int)
    positions = np.asarray(record["camera/positions"], dtype=np.float64).reshape(num_frames, 3)
    quaternions = np.asarray(record["camera/quaternions"], dtype=np.float64).reshape(num_frames, 4)
    frame_distance = np.linalg.norm(positions, axis=1)
    candidates: list[Candidate] = []
    for index, left in enumerate(observations):
        for right in observations[index + 1:]:
            if left.frame == right.frame:
                continue
            if left.instance == right.instance:
                kind = "positive"
            elif left.category != right.category:
                kind = "easy"
            elif abs(math.log(left.mask_area / right.mask_area)) <= cutoff:
                kind = "hard"
            else:
                continue
            displacement = float(np.linalg.norm(positions[left.frame] - positions[right.frame]))
            denominator = float((frame_distance[left.frame] + frame_distance[right.frame]) / 2)
            rotation = relative_rotation_degrees(quaternions[left.frame], quaternions[right.frame])
            # MOVi-D is fixed-camera; suppress sub-microdegree quaternion roundoff.
            if displacement < 1e-12 and abs(rotation) < 1e-5:
                rotation = 0.0
            candidates.append(Candidate(
                dataset, split, video_id, kind, left, right, displacement,
                rotation,
                displacement / denominator if denominator > 0 else 0.0,
            ))
    return candidates


def retain_candidates(candidates: list[Candidate], seed: int, limit: int) -> list[Candidate]:
    buckets: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        buckets[(candidate.kind, candidate.gap_bin)].append(candidate)
    retained = []
    for (kind, gap), values in buckets.items():
        values.sort(key=lambda item: (stable_rank(seed, item.dataset, item.split, kind, gap, *item.unordered_key), item.unordered_key))
        retained.extend(values[:limit])
    return retained


def select_split(
    candidates: list[Candidate], split: str, seed: int, max_pairs_per_video: int
) -> tuple[list[Candidate], dict[str, Any]]:
    selected: list[Candidate] = []
    used: set[tuple[str, str]] = set()
    video_counts: Counter[str] = Counter()
    fallbacks: list[dict[str, Any]] = []
    for kind in ("positive", "hard", "easy"):
        for gap, target in allocation(TARGETS[split][kind]).items():
            pool = [item for item in candidates if item.kind == kind and item.gap_bin == gap]
            by_video: dict[str, list[Candidate]] = defaultdict(list)
            for item in pool:
                by_video[item.video_id].append(item)
            for video_id, values in by_video.items():
                values.sort(key=lambda item: (
                    stable_rank(seed, split, kind, gap, video_id, *item.unordered_key),
                    item.unordered_key,
                ))
            indices: Counter[str] = Counter()
            chosen = 0
            while chosen < target:
                eligible_videos = [
                    video_id for video_id, values in by_video.items()
                    if indices[video_id] < len(values) and video_counts[video_id] < max_pairs_per_video
                ]
                if not eligible_videos:
                    raise RuntimeError(f"Locked quota shortfall for {split}/{kind}/{gap}: {chosen} < {target}")
                video_id = min(eligible_videos, key=lambda value: (
                    video_counts[value], stable_rank(seed, split, kind, gap, value), value
                ))
                item = by_video[video_id][indices[video_id]]
                indices[video_id] += 1
                if item.unordered_key in used:
                    continue
                selected.append(item)
                used.add(item.unordered_key)
                video_counts[item.video_id] += 1
                chosen += 1
            if len(pool) < target:
                fallbacks.append({"kind": kind, "gap_bin": gap, "reason": "retained_pool_below_target"})
    selected.sort(key=lambda item: (item.kind != "positive", item.kind, item.gap_bin, stable_rank(seed, *item.unordered_key)))
    return selected, {
        "max_video_contribution": max(video_counts.values()),
        "min_video_contribution": min(video_counts.values()),
        "videos_contributing": len(video_counts),
        "video_contribution_counts": dict(sorted(video_counts.items(), key=lambda x: int(x[0]))),
        "fallbacks": fallbacks,
    }


def rounded(value: float) -> float:
    return round(float(value), 12)


def pair_id(candidate: Candidate) -> str:
    a, b = candidate.unordered_key
    return hashlib.sha256(f"movi-de-phase5|{candidate.dataset}|{candidate.split}|{a}|{b}".encode()).hexdigest()[:24]


def pair_row(candidate: Candidate, cutoff: float, seed: int) -> dict[str, Any]:
    left, right = candidate.left, candidate.right
    if left.instance == right.instance:
        category_relation, asset_relation = "same_instance", "same_instance"
    else:
        category_relation = "same_category" if left.category == right.category else "different_category"
        asset_relation = "same_asset_different_instance" if left.asset_id == right.asset_id else "different_asset"
    motion = "-".join(sorted(("dynamic" if left.is_dynamic else "static", "dynamic" if right.is_dynamic else "static")))
    ratio = abs(math.log(left.mask_area / right.mask_area))
    return {
        "pair_id": pair_id(candidate), "dataset": candidate.dataset, "split": candidate.split,
        "label": 1 if candidate.kind == "positive" else 0,
        "negative_difficulty": None if candidate.kind == "positive" else candidate.kind,
        "observation_id_a": left.observation_id, "observation_id_b": right.observation_id,
        "video_id": candidate.video_id, "frame_index_a": left.frame, "frame_index_b": right.frame,
        "instance_index_a": left.instance, "instance_index_b": right.instance,
        "temporal_gap": candidate.gap, "temporal_gap_bin": candidate.gap_bin,
        "controls": {
            "visibility_a": left.visibility, "visibility_b": right.visibility,
            "minimum_visibility": min(left.visibility, right.visibility),
            "mask_area_a": left.mask_area, "mask_area_b": right.mask_area,
            "valid_depth_pixels_a": left.valid_depth_pixels, "valid_depth_pixels_b": right.valid_depth_pixels,
            "absolute_log_area_ratio": rounded(ratio),
            "hard_negative_cutoff_absolute_log_area_ratio": cutoff if candidate.kind == "hard" else None,
            "category_relation": category_relation, "asset_relation": asset_relation,
            "very_hard_same_asset": candidate.kind == "hard" and asset_relation == "same_asset_different_instance",
            "object_dynamic_static_status": motion,
            "camera_displacement_scene_units": rounded(candidate.camera_displacement),
            "relative_camera_rotation_degrees": rounded(candidate.camera_rotation),
            "normalized_camera_displacement": rounded(candidate.normalized_camera_displacement),
        },
        "sampling": {"seed": seed, "generator_version": VERSION, "identity_metadata_for_model_features": False},
    }


def audit(rows: list[dict[str, Any]], split: str, cutoff: float, max_pairs_per_video: int) -> dict[str, Any]:
    kinds = Counter("positive" if row["label"] else row["negative_difficulty"] for row in rows)
    gaps = {kind: Counter() for kind in ("positive", "hard", "easy")}
    videos = Counter()
    violations = []
    keys = set()
    for row in rows:
        kind = "positive" if row["label"] else row["negative_difficulty"]
        gaps[kind][row["temporal_gap_bin"]] += 1
        videos[row["video_id"]] += 1
        key = tuple(sorted((row["observation_id_a"], row["observation_id_b"])))
        if key in keys: violations.append(f"duplicate:{row['pair_id']}")
        keys.add(key)
        c = row["controls"]
        if row["frame_index_a"] == row["frame_index_b"]: violations.append(f"same_frame:{row['pair_id']}")
        if kind == "positive" and row["instance_index_a"] != row["instance_index_b"]: violations.append(f"positive_identity:{row['pair_id']}")
        if kind == "hard" and not (row["instance_index_a"] != row["instance_index_b"] and c["category_relation"] == "same_category" and c["absolute_log_area_ratio"] <= cutoff + 1e-12): violations.append(f"hard_definition:{row['pair_id']}")
        if kind == "easy" and c["category_relation"] != "different_category": violations.append(f"easy_definition:{row['pair_id']}")
    expected = TARGETS[split]
    exact_gaps = all(gaps[kind] == Counter(allocation(expected[kind])) for kind in expected)
    checks = {
        "exact_total": len(rows) == sum(expected.values()), "exact_kind_quotas": kinds == Counter(expected),
        "exact_temporal_gap_allocations": exact_gaps, "unique_unordered_pairs": len(keys) == len(rows),
        "within_video_only": all(row["video_id"] for row in rows),
        "video_cap_respected": max(videos.values()) <= max_pairs_per_video, "all_pair_definitions_pass": not violations,
    }
    return {
        "split": split, "target": sum(expected.values()), "actual": len(rows), "kind_counts": dict(kinds),
        "temporal_gap_bin_counts": {kind: dict(gaps[kind]) for kind in gaps},
        "very_hard_same_asset_count": sum(row["controls"]["very_hard_same_asset"] for row in rows),
        "camera_motion": {
            "mean_displacement": rounded(np.mean([row["controls"]["camera_displacement_scene_units"] for row in rows])),
            "mean_rotation_degrees": rounded(np.mean([row["controls"]["relative_camera_rotation_degrees"] for row in rows])),
            "mean_normalized_displacement": rounded(np.mean([row["controls"]["normalized_camera_displacement"] for row in rows])),
        },
        "max_video_contribution": max(videos.values()), "checks": checks,
        "violations": violations[:100], "status": "pass" if all(checks.values()) else "fail",
    }


def main() -> int:
    args = parse_args()
    config = json.loads(args.definition_config.read_text(encoding="utf-8"))
    if args.seed != int(config["seed"]):
        raise ValueError("Seed differs from frozen definition")
    cutoff = float(config["dataset_cutoffs"][args.dataset]["maximum_absolute_log_area_ratio"])
    manifest = load_jsonl(args.video_manifest)
    selected = {str(row["video_id"]): str(row["split"]) for row in manifest}
    if len(selected) != 150 or Counter(selected.values()) != Counter({"train": 90, "dev": 30, "test": 30}):
        raise ValueError("Video manifest is not the locked 90/30/30 design")
    by_split: dict[str, list[Candidate]] = defaultdict(list)
    found = set()
    for _, record in iter_tfrecords(discover_shards(args.tfrecord_dir, args.dataset, require_complete=True)):
        video_id = record_video_id(record)
        if video_id not in selected:
            continue
        validate_record_schema(record, args.dataset)
        found.add(video_id)
        current = video_candidates(record, args.dataset, selected[video_id], video_id, cutoff)
        retained = retain_candidates(current, args.seed, args.candidate_retention_per_video_kind_bin)
        by_split[selected[video_id]].extend(retained)
        if len(found) % 25 == 0:
            print(f"{args.dataset}: decoded {len(found)}/150 selected videos", flush=True)
    if found != set(selected):
        raise RuntimeError(f"Missing selected videos: {sorted(set(selected) - found)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    reports = {}
    for split in ("train", "dev", "test"):
        chosen, sampling = select_split(by_split[split], split, args.seed + {"train": 0, "dev": 1, "test": 2}[split], args.max_pairs_per_video)
        rows = [pair_row(item, cutoff, args.seed) for item in chosen]
        path = args.output_dir / f"{args.dataset}_{split}_pairs.jsonl"
        write_jsonl(path, rows)
        report = audit(rows, split, cutoff, args.max_pairs_per_video)
        report["sampling"] = sampling
        report["manifest"] = {"path": str(path), "sha256": sha256(path)}
        reports[split] = report
        all_rows.extend(rows)
        if report["status"] != "pass": raise RuntimeError(f"Audit failed for {args.dataset}/{split}")
    combined = args.output_dir / f"{args.dataset}_all_pairs.jsonl"
    write_jsonl(combined, all_rows)
    audit_path = args.output_dir / f"{args.dataset}_pair_generation_audit.json"
    write_json(audit_path, {
        "pipeline": "MOVi-D/E frozen Phase 5 pair generation", "generator_version": VERSION,
        "dataset": args.dataset, "seed": args.seed, "hard_negative_cutoff": cutoff,
        "source_video_manifest": {"path": str(args.video_manifest), "sha256": sha256(args.video_manifest)},
        "definition_config": {"path": str(args.definition_config), "sha256": sha256(args.definition_config)},
        "combined_manifest": {"path": str(combined), "sha256": sha256(combined)},
        "splits": reports, "status": "pass",
    })
    print(f"Complete: {audit_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
