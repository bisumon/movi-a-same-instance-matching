#!/usr/bin/env python3
"""Lock and validate the 30/10/10 video split before Phase 2 pair generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


VERSION = "1.0.0"
EXPECTED_VIDEO_COUNTS = {"train": 30, "dev": 10, "test": 10}
PAIR_TARGETS = {"train": 6000, "dev": 2000, "test": 2000}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-videos", type=Path, required=True)
    parser.add_argument("--video-splits", type=Path, required=True)
    parser.add_argument("--observation-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_by_video(rows: list[dict[str, Any]], source: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        video_id = str(row["video_id"])
        if video_id in result:
            raise ValueError(f"Duplicate video ID {video_id} in {source}")
        result[video_id] = row
    return result


def validate_and_lock(
    selected_rows: list[dict[str, Any]],
    split_rows: list[dict[str, Any]],
    observation_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = unique_by_video(selected_rows, Path("selected-videos"))
    assigned = unique_by_video(split_rows, Path("video-splits"))
    if len(selected) != 50:
        raise ValueError(f"Expected 50 selected videos, found {len(selected)}")
    if set(selected) != set(assigned):
        raise ValueError(
            "Selected/split video sets differ: "
            f"missing assignments={sorted(set(selected) - set(assigned))}, "
            f"extra assignments={sorted(set(assigned) - set(selected))}"
        )

    split_counts = Counter(str(row["split"]) for row in assigned.values())
    if dict(split_counts) != EXPECTED_VIDEO_COUNTS:
        raise ValueError(
            f"Expected video counts {EXPECTED_VIDEO_COUNTS}, got {dict(split_counts)}"
        )

    observations_by_video: Counter[str] = Counter()
    observations_by_split: Counter[str] = Counter()
    observed_splits_by_video: dict[str, set[str]] = defaultdict(set)
    observation_ids: set[str] = set()
    for row in observation_rows:
        observation_id = str(row["observation_id"])
        if observation_id in observation_ids:
            raise ValueError(f"Duplicate observation ID {observation_id}")
        observation_ids.add(observation_id)
        video_id = str(row["video_id"])
        split = str(row["split"])
        if video_id not in assigned:
            raise ValueError(f"Observation references unselected video {video_id}")
        expected_split = str(assigned[video_id]["split"])
        if split != expected_split:
            raise ValueError(
                f"Observation {observation_id} says {split}, but video {video_id} is {expected_split}"
            )
        observations_by_video[video_id] += 1
        observations_by_split[split] += 1
        observed_splits_by_video[video_id].add(split)

    missing_observations = set(assigned) - set(observations_by_video)
    if missing_observations:
        raise ValueError(f"Selected videos without Phase 1 observations: {sorted(missing_observations)}")
    if any(len(splits) != 1 for splits in observed_splits_by_video.values()):
        raise ValueError("At least one video appears in multiple observation splits")

    locked_rows = []
    for video_id in sorted(assigned):
        source = assigned[video_id]
        if "split" not in source:
            raise ValueError(f"Missing split for video {video_id}")
        locked_rows.append(
            {
                "video_id": video_id,
                "split": str(source["split"]),
                "num_phase1_observations": observations_by_video[video_id],
                "object_bin": source.get("object_bin"),
                "capacity_tertile": source.get("capacity_tertile"),
                "selection_stratum": source.get("selection_stratum"),
            }
        )

    split_sets = {
        split: {row["video_id"] for row in locked_rows if row["split"] == split}
        for split in EXPECTED_VIDEO_COUNTS
    }
    pairwise_intersections = {
        "train_dev": sorted(split_sets["train"] & split_sets["dev"]),
        "train_test": sorted(split_sets["train"] & split_sets["test"]),
        "dev_test": sorted(split_sets["dev"] & split_sets["test"]),
    }
    if any(pairwise_intersections.values()):
        raise ValueError(f"Video leakage across splits: {pairwise_intersections}")

    summary = {
        "video_counts": dict(sorted(Counter(row["split"] for row in locked_rows).items())),
        "observation_counts": dict(sorted(observations_by_split.items())),
        "pair_targets": PAIR_TARGETS,
        "pairwise_video_intersections": pairwise_intersections,
        "all_50_videos_assigned_once": len(locked_rows) == len({row["video_id"] for row in locked_rows}) == 50,
        "phase1_split_labels_match": True,
        "video_disjoint": not any(pairwise_intersections.values()),
    }
    return locked_rows, summary


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_rows = load_jsonl(args.selected_videos)
    split_rows = load_jsonl(args.video_splits)
    observation_rows = load_jsonl(args.observation_index)
    locked_rows, summary = validate_and_lock(selected_rows, split_rows, observation_rows)

    locked_path = args.output_dir / "locked_video_splits.jsonl"
    write_jsonl(locked_path, locked_rows)
    split_files = {}
    for split in ("train", "dev", "test"):
        path = args.output_dir / f"{split}_videos.jsonl"
        write_jsonl(path, (row for row in locked_rows if row["split"] == split))
        split_files[split] = {
            "filename": path.name,
            "sha256": sha256(path),
            "video_count": EXPECTED_VIDEO_COUNTS[split],
            "pair_target": PAIR_TARGETS[split],
        }

    manifest = {
        "pipeline": "MOVi-A Phase 2 split lock",
        "version": VERSION,
        "rule": "Split videos before pair generation; generate pairs independently within each locked video pool.",
        "inputs": {
            "selected_videos": {
                "path": str(args.selected_videos.resolve()),
                "sha256": sha256(args.selected_videos),
            },
            "video_splits": {
                "path": str(args.video_splits.resolve()),
                "sha256": sha256(args.video_splits),
            },
            "phase1_observation_index": {
                "path": str(args.observation_index.resolve()),
                "sha256": sha256(args.observation_index),
            },
        },
        "locked_split": {
            "filename": locked_path.name,
            "sha256": sha256(locked_path),
            "files": split_files,
        },
        "checks": summary,
        "next_stage_constraints": {
            "primary_pairs_must_be_within_video": True,
            "pair_generation_must_not_cross_split_pools": True,
            "train_pair_target": 6000,
            "dev_pair_target": 2000,
            "test_pair_target": 2000,
            "cross_video_pairs": "Excluded from the primary benchmark; optional scene-rejection stress test only.",
        },
    }
    write_json(args.output_dir / "phase2_split_manifest.json", manifest)
    print(
        "Locked Phase 2 split: "
        f"train={summary['video_counts']['train']}, "
        f"dev={summary['video_counts']['dev']}, "
        f"test={summary['video_counts']['test']}; "
        f"observations={sum(summary['observation_counts'].values())}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
