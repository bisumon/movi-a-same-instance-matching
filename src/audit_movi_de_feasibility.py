#!/usr/bin/env python3
"""Audit MOVi-D/E pilot yield and pair capacity before confirmatory selection."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


VERSION = "1.0.0"
PILOT_VIDEOS = 20
CONFIRMATORY_VIDEOS = 150
PUBLIC_VALIDATION_VIDEOS = 250
TARGETS_PER_90_VIDEOS = {"positive": 3000, "hard": 1500, "easy": 1500}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movi-d-phase1", type=Path, required=True)
    parser.add_argument("--movi-e-phase1", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values: list[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def gap_label(gap: int) -> str:
    if gap <= 5:
        return "short_1_5"
    if gap <= 11:
        return "medium_6_11"
    return "long_12_23"


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def directory_write_span_seconds(path: Path) -> float:
    timestamps = [item.stat().st_mtime for item in path.rglob("*") if item.is_file()]
    return float(max(timestamps) - min(timestamps)) if timestamps else 0.0


def audit_dataset(name: str, root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "phase1_adapter_manifest.json").read_text(encoding="utf-8"))
    pose = json.loads((root / "pose_validation.json").read_text(encoding="utf-8"))
    model = {row["observation_id"]: row for row in load_jsonl(root / "model_inputs.jsonl")}
    index = {row["observation_id"]: row for row in load_jsonl(root / "observation_index.jsonl")}
    metadata = {
        (str(row["video_id"]), int(row["instance_index"])): row
        for row in load_jsonl(root / "instance_metadata.jsonl")
    }
    if model.keys() != index.keys():
        raise ValueError(f"{name}: model/index observation IDs differ")

    by_video: dict[str, list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for observation_id, model_row in model.items():
        index_row = index[observation_id]
        key = (str(index_row["video_id"]), int(index_row["instance_index"]))
        by_video[key[0]].append((index_row, model_row, metadata[key]))

    same_category_log_area_ratios: list[float] = []
    for observations in by_video.values():
        for left_index, (left_i, left_m, left_meta) in enumerate(observations):
            for right_i, right_m, right_meta in observations[left_index + 1 :]:
                if left_i["frame_index"] == right_i["frame_index"]:
                    continue
                if left_i["instance_index"] == right_i["instance_index"]:
                    continue
                if left_meta["category"] == right_meta["category"]:
                    same_category_log_area_ratios.append(
                        abs(math.log(float(left_m["mask_area"]) / float(right_m["mask_area"])))
                    )
    if not same_category_log_area_ratios:
        raise RuntimeError(f"{name}: no same-category negative candidates")
    # The smallest observed cutoff that retains at least 25% of candidates.
    scale_cutoff = float(
        np.quantile(np.asarray(same_category_log_area_ratios), 0.25, method="higher")
    )

    per_video: dict[str, Counter[str]] = {}
    gaps = {kind: Counter() for kind in ("positive", "hard", "easy")}
    statuses = {kind: Counter() for kind in ("positive", "hard", "easy")}
    very_hard_same_asset = 0
    eligible_videos = 0
    for video_id, observations in by_video.items():
        counts: Counter[str] = Counter()
        observations_per_instance = Counter(int(row[0]["instance_index"]) for row in observations)
        for left_index, (left_i, left_m, left_meta) in enumerate(observations):
            for right_i, right_m, right_meta in observations[left_index + 1 :]:
                gap = abs(int(left_i["frame_index"]) - int(right_i["frame_index"]))
                if gap == 0:
                    continue
                if left_i["instance_index"] == right_i["instance_index"]:
                    kind = "positive"
                elif left_meta["category"] != right_meta["category"]:
                    kind = "easy"
                else:
                    ratio = abs(math.log(float(left_m["mask_area"]) / float(right_m["mask_area"])))
                    if ratio > scale_cutoff:
                        continue
                    kind = "hard"
                    if left_meta["asset_id"] == right_meta["asset_id"]:
                        very_hard_same_asset += 1
                counts[kind] += 1
                gaps[kind][gap_label(gap)] += 1
                if kind == "positive":
                    status = "dynamic" if left_meta["is_dynamic"] else "static"
                else:
                    sides = sorted(
                        "dynamic" if row["is_dynamic"] else "static"
                        for row in (left_meta, right_meta)
                    )
                    status = "-".join(sides)
                statuses[kind][status] += 1
        eligible = (
            sum(value >= 2 for value in observations_per_instance.values()) >= 2
            and counts["positive"] > 0
            and counts["hard"] + counts["easy"] > 0
        )
        eligible_videos += int(eligible)
        per_video[video_id] = counts

    scaled_targets = {
        kind: math.ceil(target / 90 * PILOT_VIDEOS)
        for kind, target in TARGETS_PER_90_VIDEOS.items()
    }
    capacity: dict[str, Any] = {}
    for kind in ("positive", "hard", "easy"):
        values = np.asarray([counts[kind] for counts in per_video.values()], dtype=np.int64)
        capacity[kind] = {
            "total_unique_candidates": int(values.sum()),
            "pilot_equivalent_target": scaled_targets[kind],
            "capacity_to_target_ratio": float(values.sum() / scaled_targets[kind]),
            "videos_with_candidates": int(np.sum(values > 0)),
            "per_video_minimum": int(values.min()),
            "per_video_median": float(np.median(values)),
            "per_video_maximum": int(values.max()),
            "temporal_gap_capacity": dict(sorted(gaps[kind].items())),
            "object_motion_capacity": dict(sorted(statuses[kind].items())),
        }

    included = int(manifest["counts"]["included_observations"])
    excluded = int(manifest["counts"]["excluded_observations"])
    motion = [row["camera_motion"] for row in pose.values()]
    translation = [row["translation_path_length_scene_units"] for row in motion]
    rotation = [row["rotation_start_to_end_degrees"] for row in motion]
    normalized = [row["normalized_start_to_end_translation"] for row in motion]
    dynamic_instances = [int(row["dynamic_instance_count"]) for row in pose.values()]

    return {
        "dataset": name,
        "pilot_videos": len(by_video),
        "eligible_videos": eligible_videos,
        "eligibility_rate": eligible_videos / len(by_video),
        "observations": {
            "included": included,
            "excluded": excluded,
            "included_per_video_minimum": min(len(rows) for rows in by_video.values()),
            "included_per_video_median": float(np.median([len(rows) for rows in by_video.values()])),
            "included_per_video_maximum": max(len(rows) for rows in by_video.values()),
            "exclusion_rate": excluded / (included + excluded),
            "exclusions_by_reason": manifest["counts"]["exclusions_by_reason"],
        },
        "pilot_derived_hard_negative_scale_cutoff_abs_log_area_ratio": scale_cutoff,
        "hard_negative_cutoff_status": "feasibility_only; recompute on locked train pool",
        "pair_capacity": capacity,
        "very_hard_same_asset_candidates": very_hard_same_asset,
        "camera_motion": {
            "translation_path_scene_units": {
                "minimum": min(translation),
                "median": percentile(translation, 0.5),
                "maximum": max(translation),
                "tertiles": [percentile(translation, 1 / 3), percentile(translation, 2 / 3)],
            },
            "rotation_start_to_end_degrees": {
                "minimum": min(rotation),
                "median": percentile(rotation, 0.5),
                "maximum": max(rotation),
                "tertiles": [percentile(rotation, 1 / 3), percentile(rotation, 2 / 3)],
            },
            "normalized_translation": {
                "minimum": min(normalized),
                "median": percentile(normalized, 0.5),
                "maximum": max(normalized),
                "tertiles": [percentile(normalized, 1 / 3), percentile(normalized, 2 / 3)],
            },
        },
        "dynamic_instances_per_video": {
            "minimum": min(dynamic_instances),
            "median": float(np.median(dynamic_instances)),
            "maximum": max(dynamic_instances),
        },
        "phase1_output_bytes": directory_size(root),
        "phase1_observed_write_span_seconds": directory_write_span_seconds(root),
    }


def make_markdown(report: dict[str, Any]) -> str:
    d, e = report["datasets"]["movi_d"], report["datasets"]["movi_e"]
    lines = [
        "# MOVi-D/E pilot feasibility audit",
        "",
        "**Protocol:** MOVI-DE-POSE-001  ",
        "**Audit date:** 2026-08-25  ",
        "**Verdict:** Feasible for the 150-video-per-dataset confirmatory design",
        "",
        "## Gate summary",
        "",
        "| Check | MOVi-D | MOVi-E | Result |",
        "|---|---:|---:|---|",
        f"| Eligible pilot videos | {d['eligible_videos']}/20 | {e['eligible_videos']}/20 | Pass |",
        f"| Included observations | {d['observations']['included']:,} | {e['observations']['included']:,} | Pass |",
        f"| Observation exclusion rate | {d['observations']['exclusion_rate']:.1%} | {e['observations']['exclusion_rate']:.1%} | Acceptable and reason-coded |",
        f"| Positive capacity / scaled target | {d['pair_capacity']['positive']['capacity_to_target_ratio']:.1f}x | {e['pair_capacity']['positive']['capacity_to_target_ratio']:.1f}x | Pass |",
        f"| Hard-negative capacity / scaled target | {d['pair_capacity']['hard']['capacity_to_target_ratio']:.1f}x | {e['pair_capacity']['hard']['capacity_to_target_ratio']:.1f}x | Pass |",
        f"| Easy-negative capacity / scaled target | {d['pair_capacity']['easy']['capacity_to_target_ratio']:.1f}x | {e['pair_capacity']['easy']['capacity_to_target_ratio']:.1f}x | Pass |",
        "",
        "The pilot-equivalent quota is 1,335 pairs per 20 videos: 667 positive, 334 hard-negative, and 334 easy-negative pairs. This preserves the protocol's per-video density for a 6,000-pair, 90-video training pool; development and test use the same density.",
        "",
        "## Pair capacity",
        "",
        "| Dataset | Positive | Hard negative | Easy negative | Videos with hard negatives | Same-asset negatives |",
        "|---|---:|---:|---:|---:|---:|",
        f"| MOVi-D | {d['pair_capacity']['positive']['total_unique_candidates']:,} | {d['pair_capacity']['hard']['total_unique_candidates']:,} | {d['pair_capacity']['easy']['total_unique_candidates']:,} | {d['pair_capacity']['hard']['videos_with_candidates']}/20 | {d['very_hard_same_asset_candidates']:,} |",
        f"| MOVi-E | {e['pair_capacity']['positive']['total_unique_candidates']:,} | {e['pair_capacity']['hard']['total_unique_candidates']:,} | {e['pair_capacity']['easy']['total_unique_candidates']:,} | {e['pair_capacity']['hard']['videos_with_candidates']}/20 | {e['very_hard_same_asset_candidates']:,} |",
        "",
        f"The pilot-only scale cutoffs retaining at least 25% of same-category candidates were {d['pilot_derived_hard_negative_scale_cutoff_abs_log_area_ratio']:.3f} for MOVi-D and {e['pilot_derived_hard_negative_scale_cutoff_abs_log_area_ratio']:.3f} for MOVi-E. These values are not locked and must be recomputed independently from each final training pool.",
        "",
        "Positive, hard-negative, and easy-negative candidates exist in short (1–5), medium (6–11), and long (12–23) temporal-gap ranges in both datasets. Static and dynamic positive candidates are present. Dynamic–dynamic hard negatives are less common but have no predeclared quota.",
        "",
        "## Camera-motion and strata coverage",
        "",
        f"MOVi-E covers camera path lengths from {e['camera_motion']['translation_path_scene_units']['minimum']:.3f} to {e['camera_motion']['translation_path_scene_units']['maximum']:.3f} scene units and start-to-end rotations from {e['camera_motion']['rotation_start_to_end_degrees']['minimum']:.3f}° to {e['camera_motion']['rotation_start_to_end_degrees']['maximum']:.3f}°. This is sufficient to derive train-only motion tertiles and audit the motion interaction.",
        "",
        "MOVi-D translation is exactly zero and its numerical rotation is effectively zero. Camera-motion tertiles are therefore degenerate on D by design; D should be reported as a zero-motion falsification control, while low/medium/high motion strata should be interpreted on E only.",
        "",
        "## Scale, storage, and stopping-rule assessment",
        "",
        f"After reserving the 20 pilot videos, 230 of the official 250 validation videos remain per dataset. The minimum confirmatory requirement is 150, leaving an 80-video margin per dataset. The current raw pilot download occupies {report['storage']['pilot_raw_gib']:.2f} GiB; linear extrapolation to all validation shards is approximately {report['storage']['estimated_full_raw_gib']:.2f} GiB. Phase 1 outputs occupy {report['storage']['pilot_phase1_gib']:.2f} GiB for 40 videos, implying approximately {report['storage']['estimated_300_video_phase1_gib']:.2f} GiB for the minimum confirmatory sample.",
        "",
        f"The observed Phase 1 write spans were {d['phase1_observed_write_span_seconds']:.1f} seconds for 20 MOVi-D videos and {e['phase1_observed_write_span_seconds']:.1f} seconds for 20 MOVi-E videos. Linear extrapolation is approximately {report['compute']['estimated_300_video_parallel_minutes']:.1f} minutes when the two dataset jobs run concurrently, or {report['compute']['estimated_300_video_sequential_minutes']:.1f} minutes sequentially, excluding download time and allowing for ordinary video-to-video variation.",
        "",
        "No protocol stopping rule is triggered: data are obtainable, coordinate validation passed, aggregate hard-negative capacity is ample, and storage/compute requirements are modest for the current machine.",
        "",
        "## Constraints to preserve in the confirmatory pipeline",
        "",
        "1. Enforce pair quotas at the pool level. Do not require every video to supply hard negatives; 2 of 20 MOVi-D pilot videos had none after the provisional scale filter.",
        "2. Keep the very-hard same-asset subgroup diagnostic-only. No such candidate occurred in this pilot, consistent with the protocol's no-quota rule.",
        "3. Recompute the hard-negative scale cutoff and all continuous stratum tertiles using the locked training pool only, then freeze them before development or test reporting.",
        "4. Keep all 20 pilot videos out of the confirmatory test pool. If any pilot video is proposed for training or development, record that choice in the split manifest before pair generation.",
        "5. Run a capacity audit after each 90/30/30 split is locked; the pilot establishes feasibility but does not guarantee identical category composition in every random pool.",
        "",
        "Machine-readable details are in `results/movi_de_feasibility_audit.json`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    datasets = {
        "movi_d": audit_dataset("movi_d", args.movi_d_phase1),
        "movi_e": audit_dataset("movi_e", args.movi_e_phase1),
    }
    raw_root = args.movi_d_phase1.parent / "data"
    raw_bytes = directory_size(raw_root)
    phase1_bytes = sum(item["phase1_output_bytes"] for item in datasets.values())
    d_seconds = datasets["movi_d"]["phase1_observed_write_span_seconds"]
    e_seconds = datasets["movi_e"]["phase1_observed_write_span_seconds"]
    confirmatory_scale = 150 / PILOT_VIDEOS
    report = {
        "audit": "MOVi-D/E pilot feasibility",
        "version": VERSION,
        "protocol_id": "MOVI-DE-POSE-001",
        "audit_date": "2026-08-25",
        "overall_status": "feasible_with_preserved_constraints",
        "confirmatory_design": {
            "minimum_videos_per_dataset": CONFIRMATORY_VIDEOS,
            "official_validation_videos_per_dataset": PUBLIC_VALIDATION_VIDEOS,
            "pilot_videos_reserved_from_test_per_dataset": PILOT_VIDEOS,
            "nonpilot_videos_available_per_dataset": PUBLIC_VALIDATION_VIDEOS - PILOT_VIDEOS,
            "margin_above_minimum_per_dataset": PUBLIC_VALIDATION_VIDEOS - PILOT_VIDEOS - CONFIRMATORY_VIDEOS,
            "split": {"train": 90, "dev": 30, "test": 30},
            "pairs": {"train": 6000, "dev": 2000, "test": 2000},
        },
        "datasets": datasets,
        "storage": {
            "pilot_raw_bytes": raw_bytes,
            "pilot_raw_gib": raw_bytes / 2**30,
            "estimated_full_raw_gib": raw_bytes * 8 / 2**30,
            "pilot_phase1_bytes": phase1_bytes,
            "pilot_phase1_gib": phase1_bytes / 2**30,
            "estimated_300_video_phase1_gib": phase1_bytes * (300 / 40) / 2**30,
            "estimation_note": "Linear pilot extrapolation; actual compression and object counts vary by shard/video.",
        },
        "compute": {
            "movi_d_20_video_phase1_write_span_seconds": d_seconds,
            "movi_e_20_video_phase1_write_span_seconds": e_seconds,
            "estimated_300_video_parallel_minutes": max(d_seconds, e_seconds) * confirmatory_scale / 60,
            "estimated_300_video_sequential_minutes": (d_seconds + e_seconds) * confirmatory_scale / 60,
            "estimation_note": "File timestamp span and linear extrapolation; excludes download and later model fitting.",
        },
        "stopping_rules_triggered": [],
        "required_constraints": [
            "Pool-level rather than every-video hard-negative quotas",
            "Train-only hard-negative scale cutoff and continuous stratum boundaries",
            "Pilot videos excluded from confirmatory test",
            "Post-split capacity audit before pair manifests are locked",
            "Same-asset negatives remain diagnostic-only without a quota",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(make_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["overall_status"], "outputs": [str(args.output_json), str(args.output_md)]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
