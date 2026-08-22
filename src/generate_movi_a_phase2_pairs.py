#!/usr/bin/env python3
"""Generate fixed, matched MOVi-A Phase 2 pairs inside locked video pools."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


VERSION = "1.0.0"
SPLIT_TARGETS = {"train": 6000, "dev": 2000, "test": 2000}
GAP_BINS = ("short", "medium", "long")
CONTROL_NAMES = (
    "frame_a",
    "frame_b",
    "temporal_gap",
    "log_visibility_a",
    "log_visibility_b",
    "log_mask_area_a",
    "log_mask_area_b",
    "log_crop_width_a",
    "log_crop_width_b",
    "log_crop_height_a",
    "log_crop_height_b",
)


@dataclass(frozen=True, slots=True)
class Observation:
    observation_id: str
    video_id: str
    split: str
    frame_index: int
    instance_index: int
    visibility: int
    mask_area: int
    padded_crop_width: int
    padded_crop_height: int


@dataclass(frozen=True, slots=True)
class Candidate:
    left: Observation
    right: Observation
    kind: str
    temporal_gap: int
    gap_bin: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.left.observation_id, self.right.observation_id)

    @property
    def frame_key(self) -> tuple[int, int]:
        return (self.left.frame_index, self.right.frame_index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locked-video-splits", type=Path, required=True)
    parser.add_argument("--observation-index", type=Path, required=True)
    parser.add_argument("--model-inputs", type=Path, required=True)
    parser.add_argument("--instance-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--min-frame-gap", type=int, default=2)
    parser.add_argument("--max-frame-gap", type=int, default=23)
    parser.add_argument("--short-gap-max", type=int, default=5)
    parser.add_argument("--medium-gap-max", type=int, default=11)
    parser.add_argument("--max-pairs-per-video", type=int, default=250)
    parser.add_argument("--max-control-smd", type=float, default=0.10)
    parser.add_argument("--mean-balance-weight", type=float, default=50.0)
    parser.add_argument("--rebalance-passes", type=int, default=8)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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


def stable_float(value: float) -> float:
    """Round derived floating-point diagnostics for cross-process byte stability."""
    return round(float(value), 12)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gap_bin(gap: int, short_max: int, medium_max: int) -> str:
    if gap <= short_max:
        return "short"
    if gap <= medium_max:
        return "medium"
    return "long"


def even_allocation(total: int, labels: tuple[str, ...] = GAP_BINS) -> dict[str, int]:
    base, remainder = divmod(total, len(labels))
    return {label: base + (index < remainder) for index, label in enumerate(labels)}


def control_vector(candidate: Candidate) -> np.ndarray:
    a, b = candidate.left, candidate.right
    return np.asarray(
        [
            a.frame_index,
            b.frame_index,
            candidate.temporal_gap,
            math.log1p(a.visibility),
            math.log1p(b.visibility),
            math.log1p(a.mask_area),
            math.log1p(b.mask_area),
            math.log1p(a.padded_crop_width),
            math.log1p(b.padded_crop_width),
            math.log1p(a.padded_crop_height),
            math.log1p(b.padded_crop_height),
        ],
        dtype=np.float64,
    )


def build_observations(
    index_rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    locked_splits: dict[str, str],
) -> list[Observation]:
    model_by_id = {str(row["observation_id"]): row for row in model_rows}
    if len(model_by_id) != len(model_rows):
        raise ValueError("Duplicate observation ID in model inputs")
    observations = []
    for index in index_rows:
        observation_id = str(index["observation_id"])
        if observation_id not in model_by_id:
            raise ValueError(f"Observation {observation_id} missing from model inputs")
        video_id = str(index["video_id"])
        split = str(index["split"])
        if locked_splits.get(video_id) != split:
            raise ValueError(f"Observation {observation_id} disagrees with locked video split")
        model = model_by_id[observation_id]
        if int(model["frame_index"]) != int(index["frame_index"]):
            raise ValueError(f"Frame mismatch for observation {observation_id}")
        observations.append(
            Observation(
                observation_id=observation_id,
                video_id=video_id,
                split=split,
                frame_index=int(index["frame_index"]),
                instance_index=int(index["instance_index"]),
                visibility=int(model["visibility"]),
                mask_area=int(model["mask_area"]),
                padded_crop_width=int(model["padded_crop_width"]),
                padded_crop_height=int(model["padded_crop_height"]),
            )
        )
    if len(observations) != len(model_rows):
        raise ValueError("Model inputs contain observations absent from the index")
    return observations


def enumerate_candidates(
    observations: list[Observation],
    attributes: dict[tuple[str, int], tuple[str, str, str]],
    args: argparse.Namespace,
) -> dict[str, dict[str, list[Candidate]]]:
    by_video: dict[str, list[Observation]] = defaultdict(list)
    for observation in observations:
        by_video[observation.video_id].append(observation)
    candidates: dict[str, dict[str, list[Candidate]]] = {
        split: {"positive": [], "hard": [], "easy": []} for split in SPLIT_TARGETS
    }
    for video_id, video_observations in by_video.items():
        ordered = sorted(video_observations, key=lambda item: (item.frame_index, item.instance_index))
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                gap = right.frame_index - left.frame_index
                if gap < args.min_frame_gap:
                    continue
                if gap > args.max_frame_gap:
                    break
                if left.instance_index == right.instance_index:
                    kind = "positive"
                elif attributes[(video_id, left.instance_index)] == attributes[
                    (video_id, right.instance_index)
                ]:
                    kind = "hard"
                else:
                    kind = "easy"
                candidates[left.split][kind].append(
                    Candidate(left, right, kind, gap, gap_bin(gap, args.short_gap_max, args.medium_gap_max))
                )
    return candidates


def select_positives(
    candidates: list[Candidate],
    target: int,
    num_videos: int,
    max_pairs_per_video: int,
    rng: random.Random,
) -> tuple[list[Candidate], Counter[str]]:
    targets = even_allocation(target)
    by_bin_video: dict[str, dict[str, list[Candidate]]] = {
        label: defaultdict(list) for label in GAP_BINS
    }
    for candidate in candidates:
        by_bin_video[candidate.gap_bin][candidate.left.video_id].append(candidate)
    for per_video in by_bin_video.values():
        for values in per_video.values():
            rng.shuffle(values)

    positive_cap = min(
        max_pairs_per_video,
        math.ceil(1.25 * target / num_videos),
    )
    selected: list[Candidate] = []
    video_counts: Counter[str] = Counter()
    bin_video_counts: Counter[tuple[str, str]] = Counter()
    for label in GAP_BINS:
        per_video = by_bin_video[label]
        indices: Counter[str] = Counter()
        for _ in range(targets[label]):
            eligible = [
                video_id
                for video_id, values in per_video.items()
                if indices[video_id] < len(values) and video_counts[video_id] < positive_cap
            ]
            if not eligible:
                raise RuntimeError(f"Insufficient positive capacity in temporal bin {label}")
            minimum_bin_count = min(bin_video_counts[(label, video_id)] for video_id in eligible)
            eligible = [
                video_id
                for video_id in eligible
                if bin_video_counts[(label, video_id)] == minimum_bin_count
            ]
            minimum_total = min(video_counts[video_id] for video_id in eligible)
            eligible = [video_id for video_id in eligible if video_counts[video_id] == minimum_total]
            video_id = rng.choice(sorted(eligible))
            selected.append(per_video[video_id][indices[video_id]])
            indices[video_id] += 1
            video_counts[video_id] += 1
            bin_video_counts[(label, video_id)] += 1
    return selected, video_counts


def partition_positive_templates(
    positives: list[Candidate],
    hard_target: int,
    rng: random.Random,
) -> tuple[list[Candidate], list[Candidate]]:
    by_bin: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in positives:
        by_bin[candidate.gap_bin].append(candidate)
    hard_templates: list[Candidate] = []
    easy_templates: list[Candidate] = []
    remaining_hard = hard_target
    for index, label in enumerate(GAP_BINS):
        values = by_bin[label]
        rng.shuffle(values)
        bins_left = len(GAP_BINS) - index
        hard_here = min(len(values), round(remaining_hard / bins_left))
        hard_templates.extend(values[:hard_here])
        easy_templates.extend(values[hard_here:])
        remaining_hard -= hard_here
    if remaining_hard != 0:
        raise RuntimeError("Unable to partition positive templates for hard negatives")
    return hard_templates, easy_templates


def match_negatives(
    templates: list[Candidate],
    negative_candidates: list[Candidate],
    video_counts: Counter[str],
    max_pairs_per_video: int,
    scales: np.ndarray,
    mean_balance_weight: float,
    rng: random.Random,
) -> list[tuple[Candidate, Candidate, float, bool]]:
    exact: dict[tuple[int, int], list[Candidate]] = defaultdict(list)
    by_bin: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in negative_candidates:
        exact[candidate.frame_key].append(candidate)
        by_bin[candidate.gap_bin].append(candidate)
    for values in exact.values():
        rng.shuffle(values)
    for values in by_bin.values():
        rng.shuffle(values)

    # Match the most constrained frame pairs first.
    decorated = [(len(exact[template.frame_key]), rng.random(), template) for template in templates]
    decorated.sort(key=lambda item: (item[0], item[1]))
    used: set[tuple[str, str]] = set()
    matched: list[tuple[Candidate, Candidate, float, bool]] = []
    target_mean = np.vstack([control_vector(template) for template in templates]).mean(axis=0)
    selected_sum = np.zeros_like(target_mean)
    for _, _, template in decorated:
        template_vector = control_vector(template)
        pool = [
            candidate
            for candidate in exact[template.frame_key]
            if candidate.key not in used
            and video_counts[candidate.left.video_id] < max_pairs_per_video
        ]
        exact_frame_match = True
        if not pool:
            exact_frame_match = False
            pool = [
                candidate
                for candidate in by_bin[template.gap_bin]
                if candidate.key not in used
                and video_counts[candidate.left.video_id] < max_pairs_per_video
            ]
        if not pool:
            raise RuntimeError(
                f"Insufficient {negative_candidates[0].kind if negative_candidates else 'negative'} "
                f"capacity for {template.gap_bin} templates under the video cap"
            )

        def distance(candidate: Candidate) -> float:
            candidate_vector = control_vector(candidate)
            delta = (candidate_vector - template_vector) / scales
            value = float(np.dot(delta, delta))
            prospective_mean = (selected_sum + candidate_vector) / (len(matched) + 1)
            mean_delta = (prospective_mean - target_mean) / scales
            value += mean_balance_weight * float(np.dot(mean_delta, mean_delta))
            if not exact_frame_match:
                value += 4.0 * (
                    abs(candidate.left.frame_index - template.left.frame_index)
                    + abs(candidate.right.frame_index - template.right.frame_index)
                )
            return value

        # Exact frame buckets are modest; fallback pools can be large, so use a
        # deterministic 4096-candidate window when necessary.
        if len(pool) > 4096:
            pool = rng.sample(pool, 4096)
        chosen = min(pool, key=lambda candidate: (distance(candidate), candidate.key))
        direct_delta = (control_vector(chosen) - template_vector) / scales
        chosen_distance = float(np.dot(direct_delta, direct_delta))
        used.add(chosen.key)
        video_counts[chosen.left.video_id] += 1
        selected_sum += control_vector(chosen)
        matched.append((chosen, template, chosen_distance, exact_frame_match))
    return matched


def rebalance_negative_matches(
    matches: list[tuple[Candidate, Candidate, float, bool]],
    candidate_pool: list[Candidate],
    video_counts: Counter[str],
    max_pairs_per_video: int,
    positive_mean: np.ndarray,
    scales: np.ndarray,
    passes: int,
    rng: random.Random,
) -> list[tuple[Candidate, Candidate, float, bool]]:
    """Swap negatives within exact frame buckets to reduce global mean imbalance."""
    by_frame: dict[tuple[int, int], list[Candidate]] = defaultdict(list)
    for candidate in candidate_pool:
        by_frame[candidate.frame_key].append(candidate)
    for values in by_frame.values():
        rng.shuffle(values)
    result = list(matches)
    used = {candidate.key for candidate, _, _, _ in result}
    selected_sum = sum((control_vector(candidate) for candidate, _, _, _ in result), np.zeros(len(CONTROL_NAMES)))
    count = len(result)

    def objective(candidate_sum: np.ndarray) -> float:
        standardized = (candidate_sum / count - positive_mean) / scales
        return 10.0 * float(np.max(np.abs(standardized))) + float(np.dot(standardized, standardized))

    for _ in range(passes):
        improved = False
        order = list(range(count))
        rng.shuffle(order)
        for match_index in order:
            current, template, _, exact_match = result[match_index]
            current_vector = control_vector(current)
            current_objective = objective(selected_sum)
            best = current
            best_sum = selected_sum
            best_objective = current_objective
            pool = by_frame[current.frame_key]
            if len(pool) > 4096:
                pool = rng.sample(pool, 4096)
            for candidate in pool:
                if candidate.key != current.key and candidate.key in used:
                    continue
                old_video = current.left.video_id
                new_video = candidate.left.video_id
                if new_video != old_video and video_counts[new_video] >= max_pairs_per_video:
                    continue
                prospective_sum = selected_sum - current_vector + control_vector(candidate)
                candidate_objective = objective(prospective_sum)
                if (candidate_objective, candidate.key) < (best_objective, best.key):
                    best = candidate
                    best_sum = prospective_sum
                    best_objective = candidate_objective
            if best.key != current.key:
                used.remove(current.key)
                used.add(best.key)
                video_counts[current.left.video_id] -= 1
                video_counts[best.left.video_id] += 1
                selected_sum = best_sum
                template_delta = (control_vector(best) - control_vector(template)) / scales
                result[match_index] = (
                    best,
                    template,
                    float(np.dot(template_delta, template_delta)),
                    exact_match,
                )
                improved = True
        if not improved:
            break
    return result


def stable_pair_id(split: str, candidate: Candidate) -> str:
    payload = (
        f"movi-a-phase2|{split}|{candidate.left.observation_id}|"
        f"{candidate.right.observation_id}"
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:24]


def pair_row(
    split: str,
    candidate: Candidate,
    template_pair_id: str | None = None,
    matching_distance: float | None = None,
    exact_frame_match: bool | None = None,
) -> dict[str, Any]:
    positive = candidate.kind == "positive"
    row = {
        "pair_id": stable_pair_id(split, candidate),
        "split": split,
        "label": 1 if positive else 0,
        "negative_difficulty": None if positive else candidate.kind,
        "attribute_relation": (
            "same_instance"
            if positive
            else "matched_shape_color_material"
            if candidate.kind == "hard"
            else "mismatched_attributes"
        ),
        "observation_id_a": candidate.left.observation_id,
        "observation_id_b": candidate.right.observation_id,
        "video_id": candidate.left.video_id,
        "instance_index_a": candidate.left.instance_index,
        "instance_index_b": candidate.right.instance_index,
        "frame_index_a": candidate.left.frame_index,
        "frame_index_b": candidate.right.frame_index,
        "temporal_gap": candidate.temporal_gap,
        "temporal_gap_bin": candidate.gap_bin,
        "controls": {
            "visibility_a": candidate.left.visibility,
            "visibility_b": candidate.right.visibility,
            "mask_area_a": candidate.left.mask_area,
            "mask_area_b": candidate.right.mask_area,
            "padded_crop_width_a": candidate.left.padded_crop_width,
            "padded_crop_width_b": candidate.right.padded_crop_width,
            "padded_crop_height_a": candidate.left.padded_crop_height,
            "padded_crop_height_b": candidate.right.padded_crop_height,
        },
        "sampling": {
            "matched_positive_pair_id": template_pair_id,
            "control_distance_squared": (
                stable_float(matching_distance) if matching_distance is not None else None
            ),
            "exact_frame_indices_matched": exact_frame_match,
        },
    }
    return row


def standardized_mean_differences(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[np.ndarray]] = {"positive": [], "hard": [], "easy": []}
    for row in rows:
        group = "positive" if row["label"] == 1 else row["negative_difficulty"]
        controls = row["controls"]
        values = np.asarray(
            [
                row["frame_index_a"],
                row["frame_index_b"],
                row["temporal_gap"],
                math.log1p(controls["visibility_a"]),
                math.log1p(controls["visibility_b"]),
                math.log1p(controls["mask_area_a"]),
                math.log1p(controls["mask_area_b"]),
                math.log1p(controls["padded_crop_width_a"]),
                math.log1p(controls["padded_crop_width_b"]),
                math.log1p(controls["padded_crop_height_a"]),
                math.log1p(controls["padded_crop_height_b"]),
            ],
            dtype=np.float64,
        )
        groups[group].append(values)
    positive = np.vstack(groups["positive"])
    result: dict[str, Any] = {}
    for group in ("hard", "easy"):
        negative = np.vstack(groups[group])
        pooled = np.sqrt((positive.var(axis=0, ddof=1) + negative.var(axis=0, ddof=1)) / 2)
        pooled[pooled == 0] = 1.0
        smd = np.abs(negative.mean(axis=0) - positive.mean(axis=0)) / pooled
        result[group] = {
            "max_abs_smd": stable_float(smd.max()),
            "by_control": {
                name: stable_float(value) for name, value in zip(CONTROL_NAMES, smd, strict=True)
            },
        }
    return result


def generate_split(
    split: str,
    target: int,
    candidates: dict[str, list[Candidate]],
    num_videos: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(args.seed + {"train": 0, "dev": 1, "test": 2}[split])
    positive_target = target // 2
    hard_target = (target - positive_target) // 2
    easy_target = target - positive_target - hard_target
    positives, video_counts = select_positives(
        candidates["positive"],
        positive_target,
        num_videos,
        args.max_pairs_per_video,
        rng,
    )
    positive_vectors = np.vstack([control_vector(candidate) for candidate in positives])
    scales = positive_vectors.std(axis=0, ddof=1)
    scales[scales < 1e-6] = 1.0

    hard_templates, easy_templates = partition_positive_templates(positives, hard_target, rng)
    if len(easy_templates) != easy_target:
        raise RuntimeError("Positive-template partition does not match the easy-negative target")
    hard_matches = match_negatives(
        hard_templates,
        candidates["hard"],
        video_counts,
        args.max_pairs_per_video,
        scales,
        args.mean_balance_weight,
        rng,
    )
    hard_matches = rebalance_negative_matches(
        hard_matches,
        candidates["hard"],
        video_counts,
        args.max_pairs_per_video,
        positive_vectors.mean(axis=0),
        scales,
        args.rebalance_passes,
        rng,
    )
    easy_matches = match_negatives(
        easy_templates,
        candidates["easy"],
        video_counts,
        args.max_pairs_per_video,
        scales,
        args.mean_balance_weight,
        rng,
    )
    easy_matches = rebalance_negative_matches(
        easy_matches,
        candidates["easy"],
        video_counts,
        args.max_pairs_per_video,
        positive_vectors.mean(axis=0),
        scales,
        args.rebalance_passes,
        rng,
    )

    positive_rows = [pair_row(split, candidate) for candidate in positives]
    positive_pair_ids = {
        candidate.key: row["pair_id"] for candidate, row in zip(positives, positive_rows, strict=True)
    }
    negative_rows = []
    for candidate, template, distance, exact_match in hard_matches + easy_matches:
        negative_rows.append(
            pair_row(
                split,
                candidate,
                template_pair_id=positive_pair_ids[template.key],
                matching_distance=distance,
                exact_frame_match=exact_match,
            )
        )
    rows = positive_rows + negative_rows
    rows.sort(key=lambda row: row["pair_id"])
    if len(rows) != target or len({row["pair_id"] for row in rows}) != target:
        raise RuntimeError(f"Pair count or uniqueness failure for {split}")
    if max(video_counts.values()) > args.max_pairs_per_video:
        raise RuntimeError(f"Per-video cap exceeded for {split}")

    smd = standardized_mean_differences(rows)
    max_smd = max(smd[group]["max_abs_smd"] for group in ("hard", "easy"))
    if max_smd > args.max_control_smd:
        raise RuntimeError(
            f"Control matching failed for {split}: max SMD {max_smd:.4f} "
            f"exceeds {args.max_control_smd:.4f}"
        )
    exact_frame_matches = Counter(
        row["negative_difficulty"]
        for row in negative_rows
        if row["sampling"]["exact_frame_indices_matched"]
    )
    summary = {
        "target": target,
        "labels": dict(sorted(Counter(str(row["label"]) for row in rows).items())),
        "negative_difficulty": dict(
            sorted(Counter(row["negative_difficulty"] for row in negative_rows).items())
        ),
        "temporal_gap_bins": {
            group: dict(
                sorted(
                    Counter(
                        row["temporal_gap_bin"]
                        for row in rows
                        if (group == "positive" and row["label"] == 1)
                        or row["negative_difficulty"] == group
                    ).items()
                )
            )
            for group in ("positive", "hard", "easy")
        },
        "video_pair_counts": {
            "minimum": min(video_counts.values()),
            "median": float(np.median(list(video_counts.values()))),
            "maximum": max(video_counts.values()),
            "videos_represented": len(video_counts),
        },
        "exact_frame_index_matches": dict(sorted(exact_frame_matches.items())),
        "control_matching_smd": smd,
    }
    return rows, summary


def main() -> int:
    args = parse_args()
    if not (1 <= args.min_frame_gap <= args.short_gap_max < args.medium_gap_max <= args.max_frame_gap):
        raise ValueError("Temporal-gap boundaries are inconsistent")
    if args.max_pairs_per_video < 1:
        raise ValueError("--max-pairs-per-video must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    locked_rows = load_jsonl(args.locked_video_splits)
    locked_splits = {str(row["video_id"]): str(row["split"]) for row in locked_rows}
    if len(locked_splits) != 50:
        raise ValueError("Locked split must contain exactly 50 unique videos")
    video_counts = Counter(locked_splits.values())
    if video_counts != Counter({"train": 30, "dev": 10, "test": 10}):
        raise ValueError(f"Locked split is not 30/10/10: {dict(video_counts)}")

    index_rows = load_jsonl(args.observation_index)
    model_rows = load_jsonl(args.model_inputs)
    metadata_rows = load_jsonl(args.instance_metadata)
    attributes = {
        (str(row["video_id"]), int(row["instance_index"])): (
            str(row["shape_label"]),
            str(row["color_label"]),
            str(row["material_label"]),
        )
        for row in metadata_rows
    }
    observations = build_observations(index_rows, model_rows, locked_splits)
    candidates = enumerate_candidates(observations, attributes, args)

    all_rows = []
    summaries = {}
    candidate_capacity = {}
    output_hashes = {}
    for split in ("train", "dev", "test"):
        candidate_capacity[split] = {
            kind: {
                "total": len(candidates[split][kind]),
                "by_gap_bin": dict(
                    sorted(Counter(item.gap_bin for item in candidates[split][kind]).items())
                ),
            }
            for kind in ("positive", "hard", "easy")
        }
        rows, summary = generate_split(
            split,
            SPLIT_TARGETS[split],
            candidates[split],
            video_counts[split],
            args,
        )
        path = args.output_dir / f"{split}_pairs.jsonl"
        write_jsonl(path, rows)
        output_hashes[path.name] = sha256(path)
        all_rows.extend(rows)
        summaries[split] = summary
        print(
            f"{split}: pairs={len(rows)} positives={summary['labels']['1']} "
            f"hard={summary['negative_difficulty']['hard']} "
            f"easy={summary['negative_difficulty']['easy']} "
            f"max_video={summary['video_pair_counts']['maximum']}",
            flush=True,
        )

    all_rows.sort(key=lambda row: (row["split"], row["pair_id"]))
    all_path = args.output_dir / "all_pairs.jsonl"
    write_jsonl(all_path, all_rows)
    output_hashes[all_path.name] = sha256(all_path)

    with (args.output_dir / "pair_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "pairs",
                "positives",
                "hard_negatives",
                "easy_negatives",
                "max_pairs_per_video",
                "max_hard_control_smd",
                "max_easy_control_smd",
            ],
        )
        writer.writeheader()
        for split in ("train", "dev", "test"):
            summary = summaries[split]
            writer.writerow(
                {
                    "split": split,
                    "pairs": summary["target"],
                    "positives": summary["labels"]["1"],
                    "hard_negatives": summary["negative_difficulty"]["hard"],
                    "easy_negatives": summary["negative_difficulty"]["easy"],
                    "max_pairs_per_video": summary["video_pair_counts"]["maximum"],
                    "max_hard_control_smd": summary["control_matching_smd"]["hard"]["max_abs_smd"],
                    "max_easy_control_smd": summary["control_matching_smd"]["easy"]["max_abs_smd"],
                }
            )
    output_hashes["pair_summary.csv"] = sha256(args.output_dir / "pair_summary.csv")

    pair_keys = {
        split: {
            tuple(sorted((row["observation_id_a"], row["observation_id_b"])))
            for row in all_rows
            if row["split"] == split
        }
        for split in SPLIT_TARGETS
    }
    observation_video = {
        observation.observation_id: observation.video_id for observation in observations
    }
    cross_split_pair_intersections = {
        "train_dev": len(pair_keys["train"] & pair_keys["dev"]),
        "train_test": len(pair_keys["train"] & pair_keys["test"]),
        "dev_test": len(pair_keys["dev"] & pair_keys["test"]),
    }
    manifest = {
        "pipeline": "MOVi-A Phase 2 within-video matched pair generation",
        "version": VERSION,
        "seed": args.seed,
        "inputs": {
            "locked_video_splits": {
                "path": str(args.locked_video_splits.resolve()),
                "sha256": sha256(args.locked_video_splits),
            },
            "observation_index": {
                "path": str(args.observation_index.resolve()),
                "sha256": sha256(args.observation_index),
            },
            "model_inputs": {
                "path": str(args.model_inputs.resolve()),
                "sha256": sha256(args.model_inputs),
            },
            "instance_metadata": {
                "path": str(args.instance_metadata.resolve()),
                "sha256": sha256(args.instance_metadata),
            },
        },
        "protocol": {
            "split_pair_targets": SPLIT_TARGETS,
            "label_mix": {
                "positive": 0.50,
                "hard_negative": 0.25,
                "easy_negative": 0.25,
            },
            "positive_definition": "Same instance, different frames, same video.",
            "hard_negative_definition": "Different instances in the same video matching shape, color, and material.",
            "easy_negative_definition": "Different instances in the same video with at least one mismatched shape/color/material attribute.",
            "temporal_gap_bins": {
                "short": [args.min_frame_gap, args.short_gap_max],
                "medium": [args.short_gap_max + 1, args.medium_gap_max],
                "long": [args.medium_gap_max + 1, args.max_frame_gap],
            },
            "max_pairs_per_video": args.max_pairs_per_video,
            "negative_matching": "Greedy nearest match to a unique positive template on frame indices, visibility, mask area, and padded crop width/height.",
            "negative_mean_balance_weight": args.mean_balance_weight,
            "negative_rebalance_passes": args.rebalance_passes,
            "max_allowed_absolute_standardized_mean_difference": args.max_control_smd,
            "cross_video_pairs": "Not included in the primary benchmark.",
        },
        "candidate_capacity": candidate_capacity,
        "results": summaries,
        "global_checks": {
            "total_pairs": len(all_rows),
            "unique_pair_ids": len({row["pair_id"] for row in all_rows}),
            "all_pairs_within_video": all(
                observation_video[row["observation_id_a"]]
                == row["video_id"]
                == observation_video[row["observation_id_b"]]
                for row in all_rows
            ),
            "cross_split_pair_intersections": cross_split_pair_intersections,
            "video_split_membership_locked": all(
                locked_splits[row["video_id"]] == row["split"] for row in all_rows
            ),
        },
        "output_sha256": output_hashes,
    }
    write_json(args.output_dir / "phase2_pair_manifest.json", manifest)
    print(f"Complete: {len(all_rows)} fixed pairs", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
