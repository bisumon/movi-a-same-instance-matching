#!/usr/bin/env python3
"""Create an auditable, stratified 50-video MOVi-A selection and 30/10/10 split.

The pipeline never inspects model predictions or image embeddings.  It uses only
predeclared metadata: instance visibility, categorical attributes, and a stable
video identifier.  It supports either a JSONL inventory or direct TFDS loading.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


VERSION = "1.1.0"


def canonical_id(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def json_ready(value: Any) -> Any:
    """Convert TFDS / NumPy / TensorFlow scalar and array values to JSON values."""
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "tolist"):
        return json_ready(value.tolist())
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    return value


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory_from_tfds(data_dir: str | None, out_path: Path) -> None:
    try:
        import tensorflow_datasets as tfds  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "TFDS mode requires tensorflow-datasets. Install it in your project environment, "
            "or supply a JSONL inventory created elsewhere."
        ) from exc

    dataset = tfds.load(
        "movi_a/128x128",
        split="validation",
        data_dir=data_dir,
        shuffle_files=False,
    )
    records = []
    for source_index, sample in enumerate(tfds.as_numpy(dataset)):
        metadata = sample["metadata"]
        instances = sample["instances"]
        records.append(
            {
                "video_id": canonical_id(json_ready(metadata["video_name"])),
                "source_index": source_index,
                "visibility": json_ready(instances["visibility"]),
                "shape_label": json_ready(instances["shape_label"]),
                "color_label": json_ready(instances["color_label"]),
                "material_label": json_ready(instances["material_label"]),
            }
        )
    records.sort(key=lambda row: row["video_id"])
    write_jsonl(out_path, records)


def valid_frames(visibility: list[Any], threshold: int) -> list[int]:
    return [frame for frame, pixels in enumerate(visibility) if int(pixels) >= threshold]


def count_frame_pairs(left: list[int], right: list[int], min_gap: int, max_gap: int, same_instance: bool) -> int:
    count = 0
    for frame_a in left:
        for frame_b in right:
            gap = abs(frame_a - frame_b)
            if min_gap <= gap <= max_gap and (not same_instance or frame_a != frame_b):
                count += 1
    return count


def object_bin(num_instances: int) -> str:
    if 3 <= num_instances <= 4:
        return "03-04"
    if 5 <= num_instances <= 6:
        return "05-06"
    if 7 <= num_instances <= 8:
        return "07-08"
    if 9 <= num_instances <= 10:
        return "09-10"
    return f"other-{num_instances:02d}"


def assess_video(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    required = ("video_id", "visibility", "shape_label", "color_label", "material_label")
    missing = [key for key in required if key not in record]
    if missing:
        return {"video_id": record.get("video_id", "<missing>"), "eligible": False, "reason": f"missing:{','.join(missing)}"}

    video_id = canonical_id(record["video_id"])
    visibility = record["visibility"]
    shapes, colors, materials = record["shape_label"], record["color_label"], record["material_label"]
    if not isinstance(visibility, list) or not visibility:
        return {"video_id": video_id, "eligible": False, "reason": "invalid_visibility"}
    num_instances = len(visibility)
    if any(len(values) != num_instances for values in (shapes, colors, materials)):
        return {"video_id": video_id, "eligible": False, "reason": "attribute_length_mismatch"}
    if any(not isinstance(frames, list) for frames in visibility):
        return {"video_id": video_id, "eligible": False, "reason": "invalid_visibility_shape"}

    frames_by_instance = [valid_frames(frames, args.min_visible_pixels) for frames in visibility]
    positive_candidates = sum(
        count_frame_pairs(frames, frames, args.min_frame_gap, args.max_frame_gap, same_instance=True)
        for frames in frames_by_instance
    )
    hard_negative_candidates = 0
    for left in range(num_instances):
        left_attributes = (canonical_id(shapes[left]), canonical_id(colors[left]), canonical_id(materials[left]))
        for right in range(left + 1, num_instances):
            right_attributes = (canonical_id(shapes[right]), canonical_id(colors[right]), canonical_id(materials[right]))
            if left_attributes == right_attributes:
                hard_negative_candidates += count_frame_pairs(
                    frames_by_instance[left], frames_by_instance[right], args.min_frame_gap, args.max_frame_gap, same_instance=False
                )

    usable_observations = sum(len(frames) for frames in frames_by_instance)
    reasons = []
    if positive_candidates < args.min_positive_candidates:
        reasons.append("insufficient_positive_pairs")
    if hard_negative_candidates < args.min_hard_negative_candidates:
        reasons.append("insufficient_hard_negative_pairs")
    return {
        "video_id": video_id,
        "source_index": record.get("source_index"),
        "eligible": not reasons,
        "reason": ";".join(reasons) if reasons else "eligible",
        "num_instances": num_instances,
        "object_bin": object_bin(num_instances),
        "usable_observations": usable_observations,
        "positive_candidate_pairs": positive_candidates,
        "hard_negative_candidate_pairs": hard_negative_candidates,
        "pair_capacity": min(positive_candidates, hard_negative_candidates),
    }


def assign_capacity_tertiles(rows: list[dict[str, Any]]) -> None:
    """Assign deterministic rank tertiles without requiring pandas or NumPy."""
    ordered = sorted(rows, key=lambda row: (row["pair_capacity"], row["video_id"]))
    n = len(ordered)
    for rank, row in enumerate(ordered):
        row["capacity_tertile"] = ("low", "medium", "high")[min(2, (3 * rank) // max(n, 1))]
        row["selection_stratum"] = f"{row['object_bin']}|{row['capacity_tertile']}"


def proportional_allocation(capacities: dict[str, int], target: int) -> dict[str, int]:
    available = sum(capacities.values())
    if target > available:
        raise ValueError(f"Requested {target} videos, but only {available} are available")
    raw = {key: target * count / available for key, count in capacities.items()}
    allocation = {key: min(capacities[key], math.floor(raw[key])) for key in capacities}
    remaining = target - sum(allocation.values())
    order = sorted(capacities, key=lambda key: (raw[key] - allocation[key], capacities[key], key), reverse=True)
    while remaining:
        progressed = False
        for key in order:
            if allocation[key] < capacities[key] and remaining:
                allocation[key] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise RuntimeError("Unable to satisfy proportional allocation")
    return allocation


def select_rows(eligible: list[dict[str, Any]], target: int, seed: int) -> list[dict[str, Any]]:
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_stratum[row["selection_stratum"]].append(row)
    allocation = proportional_allocation({key: len(value) for key, value in by_stratum.items()}, target)
    rng = random.Random(seed)
    selected = []
    for stratum in sorted(by_stratum):
        candidates = sorted(by_stratum[stratum], key=lambda row: row["video_id"])
        rng.shuffle(candidates)
        selected.extend(candidates[: allocation[stratum]])
    return sorted(selected, key=lambda row: row["video_id"])


def split_rows(selected: list[dict[str, Any]], split_sizes: dict[str, int], seed: int) -> list[dict[str, Any]]:
    """Assign exact sizes, stratifying splits by the primary object-count bin.

    Capacity tertiles remain part of selection and are audited after splitting.
    Using the broader object-count bin here prevents sparse composite strata from
    disappearing from dev or test.
    """
    total = len(selected)
    if total != sum(split_sizes.values()):
        raise ValueError("Selected count must equal the sum of split sizes")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        groups[row["object_bin"]].append(row)
    rng = random.Random(seed + 1)
    assignments: dict[str, str] = {}
    current = Counter()
    split_order = list(split_sizes)
    quotas: dict[str, dict[str, int]] = {}
    for stratum in sorted(groups):
        rows = sorted(groups[stratum], key=lambda row: row["video_id"])
        desired = {split: len(rows) * split_sizes[split] / total for split in split_order}
        # If the bin has at least three videos, guarantee one in every split.
        # Additional videos are allocated toward the target proportions.
        local = {split: (1 if len(rows) >= len(split_order) else 0) for split in split_order}
        for split in split_order:
            current[split] += local[split]
        remainder = len(rows) - sum(local.values())
        for _ in range(remainder):
            choices = [split for split in split_order if current[split] < split_sizes[split]]
            best = max(
                choices,
                key=lambda split: (
                    desired[split] - local[split],
                    split_sizes[split] - current[split],
                    -split_order.index(split),
                ),
            )
            local[best] += 1
            current[best] += 1
        quotas[stratum] = local
    if dict(current) != split_sizes:
        raise RuntimeError(f"Could not make exact split allocation: {dict(current)} != {split_sizes}")

    # Fill each object-bin quota while balancing capacity tertiles globally.
    capacity_totals = Counter(row["capacity_tertile"] for row in selected)
    capacity_current: dict[str, Counter[str]] = defaultdict(Counter)
    for stratum in sorted(groups):
        rows = sorted(groups[stratum], key=lambda row: row["video_id"])
        rng.shuffle(rows)
        rows.sort(key=lambda row: (capacity_totals[row["capacity_tertile"]], row["capacity_tertile"]))
        remaining = dict(quotas[stratum])
        for row in rows:
            capacity = row["capacity_tertile"]
            choices = [split for split in split_order if remaining[split] > 0]
            best = max(
                choices,
                key=lambda split: (
                    capacity_totals[capacity] * split_sizes[split] / total - capacity_current[capacity][split],
                    remaining[split],
                    -split_order.index(split),
                ),
            )
            assignments[row["video_id"]] = best
            remaining[best] -= 1
            capacity_current[capacity][best] += 1

    # Improve the capacity-tertile balance with deterministic within-bin swaps.
    # Swapping only inside an object bin preserves every primary split quota.
    def capacity_score() -> float:
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        for row in selected:
            counts[row["capacity_tertile"]][assignments[row["video_id"]]] += 1
        return sum(
            (
                counts[capacity][split]
                - capacity_totals[capacity] * split_sizes[split] / total
            )
            ** 2
            for capacity in capacity_totals
            for split in split_order
        )

    while True:
        before = capacity_score()
        best_swap = None
        best_score = before
        for stratum in sorted(groups):
            rows = sorted(groups[stratum], key=lambda row: row["video_id"])
            for left_index, left in enumerate(rows):
                for right in rows[left_index + 1 :]:
                    left_split = assignments[left["video_id"]]
                    right_split = assignments[right["video_id"]]
                    if left_split == right_split or left["capacity_tertile"] == right["capacity_tertile"]:
                        continue
                    assignments[left["video_id"]], assignments[right["video_id"]] = right_split, left_split
                    score = capacity_score()
                    assignments[left["video_id"]], assignments[right["video_id"]] = left_split, right_split
                    if score < best_score - 1e-9:
                        best_score = score
                        best_swap = (left["video_id"], right["video_id"])
        if best_swap is None:
            break
        left_id, right_id = best_swap
        assignments[left_id], assignments[right_id] = assignments[right_id], assignments[left_id]
    result = []
    for row in selected:
        result.append({**row, "split": assignments[row["video_id"]]})
    return sorted(result, key=lambda row: row["video_id"])


def summary_rows(audit: list[dict[str, Any]], assignments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    split_by_id = {row["video_id"]: row["split"] for row in assignments}
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    for row in audit:
        stratum = row.get("selection_stratum", "ineligible")
        buckets[stratum]["eligible" if row.get("eligible") else "ineligible"] += 1
        if row["video_id"] in split_by_id:
            buckets[stratum]["selected"] += 1
            buckets[stratum][split_by_id[row["video_id"]]] += 1
    return [
        {
            "selection_stratum": stratum,
            "eligible": counts["eligible"],
            "ineligible": counts["ineligible"],
            "selected": counts["selected"],
            "train": counts["train"],
            "dev": counts["dev"],
            "test": counts["test"],
        }
        for stratum, counts in sorted(buckets.items())
    ]


def run_selection(args: argparse.Namespace) -> None:
    input_path, out_dir = Path(args.input_jsonl), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(input_path)
    seen = set()
    audit = []
    for record in records:
        row = assess_video(record, args)
        if row["video_id"] in seen:
            row["eligible"] = False
            row["reason"] = "duplicate_video_id"
        seen.add(row["video_id"])
        audit.append(row)
    eligible = [row for row in audit if row["eligible"]]
    assign_capacity_tertiles(eligible)
    audit_by_id = {row["video_id"]: row for row in audit}
    for row in eligible:
        audit_by_id[row["video_id"]].update({key: row[key] for key in ("capacity_tertile", "selection_stratum")})

    write_jsonl(out_dir / "selection_audit.jsonl", sorted(audit, key=lambda row: row["video_id"]))
    if len(eligible) < args.select_count:
        raise RuntimeError(
            f"Only {len(eligible)} videos meet the predeclared rules; {args.select_count} are required. "
            "Inspect selection_audit.jsonl and revise thresholds before selection—not after model results."
        )
    selected = select_rows(eligible, args.select_count, args.seed)
    assignments = split_rows(selected, {"train": args.train_count, "dev": args.dev_count, "test": args.test_count}, args.seed)
    write_jsonl(out_dir / "selected_50.jsonl", selected)
    write_jsonl(out_dir / "video_splits.jsonl", assignments)
    summaries = summary_rows(audit, assignments)
    with (out_dir / "strata_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]) if summaries else ["selection_stratum"])
        writer.writeheader()
        writer.writerows(summaries)
    manifest = {
        "pipeline": "movi_a_video_selection.py",
        "pipeline_version": VERSION,
        "input_inventory": str(input_path.resolve()),
        "input_sha256": sha256(input_path),
        "seed": args.seed,
        "selection_rules": {
            "min_visible_pixels": args.min_visible_pixels,
            "min_frame_gap": args.min_frame_gap,
            "max_frame_gap": args.max_frame_gap,
            "min_positive_candidates": args.min_positive_candidates,
            "min_hard_negative_candidates": args.min_hard_negative_candidates,
            "selection_strata": ["object_bin", "pair_capacity_tertile"],
            "selection_method": "proportional stratified random sample",
            "split_method": "object-count-stratified video-disjoint assignment; capacity tertile audited",
        },
        "counts": {"inventory": len(records), "eligible": len(eligible), "selected": len(selected), "splits": Counter(row["split"] for row in assignments)},
        "outputs": ["selection_audit.jsonl", "selected_50.jsonl", "video_splits.jsonl", "strata_summary.csv"],
    }
    write_json(out_dir / "selection_manifest.json", manifest)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inventory = subparsers.add_parser("inventory", help="Create a minimal JSONL inventory directly from TFDS.")
    inventory.add_argument("--tfds-data-dir", default=None, help="TFDS data directory containing movi_a/128x128.")
    inventory.add_argument("--output-jsonl", required=True)
    select = subparsers.add_parser("select", help="Select videos from a JSONL inventory and create manifests.")
    select.add_argument("--input-jsonl", required=True)
    select.add_argument("--output-dir", required=True)
    select.add_argument("--seed", type=int, default=20260727)
    select.add_argument("--select-count", type=int, default=50)
    select.add_argument("--train-count", type=int, default=30)
    select.add_argument("--dev-count", type=int, default=10)
    select.add_argument("--test-count", type=int, default=10)
    select.add_argument("--min-visible-pixels", type=int, default=32)
    select.add_argument("--min-frame-gap", type=int, default=2)
    select.add_argument("--max-frame-gap", type=int, default=23)
    select.add_argument("--min-positive-candidates", type=int, default=20)
    select.add_argument("--min-hard-negative-candidates", type=int, default=20)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.command == "inventory":
        output = Path(args.output_jsonl)
        output.parent.mkdir(parents=True, exist_ok=True)
        inventory_from_tfds(args.tfds_data_dir, output)
    else:
        if args.select_count != args.train_count + args.dev_count + args.test_count:
            raise SystemExit("--select-count must equal --train-count + --dev-count + --test-count")
        if args.min_frame_gap < 1 or args.max_frame_gap < args.min_frame_gap:
            raise SystemExit("Frame-gap bounds must satisfy 1 <= min <= max")
        run_selection(args)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
