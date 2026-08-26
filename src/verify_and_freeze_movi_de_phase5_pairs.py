#!/usr/bin/env python3
"""Independently verify and checksum-lock generated MOVi-D/E Phase 5 pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


TARGETS = {
    "train": {"positive": 3000, "hard": 1500, "easy": 1500},
    "dev": {"positive": 1000, "hard": 500, "easy": 500},
    "test": {"positive": 1000, "hard": 500, "easy": 500},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def expected_observation_id(dataset: str, video_id: str, frame: int, instance: int) -> str:
    return hashlib.sha256(f"{dataset}|{video_id}|{frame}|{instance}".encode()).hexdigest()[:20]


def expected_pair_id(dataset: str, split: str, observation_a: str, observation_b: str) -> str:
    a, b = sorted((observation_a, observation_b))
    return hashlib.sha256(f"movi-de-phase5|{dataset}|{split}|{a}|{b}".encode()).hexdigest()[:24]


def main() -> int:
    args = parse_args()
    root = args.repository_root.resolve()
    pair_dir = root / "manifests/pairs/movi_de"
    definition = json.loads((root / "configs/movi_de_hard_negatives.json").read_text())
    checks: dict[str, bool] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    all_pair_ids = set()
    all_unordered = set()
    total = 0
    for dataset in ("movi_d", "movi_e"):
        video_rows = rows(root / f"manifests/movi_de/confirmatory_{dataset}_150.jsonl")
        split_by_video = {str(row["video_id"]): row["split"] for row in video_rows}
        cutoff = float(definition["dataset_cutoffs"][dataset]["maximum_absolute_log_area_ratio"])
        split_rows = []
        for split in ("train", "dev", "test"):
            path = pair_dir / f"{dataset}_{split}_pairs.jsonl"
            current = rows(path)
            split_rows.extend(current)
            total += len(current)
            kinds = Counter("positive" if row["label"] == 1 else row["negative_difficulty"] for row in current)
            prefix = f"{dataset}_{split}"
            checks[f"{prefix}_exact_quotas"] = kinds == Counter(TARGETS[split])
            checks[f"{prefix}_unique_pair_ids"] = len({row["pair_id"] for row in current}) == len(current)
            checks[f"{prefix}_locked_video_scope"] = all(split_by_video.get(str(row["video_id"])) == split for row in current)
            checks[f"{prefix}_same_video_pair_schema"] = all(row["video_id"] and row["frame_index_a"] != row["frame_index_b"] for row in current)
            checks[f"{prefix}_observation_ids_bind_dataset_video_frame_instance"] = all(
                row["observation_id_a"] == expected_observation_id(dataset, str(row["video_id"]), row["frame_index_a"], row["instance_index_a"])
                and row["observation_id_b"] == expected_observation_id(dataset, str(row["video_id"]), row["frame_index_b"], row["instance_index_b"])
                for row in current
            )
            checks[f"{prefix}_pair_ids_bind_unordered_observations"] = all(
                row["pair_id"] == expected_pair_id(dataset, split, row["observation_id_a"], row["observation_id_b"])
                for row in current
            )
            checks[f"{prefix}_definitions"] = all(
                (row["label"] == 1 and row["instance_index_a"] == row["instance_index_b"])
                or (row["negative_difficulty"] == "easy" and row["instance_index_a"] != row["instance_index_b"] and row["controls"]["category_relation"] == "different_category")
                or (row["negative_difficulty"] == "hard" and row["instance_index_a"] != row["instance_index_b"] and row["controls"]["category_relation"] == "same_category" and row["controls"]["absolute_log_area_ratio"] <= cutoff + 1e-12)
                for row in current
            )
            checks[f"{prefix}_eligibility_controls"] = all(
                row["controls"]["visibility_a"] >= 32 and row["controls"]["visibility_b"] >= 32
                and row["controls"]["mask_area_a"] >= 32 and row["controls"]["mask_area_b"] >= 32
                and row["controls"]["valid_depth_pixels_a"] >= 32 and row["controls"]["valid_depth_pixels_b"] >= 32
                for row in current
            )
            for row in current:
                pair_key = (dataset, *sorted((row["observation_id_a"], row["observation_id_b"])))
                checks[f"pair_id_collision_{row['pair_id']}"] = row["pair_id"] not in all_pair_ids
                checks[f"unordered_collision_{dataset}_{row['pair_id']}"] = pair_key not in all_unordered
                all_pair_ids.add(row["pair_id"])
                all_unordered.add(pair_key)
            artifacts[str(path.relative_to(root))] = {"sha256": sha256(path), "rows": len(current)}
        combined_path = pair_dir / f"{dataset}_all_pairs.jsonl"
        combined = rows(combined_path)
        checks[f"{dataset}_combined_matches_splits"] = combined == split_rows
        checks[f"{dataset}_fixed_camera_zero"] = dataset != "movi_d" or all(
            row["controls"]["camera_displacement_scene_units"] == 0
            and row["controls"]["relative_camera_rotation_degrees"] == 0
            and row["controls"]["normalized_camera_displacement"] == 0 for row in combined
        )
        checks[f"{dataset}_moving_camera_nonzero"] = dataset != "movi_e" or any(
            row["controls"]["camera_displacement_scene_units"] > 0
            and row["controls"]["relative_camera_rotation_degrees"] > 0 for row in combined
        )
        artifacts[str(combined_path.relative_to(root))] = {"sha256": sha256(combined_path), "rows": len(combined)}
        audit_path = pair_dir / f"{dataset}_pair_generation_audit.json"
        audit = json.loads(audit_path.read_text())
        checks[f"{dataset}_generator_audit_pass"] = audit["status"] == "pass"
        artifacts[str(audit_path.relative_to(root))] = {"sha256": sha256(audit_path)}
    compact_checks = {key: value for key, value in checks.items() if not key.startswith(("pair_id_collision_", "unordered_collision_"))}
    compact_checks["all_20000_pair_ids_globally_unique"] = len(all_pair_ids) == 20000
    compact_checks["all_20000_unordered_pairs_unique_within_dataset"] = len(all_unordered) == 20000
    compact_checks["exact_total_pairs"] = total == 20000
    compact_checks["all_expanded_checks_pass"] = all(checks.values())
    support_paths = [
        "configs/movi_de_phase5_pairs.json", "configs/movi_de_hard_negatives.json",
        "src/generate_movi_de_phase5_pairs.py", "src/verify_and_freeze_movi_de_phase5_pairs.py",
        "docs/MOVI_DE_PHASE5_PAIR_MANIFESTS.md", "manifests/pairs/movi_de/reproducibility_audit.json",
        "manifests/movi_de/protocol_freeze_v0.2.json", "manifests/movi_de/confirmatory_video_pool_freeze.json",
        "manifests/movi_de/hard_negative_definition_freeze.json",
    ]
    for relative in support_paths:
        path = root / relative
        artifacts[relative] = {"sha256": sha256(path)}
    passed = all(compact_checks.values())
    result = {
        "lock_id": "MOVI-DE-POSE-001-PHASE5-PAIRS-v1.0.0", "lock_date": "2026-08-25",
        "seed": 20260825, "status": "locked" if passed else "failed", "total_pairs": total,
        "checks": compact_checks, "artifacts": dict(sorted(artifacts.items())),
        "change_control": "Any change to pair membership, labels, controls, source pools, hard-negative cutoffs, or sampling configuration requires a prospective amendment and replacement lock before model fitting.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("Pair verification failed")
    print(f"Locked {total} pairs: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
