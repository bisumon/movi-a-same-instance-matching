#!/usr/bin/env python3
"""Independently verify and checksum-lock the Phase 10 failure analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


SYSTEMS = ("C_camera_geometry", "D_pose_aligned_geometry")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def classify(label: int, score: float, threshold: float) -> str | None:
    predicted = score >= threshold
    if label == 0 and predicted:
        return "false_positive"
    if label == 1 and not predicted:
        return "false_negative"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    gallery = root / "failure_gallery/movi_de_phase10"
    selected = read_jsonl(gallery / "selected_failure_pairs.jsonl")
    manifest = json.loads((gallery / "selection_manifest.json").read_text(encoding="utf-8"))
    capacity = json.loads((gallery / "capacity_audit.json").read_text(encoding="utf-8"))

    pair_maps, prediction_maps, thresholds = {}, {}, {}
    for dataset, regime in (("movi_d", "regime2"), ("movi_e", "regime1")):
        prefix = "movi_d_phase8_regime2" if dataset == "movi_d" else "movi_e_phase8_regime1"
        base = root / f"runs/movi_de_confirmatory/phase8_{regime}/in_domain_{dataset}"
        pair_maps[dataset] = {row["pair_id"]: row for row in read_jsonl(root / f"manifests/pairs/movi_de/{dataset}_all_pairs.jsonl")}
        prediction_maps[dataset] = {row["pair_id"]: row for row in read_jsonl(base / f"{prefix}_predictions.jsonl")}
        lock = json.loads((base / f"{prefix}_locked_config.json").read_text(encoding="utf-8"))
        thresholds[dataset] = {system: float(lock["systems"][system]["recall_90_threshold"]) for system in SYSTEMS}

    score_checks, identity_checks, threshold_checks, error_checks, asset_checks = [], [], [], [], []
    for row in selected:
        dataset, pair_id = row["dataset"], row["pair_id"]
        pair, prediction = pair_maps[dataset][pair_id], prediction_maps[dataset][pair_id]
        identity_checks.append(pair["split"] == "test" and row["label"] == pair["label"] and str(row["video_id"]) == str(pair["video_id"]))
        score_checks.append(all(row["scores"][system] == prediction["scores"][system] for system in SYSTEMS))
        threshold_checks.append(all(row["thresholds"][system] == thresholds[dataset][system] for system in SYSTEMS))
        error_checks.append(classify(row["label"], row["scores"][row["system"]], row["thresholds"][row["system"]]) == row["error_type"])
        for side in ("a", "b"):
            path = gallery / row[f"gallery_crop_path_{side}"]
            with Image.open(path) as image:
                asset_checks.append(image.size == (96, 96) and image.format == "PNG")

    balance = {
        "datasets": Counter(row["dataset"] for row in selected),
        "systems": Counter(row["system"] for row in selected),
        "errors": Counter(row["error_type"] for row in selected),
        "cells": Counter(f"{row['dataset']}|{row['system']}|{row['error_type']}" for row in selected),
        "dynamics": Counter(row["dynamic_group"] for row in selected),
        "difficulty": Counter(row["negative_difficulty"] for row in selected if row["error_type"] == "false_positive"),
        "gaps": Counter(row["temporal_gap_bin"] for row in selected),
        "e_motion": Counter(row["motion_stratum"] for row in selected if row["dataset"] == "movi_e"),
    }
    html = (gallery / "failure_gallery.html").read_text(encoding="utf-8")
    with Image.open(gallery / "contact_sheet.png") as preview:
        preview_valid = preview.size == (880, 792) and preview.format == "PNG"
    expected_cells = Counter({f"{dataset}|{system}|{error}": 3 for dataset in ("movi_d", "movi_e") for system in SYSTEMS for error in ("false_positive", "false_negative")})
    checks = {
        "selection_manifest_status_pass": manifest["status"] == "pass" and all(manifest["checks"].values()),
        "capacity_audit_status_pass": capacity["status"] == "pass" and all(int(value) > 0 for value in capacity["initial_candidate_capacity_per_slot"].values()),
        "exact_24_unique_pair_ids": len(selected) == 24 and len({row["pair_id"] for row in selected}) == 24,
        "all_pair_identity_fields_exact": all(identity_checks),
        "all_C_D_scores_exact": all(score_checks),
        "all_locked_thresholds_exact": all(threshold_checks),
        "all_assigned_items_are_errors": all(error_checks),
        "exact_dataset_balance": balance["datasets"] == Counter({"movi_d": 12, "movi_e": 12}),
        "exact_system_balance": balance["systems"] == Counter({SYSTEMS[0]: 12, SYSTEMS[1]: 12}),
        "exact_FP_FN_balance": balance["errors"] == Counter({"false_positive": 12, "false_negative": 12}),
        "exact_three_per_primary_cell": balance["cells"] == expected_cells,
        "exact_static_dynamic_balance": balance["dynamics"] == Counter({"static": 12, "dynamic": 12}),
        "exact_hard_easy_FP_balance": balance["difficulty"] == Counter({"hard": 6, "easy": 6}),
        "exact_short_medium_long_balance": balance["gaps"] == Counter({"short_1_5": 8, "medium_6_11": 8, "long_12_23": 8}),
        "MOVi_E_motion_capacity_balance": balance["e_motion"] == Counter({"low": 5, "medium": 2, "high": 5}),
        "all_48_assets_valid": len(asset_checks) == 48 and all(asset_checks),
        "HTML_has_24_cards_and_48_images": html.count('<article class="card"') == 24 and html.count('<img src="assets/') == 48,
        "contact_sheet_valid": preview_valid,
        "all_diagnoses_one_line_and_tagged": all(row["diagnosis_category"] and row["diagnosis"] and "\n" not in row["diagnosis"] for row in selected),
        "no_same_asset_test_negative_capacity": all(value["same_asset_test_negative_pairs"] == 0 for value in capacity["dataset_error_capacity"].values()),
    }
    paths = [
        root / "docs/MOVI_DE_CAMERA_POSE_PROTOCOL_v0.2.md",
        root / "docs/MOVI_DE_PHASE10_FAILURE_ANALYSIS.md",
        root / "src/select_movi_de_phase10_failures.py",
        root / "src/freeze_movi_de_phase10_failures.py",
        root / "tests/test_movi_de_phase10_failures.py",
        root / "manifests/movi_de/phase8_regime1_in_domain_movi_e_freeze.json",
        root / "manifests/movi_de/phase8_regime2_in_domain_movi_d_freeze.json",
        gallery / "selected_failure_pairs.jsonl",
        gallery / "failure_review.csv",
        gallery / "failure_gallery.html",
        gallery / "contact_sheet.png",
        gallery / "capacity_audit.json",
        gallery / "selection_manifest.json",
        *sorted((gallery / "assets").glob("*.png")),
    ]
    artifacts = {str(path.relative_to(root)): {"sha256": sha256(path), "size_bytes": path.stat().st_size} for path in paths}
    status = "locked" if all(checks.values()) else "failed"
    lock = {
        "lock_id": "MOVI-DE-POSE-001-PHASE10-v1.0.0", "lock_date": "2026-08-25", "status": status,
        "seed": 20260825, "selection_count": 24, "operating_point": "per-system MOVi-D/E development-selected 90%-recall thresholds",
        "checks": checks, "balance": {name: dict(value) for name, value in balance.items()},
        "artifacts": dict(sorted(artifacts.items())),
        "change_control": "Failure analysis is frozen and qualitative. It cannot alter Phase 7-9 predictions, thresholds, metrics, confidence intervals, or conclusions.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if status != "locked":
        raise RuntimeError(f"Freeze failed: {[name for name, passed in checks.items() if not passed]}")
    print(f"Locked Phase 10 with {len(artifacts)} checksummed artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
