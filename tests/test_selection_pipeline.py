#!/usr/bin/env python3
"""Smoke test for deterministic selection and exact video-disjoint splits."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "src" / "movi_a_video_selection.py"


def record(video_id, instances, matching_pair=True):
    visibility = [[64] * 24 for _ in range(instances)]
    shapes = ["cube"] * instances
    colors = ["red"] * instances
    materials = ["metal"] * instances
    if not matching_pair:
        colors = [f"color-{index}" for index in range(instances)]
    return {
        "video_id": video_id,
        "visibility": visibility,
        "shape_label": shapes,
        "color_label": colors,
        "material_label": materials,
    }


with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    inventory = tmp_path / "inventory.jsonl"
    with inventory.open("w", encoding="utf-8") as handle:
        for index in range(80):
            handle.write(json.dumps(record(f"video-{index:03d}", 3 + index % 8)) + "\n")
        handle.write(json.dumps(record("ineligible-no-hard-negative", 3, matching_pair=False)) + "\n")
    output = tmp_path / "out"
    command = [
        sys.executable,
        str(SCRIPT),
        "select",
        "--input-jsonl",
        str(inventory),
        "--output-dir",
        str(output),
        "--min-hard-negative-candidates",
        "2",
    ]
    subprocess.run(command, check=True)
    rows = [json.loads(line) for line in (output / "video_splits.jsonl").read_text().splitlines()]
    assert len(rows) == 50
    assert len({row["video_id"] for row in rows}) == 50
    assert {row["split"] for row in rows} == {"train", "dev", "test"}
    assert sum(row["split"] == "train" for row in rows) == 30
    assert sum(row["split"] == "dev" for row in rows) == 10
    assert sum(row["split"] == "test" for row in rows) == 10
    for object_bin in {row["object_bin"] for row in rows}:
        bin_rows = [row for row in rows if row["object_bin"] == object_bin]
        if len(bin_rows) >= 3:
            assert {row["split"] for row in bin_rows} == {"train", "dev", "test"}
    print("selection-pipeline smoke test passed")
