#!/usr/bin/env python3
"""Tests for the Phase 2 split-lock gate."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "src" / "prepare_movi_a_phase2_split.py"
SPEC = importlib.util.spec_from_file_location("phase2_split", SCRIPT)
assert SPEC and SPEC.loader
phase2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(phase2)


def fixture() -> tuple[list[dict], list[dict], list[dict]]:
    assignments = ["train"] * 30 + ["dev"] * 10 + ["test"] * 10
    selected = [{"video_id": str(index)} for index in range(50)]
    splits = [
        {"video_id": str(index), "split": split}
        for index, split in enumerate(assignments)
    ]
    observations = [
        {
            "observation_id": f"observation-{index}",
            "video_id": str(index),
            "split": split,
        }
        for index, split in enumerate(assignments)
    ]
    return selected, splits, observations


class Phase2SplitTests(unittest.TestCase):
    def test_valid_30_10_10_split(self) -> None:
        selected, splits, observations = fixture()
        locked, summary = phase2.validate_and_lock(selected, splits, observations)
        self.assertEqual(len(locked), 50)
        self.assertEqual(summary["video_counts"], {"train": 30, "dev": 10, "test": 10})
        self.assertTrue(summary["video_disjoint"])

    def test_rejects_observation_split_mismatch(self) -> None:
        selected, splits, observations = fixture()
        observations[0]["split"] = "test"
        with self.assertRaises(ValueError):
            phase2.validate_and_lock(selected, splits, observations)

    def test_rejects_wrong_video_counts(self) -> None:
        selected, splits, observations = fixture()
        splits[29]["split"] = "dev"
        observations[29]["split"] = "dev"
        with self.assertRaises(ValueError):
            phase2.validate_and_lock(selected, splits, observations)


if __name__ == "__main__":
    unittest.main()
