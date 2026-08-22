#!/usr/bin/env python3
"""Fast unit tests for Phase 2 pair generation helpers."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "src" / "generate_movi_a_phase2_pairs.py"
SPEC = importlib.util.spec_from_file_location("phase2_pairs", SCRIPT)
assert SPEC and SPEC.loader
phase2 = importlib.util.module_from_spec(SPEC)
sys.modules["phase2_pairs"] = phase2
SPEC.loader.exec_module(phase2)


class Phase2PairTests(unittest.TestCase):
    def test_gap_bins(self) -> None:
        self.assertEqual(phase2.gap_bin(2, 5, 11), "short")
        self.assertEqual(phase2.gap_bin(5, 5, 11), "short")
        self.assertEqual(phase2.gap_bin(6, 5, 11), "medium")
        self.assertEqual(phase2.gap_bin(11, 5, 11), "medium")
        self.assertEqual(phase2.gap_bin(12, 5, 11), "long")

    def test_even_allocation(self) -> None:
        self.assertEqual(
            phase2.even_allocation(1000),
            {"short": 334, "medium": 333, "long": 333},
        )

    def test_pair_id_is_order_and_split_specific(self) -> None:
        left = phase2.Observation("a", "1", "train", 1, 0, 100, 50, 12, 10)
        right = phase2.Observation("b", "1", "train", 3, 0, 90, 45, 11, 9)
        candidate = phase2.Candidate(left, right, "positive", 2, "short")
        self.assertEqual(
            phase2.stable_pair_id("train", candidate),
            phase2.stable_pair_id("train", candidate),
        )
        self.assertNotEqual(
            phase2.stable_pair_id("train", candidate),
            phase2.stable_pair_id("dev", candidate),
        )

    def test_control_vector_is_finite(self) -> None:
        left = phase2.Observation("a", "1", "train", 1, 0, 100, 50, 12, 10)
        right = phase2.Observation("b", "1", "train", 3, 1, 90, 45, 11, 9)
        candidate = phase2.Candidate(left, right, "hard", 2, "short")
        self.assertTrue(np.isfinite(phase2.control_vector(candidate)).all())


if __name__ == "__main__":
    unittest.main()
