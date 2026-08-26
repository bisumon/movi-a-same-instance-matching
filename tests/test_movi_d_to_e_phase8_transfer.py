#!/usr/bin/env python3
"""Fast tests for Phase 8 regime 3 transfer helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
import run_movi_d_to_e_phase8_transfer as transfer  # noqa: E402


class Phase8Regime3Tests(unittest.TestCase):
    def test_identical_ranking_comparison_is_zero(self) -> None:
        labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int8)
        scores = np.asarray([0.1, 0.8, 0.2, 0.9, 0.3, 0.7])
        videos = np.asarray(["1", "1", "2", "2", "3", "3"])
        result = transfer.comparison(labels, scores, 0.5, scores, 0.5, videos, 20260825, 100)
        self.assertEqual(result["auroc"]["transfer_minus_reference"], 0.0)
        self.assertEqual(result["false_match_rate"]["comparison_minus_reference"], 0.0)
        self.assertEqual(result["recall"]["comparison_minus_reference"], 0.0)

    def test_transfer_system_is_clean_D_only(self) -> None:
        self.assertEqual(transfer.SYSTEM_ID, "D_pose_aligned_geometry")


if __name__ == "__main__":
    unittest.main()
