#!/usr/bin/env python3
"""Fast tests for MOVi-D Phase 8 regime 2 helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(ROOT))

import run_movi_d_phase8_in_domain as regime  # noqa: E402


class Phase8Regime2Tests(unittest.TestCase):
    def test_identical_system_scores_have_zero_paired_differences(self) -> None:
        labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int8)
        mask = np.ones(6, dtype=bool)
        scores = np.asarray([0.1, 0.9, 0.4, 0.7, 0.2, 0.6])
        systems = {"C": scores, "D": scores.copy()}
        locked = {key: {"recall_90_threshold": 0.5} for key in systems}
        videos = np.asarray(["1", "1", "2", "2", "3", "3"])
        result = regime.paired_comparisons(("C", "D"), "C", systems, locked, labels, mask, videos, 20260825, 100)
        self.assertEqual(result["D"]["auroc"]["system_minus_reference"], 0.0)
        self.assertEqual(result["D"]["auroc"]["paired_video_cluster_ci_low"], 0.0)
        self.assertEqual(result["D"]["false_match_rate"]["comparison_minus_reference"], 0.0)

    def test_structural_zero_pose_matrix_is_finite(self) -> None:
        matrix = np.zeros((10000, 3), dtype=np.float32)
        self.assertTrue(np.isfinite(matrix).all())
        self.assertEqual(np.unique(matrix).tolist(), [0.0])

    def test_same_eight_clean_systems_are_used(self) -> None:
        self.assertEqual(len(regime.SYSTEM_ORDER), 8)
        self.assertIn("P_pose_only", regime.SYSTEM_ORDER)
        self.assertNotIn("N_noisy_pose", regime.SYSTEM_ORDER)


if __name__ == "__main__": unittest.main()
