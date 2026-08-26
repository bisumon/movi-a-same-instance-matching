#!/usr/bin/env python3
"""Fast tests for MOVi-E Phase 8 regime 1 helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(ROOT))

import build_movi_e_phase8_in_domain_features as features  # noqa: E402
import run_movi_e_phase8_in_domain as regime  # noqa: E402


class Phase8Regime1Tests(unittest.TestCase):
    def test_shuffled_rigid_summary_preserves_centroid_distances_with_shared_donor(self) -> None:
        rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        donor_position = np.asarray([4.0, 2.0, 1.0])
        def row(center):
            return {
                "pose_aligned_world_visible_surface_centroid_xyz": center,
                "pose_aligned_world_visible_surface_extent_q05_q95_xyz": [1.0, 2.0, 3.0],
                "camera_pose": {"position_world_xyz": [0.0, 0.0, 0.0], "camera_to_world_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
            }
        first, _ = features.shuffled_summary(row([1.0, 2.0, 3.0]), donor_position, rotation)
        second, _ = features.shuffled_summary(row([2.0, 1.0, 4.0]), donor_position, rotation)
        self.assertAlmostEqual(np.linalg.norm(first - second), np.sqrt(3.0), places=12)

    def test_calibration_is_zero_for_perfect_probabilities(self) -> None:
        labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
        scores = labels.astype(np.float64)
        value = regime.calibration(labels, scores)
        self.assertEqual(value["brier_score"], 0.0)
        self.assertEqual(value["expected_calibration_error_10_bins"], 0.0)

    def test_identical_operating_predictions_have_zero_paired_interval(self) -> None:
        labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int8)
        predictions = np.asarray([0, 1, 1, 1, 0, 0], dtype=bool)
        videos = np.asarray(["1", "1", "2", "2", "3", "3"])
        for target in (0, 1):
            result = regime.paired_rate_interval(labels, predictions, predictions, videos, target, 20260825, 100)
            self.assertEqual(result["comparison_minus_reference"], 0.0)
            self.assertEqual(result["paired_video_cluster_ci_low"], 0.0)
            self.assertEqual(result["paired_video_cluster_ci_high"], 0.0)

    def test_system_order_excludes_noisy_pose_family(self) -> None:
        self.assertNotIn("N_noisy_pose", regime.SYSTEM_ORDER)
        self.assertIn("S_shuffled_pose", regime.SYSTEM_ORDER)
        self.assertEqual(len(regime.SYSTEM_ORDER), 8)


if __name__ == "__main__":
    unittest.main()
