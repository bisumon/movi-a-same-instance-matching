#!/usr/bin/env python3
"""Fast deterministic tests for the MOVi-D/E Phase 7 pose-noise study."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("phase7_noise", ROOT / "run_movi_de_phase7_pose_noise.py")
assert spec and spec.loader
phase7 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = phase7
spec.loader.exec_module(phase7)


def observation(center: list[float], extent: list[float]) -> dict:
    return {
        "pose_aligned_world_visible_surface_centroid_xyz": center,
        "pose_aligned_world_visible_surface_extent_q05_q95_xyz": extent,
        "camera_pose": {
            "position_world_xyz": [2.0, -1.0, 3.0],
            "camera_to_world_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
    }


class Phase7NoiseTests(unittest.TestCase):
    def test_zero_noise_is_exact(self) -> None:
        row = observation([2.5, -0.5, 8.0], [1.0, 2.0, 3.0])
        condition = {"condition_id": "N_t0_r0", "translation_std_scene_units": 0.0, "rotation_std_degrees": 0.0}
        center, extent = phase7.noisy_summary(row, "movi_e", "train", "17", 4, condition, 20260825)
        self.assertTrue(np.array_equal(center, np.asarray(row["pose_aligned_world_visible_surface_centroid_xyz"])))
        self.assertTrue(np.array_equal(extent, np.asarray(row["pose_aligned_world_visible_surface_extent_q05_q95_xyz"])))

    def test_frame_noise_is_deterministic_and_reused(self) -> None:
        condition = {"condition_id": "N_t0p1_r2", "translation_std_scene_units": 0.1, "rotation_std_degrees": 2.0}
        first = observation([2.5, -0.5, 8.0], [1.0, 2.0, 3.0])
        second = observation([3.0, 0.0, 7.0], [0.5, 1.0, 2.0])
        c1, _ = phase7.noisy_summary(first, "movi_e", "test", "17", 4, condition, 20260825)
        c1_again, _ = phase7.noisy_summary(first, "movi_e", "test", "17", 4, condition, 20260825)
        c2, _ = phase7.noisy_summary(second, "movi_e", "test", "17", 4, condition, 20260825)
        self.assertTrue(np.array_equal(c1, c1_again))
        clean_delta = np.asarray(first["pose_aligned_world_visible_surface_centroid_xyz"]) - np.asarray(second["pose_aligned_world_visible_surface_centroid_xyz"])
        noisy_delta = c1 - c2
        self.assertAlmostEqual(np.linalg.norm(clean_delta), np.linalg.norm(noisy_delta), places=10)

    def test_different_conditions_use_different_substreams(self) -> None:
        row = observation([2.5, -0.5, 8.0], [1.0, 2.0, 3.0])
        a = {"condition_id": "N_t0p1_r1", "translation_std_scene_units": 0.1, "rotation_std_degrees": 1.0}
        b = {"condition_id": "N_t0p1_r2", "translation_std_scene_units": 0.1, "rotation_std_degrees": 2.0}
        center_a, _ = phase7.noisy_summary(row, "movi_e", "train", "17", 4, a, 20260825)
        center_b, _ = phase7.noisy_summary(row, "movi_e", "train", "17", 4, b, 20260825)
        self.assertFalse(np.array_equal(center_a, center_b))

    def test_geometry_feature_width(self) -> None:
        values = phase7.geometry_features(
            np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 1.5, 2.0]),
            np.asarray([2.0, 1.0, 4.0]), np.asarray([1.2, 1.0, 2.2]), 5,
        )
        self.assertEqual(len(values), len(phase7.GEOMETRY_NAMES))
        self.assertTrue(np.isfinite(values).all())


if __name__ == "__main__":
    unittest.main()
