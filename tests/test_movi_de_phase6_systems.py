#!/usr/bin/env python3
"""Tests for frozen MOVi-D/E Phase 6 system definitions."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

import validate_movi_de_phase6_systems as systems  # noqa: E402


class Phase6SystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((ROOT / "configs/movi_de_phase6_systems.json").read_text())

    def test_complete_configuration_passes(self) -> None:
        checks = systems.validate_config(self.config)
        self.assertTrue(all(checks.values()), [name for name, passed in checks.items() if not passed])

    def test_primary_systems_have_equal_dimensions(self) -> None:
        camera = systems.resolve_features(self.config, "C_camera_geometry")
        world = systems.resolve_features(self.config, "D_pose_aligned_geometry")
        self.assertEqual(len(camera), len(world))
        self.assertEqual(len(camera), 31)

    def test_pose_scalars_are_only_in_pose_only_system(self) -> None:
        pose = set(self.config["feature_groups"]["pose_only"])
        for system_id in self.config["systems"]:
            overlap = pose & set(systems.resolve_features(self.config, system_id))
            self.assertEqual(overlap, pose if system_id == "P_pose_only" else set())

    def test_noise_grid_is_full_and_zero_matches_clean_reference(self) -> None:
        conditions = systems.noise_conditions(self.config)
        self.assertEqual(len(conditions), 36)
        self.assertEqual(len({row["condition_id"] for row in conditions}), 36)
        self.assertIn({"condition_id": "N_t0_r0", "translation_std_scene_units": 0.0, "rotation_std_degrees": 0.0}, conditions)
        self.assertEqual(
            systems.resolve_features(self.config, "D_pose_aligned_geometry"),
            systems.resolve_features(self.config, "N_noisy_pose"),
        )

    def test_seeded_shuffle_is_deterministic_and_has_no_fixed_points(self) -> None:
        frame_keys = [("movi_e", "train", str(video), frame) for video in range(3) for frame in range(4)]
        first = systems.deranged_pose_assignment(frame_keys, 20260825)
        second = systems.deranged_pose_assignment(list(reversed(frame_keys)), 20260825)
        self.assertEqual(first, second)
        self.assertTrue(all(source != donor for source, donor in first.items()))
        self.assertEqual(set(first), set(first.values()))

    def test_evaluation_only_fields_are_absent_from_every_model_matrix(self) -> None:
        forbidden = set(self.config["information_boundary"]["evaluation_or_sampling_only_fields"])
        for system_id in self.config["systems"]:
            self.assertFalse(forbidden & set(systems.resolve_features(self.config, system_id)))


if __name__ == "__main__":
    unittest.main()
