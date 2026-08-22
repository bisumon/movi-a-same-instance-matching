#!/usr/bin/env python3
"""Fast tests for the MOVi-A Phase 1 extractor."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).parents[1] / "src" / "extract_movi_a_phase1.py"
SPEC = importlib.util.spec_from_file_location("phase1", SCRIPT)
assert SPEC and SPEC.loader
phase1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(phase1)


class Phase1UnitTests(unittest.TestCase):
    def test_depth_decode_endpoints(self) -> None:
        encoded = np.asarray([[0, 65535]], dtype=np.uint16)
        with tempfile.NamedTemporaryFile(suffix=".png") as handle:
            Image.fromarray(encoded).save(handle.name)
            blob = Path(handle.name).read_bytes()
        decoded = phase1.decode_depth(blob, np.asarray([7.0, 67.0]))
        np.testing.assert_allclose(decoded, [[7.0, 67.0]], atol=1e-9)

    def test_padded_bbox_clips_to_frame(self) -> None:
        mask = np.zeros((10, 12), dtype=bool)
        mask[0:4, 8:12] = True
        tight, padded = phase1.padded_bbox(mask, 0.25)
        self.assertEqual(tight, (8, 0, 12, 4))
        self.assertEqual(padded, (7, 0, 12, 5))

    def test_backprojection_uses_radial_depth(self) -> None:
        mask = np.zeros((2, 2), dtype=bool)
        mask[0, 0] = True
        depth = np.full((2, 2), 10.0)
        point = phase1.masked_camera_points(mask, depth, 1.0, 1.0)[0]
        self.assertAlmostEqual(float(np.linalg.norm(point)), 10.0)

    def test_identity_quaternion(self) -> None:
        rotation = phase1.quaternion_to_rotation_matrix(np.asarray([1.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-12)

    def test_leakage_guard_rejects_ground_truth(self) -> None:
        with self.assertRaises(ValueError):
            phase1.validate_model_record({"gt_world_position_xyz": [0, 0, 0]})

    def test_stable_observation_id(self) -> None:
        first = phase1.stable_observation_id("3835", 0, 1)
        second = phase1.stable_observation_id("3835", 0, 1)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 20)


if __name__ == "__main__":
    unittest.main()
