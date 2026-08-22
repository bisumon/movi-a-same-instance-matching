#!/usr/bin/env python3
"""Fast tests for Phase 3 feature and threshold helpers."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1] / "src"


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


features = load_module("phase3_features", "build_movi_a_phase3_pair_features.py")
baselines = load_module("phase3_baselines", "run_movi_a_phase3_baselines.py")


def observation(offset: float) -> dict:
    return {
        "mask_center_x_normalized": 0.4 + offset,
        "mask_center_y_normalized": 0.5 - offset,
        "mask_area": 100 + int(offset * 10),
        "padded_crop_width": 20,
        "padded_crop_height": 18,
        "bbox_aspect_ratio": 1.1,
        "visibility": 400,
        "camera_space_visible_surface_centroid_xyz": [offset, 0.2, 8.0 + offset],
        "camera_space_visible_surface_extent_q05_q95_xyz": [1.0, 1.2, 0.8],
        "depth": {
            "q05": 7.5 + offset,
            "q25": 7.8 + offset,
            "median": 8.0 + offset,
            "q75": 8.2 + offset,
            "q95": 8.5 + offset,
        },
    }


class Phase3Tests(unittest.TestCase):
    def test_pair_feature_width_and_finiteness(self) -> None:
        left_embedding = np.ones(512, dtype=np.float32)
        right_embedding = np.full(512, 2.0, dtype=np.float32)
        values = features.pair_features(
            observation(0.0), observation(0.1), left_embedding, right_embedding, 4
        )
        self.assertEqual(len(values), len(features.FEATURE_NAMES))
        self.assertTrue(np.isfinite(values).all())
        self.assertAlmostEqual(values[0], 1.0, places=6)

    def test_feature_names_exclude_leakage_fields(self) -> None:
        for name in features.FEATURE_NAMES:
            for fragment in features.FORBIDDEN_FEATURE_FRAGMENTS:
                self.assertNotIn(fragment, name)

    def test_f1_threshold_is_dev_optimized(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.1, 0.4, 0.6, 0.9])
        threshold, value = baselines.choose_f1_threshold(labels, scores)
        self.assertEqual(threshold, 0.5)
        self.assertEqual(value, 1.0)

    def test_recall_threshold_meets_target(self) -> None:
        labels = np.asarray([0, 0, 1, 1, 1, 1])
        scores = np.asarray([0.1, 0.8, 0.4, 0.5, 0.6, 0.9])
        threshold, recall, false_match_rate = baselines.choose_recall_threshold(
            labels, scores, target_recall=0.75
        )
        self.assertEqual(threshold, 0.45)
        self.assertEqual(recall, 0.75)
        self.assertEqual(false_match_rate, 0.5)


if __name__ == "__main__":
    unittest.main()
