#!/usr/bin/env python3
"""Fast unit tests for the Phase 4 evaluator."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1] / "src"
spec = importlib.util.spec_from_file_location("phase4_evaluation", ROOT / "evaluate_movi_a_phase4.py")
assert spec and spec.loader
phase4 = importlib.util.module_from_spec(spec)
sys.modules["phase4_evaluation"] = phase4
spec.loader.exec_module(phase4)


class Phase4Tests(unittest.TestCase):
    def test_metrics_use_locked_thresholds(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.1, 0.7, 0.6, 0.9])
        result = phase4.evaluate(labels, scores, f1_threshold=0.5, recall_threshold=0.5)
        self.assertAlmostEqual(result["auroc"], 0.75)
        self.assertAlmostEqual(result["f1_at_locked_threshold"], 0.8)
        self.assertAlmostEqual(result["false_match_rate_at_locked_90_recall_threshold"], 0.5)
        self.assertAlmostEqual(result["achieved_recall_at_locked_90_recall_threshold"], 1.0)

    def test_train_tertiles_are_fixed_for_application(self) -> None:
        cutpoints = phase4.train_tertile_cutpoints(np.asarray([1, 2, 3, 4, 5, 6], dtype=float))
        assigned = phase4.assign_tertiles(np.asarray([0, cutpoints[0], 3.5, cutpoints[1], 99]), cutpoints)
        self.assertEqual(assigned.tolist(), ["low", "low", "medium", "medium", "high"])

    def test_hard_and_easy_comparisons_reuse_all_positives(self) -> None:
        labels = np.asarray([1, 1, 0, 0, 0, 0])
        difficulties = np.asarray(["positive", "positive", "hard", "easy", "hard", "easy"])
        hard = phase4.hard_easy_mask(labels, difficulties, "hard")
        easy = phase4.hard_easy_mask(labels, difficulties, "easy")
        self.assertTrue(np.array_equal(np.flatnonzero(hard & (labels == 1)), [0, 1]))
        self.assertTrue(np.array_equal(np.flatnonzero(easy & (labels == 1)), [0, 1]))
        self.assertEqual(int(hard.sum()), 4)
        self.assertEqual(int(easy.sum()), 4)


if __name__ == "__main__":
    unittest.main()
