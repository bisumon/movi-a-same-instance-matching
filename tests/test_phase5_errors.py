#!/usr/bin/env python3
"""Fast tests for the Phase 5 error-review selector."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).parents[1] / "src"
spec = importlib.util.spec_from_file_location("phase5_errors", ROOT / "select_movi_a_phase5_errors.py")
assert spec and spec.loader
phase5 = importlib.util.module_from_spec(spec)
sys.modules["phase5_errors"] = phase5
spec.loader.exec_module(phase5)


class Phase5Tests(unittest.TestCase):
    def test_fixed_threshold_error_classification(self) -> None:
        self.assertEqual(phase5.classify_error(0, 0.8, 0.5), "false_positive")
        self.assertEqual(phase5.classify_error(1, 0.2, 0.5), "false_negative")
        self.assertIsNone(phase5.classify_error(1, 0.8, 0.5))
        self.assertIsNone(phase5.classify_error(0, 0.2, 0.5))

    def test_slots_are_exactly_balanced(self) -> None:
        slots = phase5.selection_slots()
        self.assertEqual(len(slots), 24)
        self.assertEqual(Counter(slot.configuration for slot in slots), Counter({method: 6 for method in phase5.CONFIG_ORDER}))
        self.assertEqual(Counter(slot.error_type for slot in slots), Counter({"false_positive": 12, "false_negative": 12}))
        self.assertEqual(Counter(slot.temporal_gap_bin for slot in slots), Counter({gap: 8 for gap in phase5.GAP_ORDER}))
        self.assertEqual(Counter(slot.negative_difficulty for slot in slots if slot.error_type == "false_positive"), Counter({"hard": 6, "easy": 6}))

    def test_selector_returns_unique_pair_per_slot(self) -> None:
        candidates = []
        pair_number = 0
        for slot in phase5.selection_slots():
            pair_number += 1
            candidates.append(
                {
                    "pair_id": f"pair-{pair_number}",
                    "configuration": slot.configuration,
                    "error_type": slot.error_type,
                    "temporal_gap_bin": slot.temporal_gap_bin,
                    "negative_difficulty": slot.negative_difficulty,
                    "visibility_stratum": ("low", "medium", "high")[pair_number % 3],
                    "motion_stratum": ("high", "medium", "low")[pair_number % 3],
                    "video_id": str(pair_number % 10),
                    "threshold_margin": pair_number / 100,
                }
            )
        selected = phase5.select_candidates(candidates)
        self.assertEqual(len(selected), 24)
        self.assertEqual(len({row["pair_id"] for row in selected}), 24)


if __name__ == "__main__":
    unittest.main()
