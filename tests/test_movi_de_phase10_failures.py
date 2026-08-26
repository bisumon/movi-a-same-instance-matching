#!/usr/bin/env python3
"""Fast tests for MOVi-D/E Phase 10 failure-selection helpers."""

from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
import select_movi_de_phase10_failures as phase10  # noqa: E402


class Phase10FailureTests(unittest.TestCase):
    def test_error_classification_uses_greater_equal_threshold(self) -> None:
        self.assertEqual(phase10.classify_error(0, 0.5, 0.5), "false_positive")
        self.assertEqual(phase10.classify_error(1, 0.49, 0.5), "false_negative")
        self.assertIsNone(phase10.classify_error(1, 0.5, 0.5))

    def test_exact_primary_slot_balance(self) -> None:
        values = phase10.slots()
        self.assertEqual(len(values), 24)
        cells = Counter((row.dataset, row.system, row.error_type) for row in values)
        self.assertTrue(all(count == 3 for count in cells.values()))
        self.assertEqual(len(cells), 8)

    def test_slot_level_static_dynamic_and_difficulty_balance(self) -> None:
        values = phase10.slots()
        self.assertEqual(Counter(row.dynamic_group for row in values), Counter({"static": 12, "dynamic": 12}))
        false_positives = [row for row in values if row.error_type == "false_positive"]
        self.assertEqual(Counter(row.negative_difficulty for row in false_positives), Counter({"easy": 6, "hard": 6}))


if __name__ == "__main__":
    unittest.main()
