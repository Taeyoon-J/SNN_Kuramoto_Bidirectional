"""Synthetic checks for the aligned stage-diagnostic input and AUC helpers."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "collaborative_test" / "SW_0034_stage_signal_diagnostic"))

from diagnose import auc, gamma_rows_for_ids, pair_summary


class AlignedStageSignalTest(unittest.TestCase):
    def test_aligned_validation_ids_map_to_local_rows(self):
        self.assertEqual(gamma_rows_for_ids([1320, 1321, 1639], 1320, 320),
                         [0, 1, 319])

    def test_full_gamma_default_global_origin_remains_compatible(self):
        self.assertEqual(gamma_rows_for_ids([1320, 1639], 0, 10000), [1320, 1639])

    def test_alignment_rejects_missing_global_id(self):
        with self.assertRaisesRegex(ValueError, "does not cover"):
            gamma_rows_for_ids([1319, 1320], 1320, 320)

    def test_synthetic_distance_conditioned_auc(self):
        self.assertEqual(auc([3.0, 4.0], [1.0, 2.0]), 1.0)
        self.assertEqual(auc([1.0, 2.0], [3.0, 4.0]), 0.0)
        self.assertEqual(pair_summary([], [1.0])["auc"], None)


if __name__ == "__main__":
    unittest.main()
