from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from collaborative_test.SW_0154_native64_resolution_compare import score_slot64


class Slot64ScoreTests(unittest.TestCase):
    def test_gt_is_not_opened_until_prediction_preparation_succeeds(self):
        dataset = Path("unused_nonexistent_gt.h5")
        with mock.patch.object(score_slot64, "prepare_all_slot_predictions",
                               side_effect=ValueError("tampered prediction")), \
             mock.patch.object(score_slot64.h5py, "File") as open_hdf5:
            with self.assertRaisesRegex(ValueError, "tampered prediction"):
                score_slot64.score_slot_baseline(dataset=dataset, slot_root=Path("unused"))
            open_hdf5.assert_not_called()

    def test_fixed320_native64_metrics_use_production_metric_names(self):
        pred = np.zeros((320, 64, 64), dtype=np.int64)
        pred[:, 16:48, 16:48] = 1
        target = pred.copy()
        target[:, 16:18, 16:18] = 0
        metrics = score_slot64._production_metrics64(pred, target)
        self.assertEqual(set(metrics), {"fg_ari", "foreground_iou", "matched_object_iou"})
        self.assertTrue(all(value["valid_count"] == 320 for value in metrics.values()))
        with self.assertRaises(ValueError):
            score_slot64._production_metrics64(pred[:319], target[:319])

    def test_fingerprint_resolves_exact_slot_and_evaluation_dependencies(self):
        record = score_slot64.implementation_fingerprint()
        expected = {
            "collaborative_test/SW_0154_native64_resolution_compare/score_slot64.py",
            "collaborative_test/SW_0154_native64_resolution_compare/resolution64.py",
            "collaborative_test/SW_0135_native32_spike_binding/evaluation_contract32.py",
            "snn_kuramoto_bidirectional/evaluation.py",
        }
        self.assertEqual(set(record), expected)
        self.assertTrue(all(len(value) == 64 for value in record.values()))


if __name__ == "__main__":
    unittest.main()
