"""Pure tests for SW0056 phase validation and scheduler recovery guards."""
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from load_policy import load_is_acceptable
from scheduler_contract import ORDER, validate_state
from validate_phase import validate_prediction_arrays, validate_loss_rows


class PhaseRecoveryTest(unittest.TestCase):
    def test_loss_rows_reject_partial_nonfinite_or_wrong_tail(self):
        rows = [{"step": str(i), "batch_size": "16", "learning_rate": "0.0001",
                 "reconstruction_mse": "0.2", "elapsed_seconds": str(i)} for i in range(1, 1563)]
        rows.append({"step": "1563", "batch_size": "8", "learning_rate": "0.0001",
                     "reconstruction_mse": "0.1", "elapsed_seconds": "1563"})
        validate_loss_rows(rows, 1563, smoke=False)
        with self.assertRaises(ValueError):
            validate_loss_rows(rows[:-1], 1563, smoke=False)
        bad = [dict(row) for row in rows]; bad[-1]["reconstruction_mse"] = "nan"
        with self.assertRaises(ValueError):
            validate_loss_rows(bad, 1563, smoke=False)

    def test_prediction_contract_requires_exact_ids_shape_and_finite_diagnostics(self):
        ids = np.arange(1320, 1640, dtype=np.int64)
        labels = np.zeros((320, 128, 128), dtype=np.uint8)
        backgrounds = np.zeros(320, dtype=np.int64)
        mse = np.ones(320, dtype=np.float32)
        validate_prediction_arrays(ids, labels, backgrounds, mse)
        with self.assertRaises(ValueError):
            validate_prediction_arrays(ids + 1, labels, backgrounds, mse)
        with self.assertRaises(ValueError):
            validate_prediction_arrays(ids, labels[:319], backgrounds, mse)
        with self.assertRaises(ValueError):
            validate_prediction_arrays(ids, labels, backgrounds, np.full(320, np.nan))

    def test_low_load_limits_are_conjunctive(self):
        self.assertTrue(load_is_acceptable(32, 36, 40, 8 * 1024 * 1024))
        self.assertFalse(load_is_acceptable(32.01, 20, 20, 9 * 1024 * 1024))
        self.assertFalse(load_is_acceptable(20, 20, 20, 8 * 1024 * 1024 - 1))

    def test_scheduler_state_prefix_and_completion(self):
        markers = set(ORDER[:4])
        with patch.object(Path, "is_file", lambda path: path.name in markers):
            self.assertEqual(validate_state(Path("/fake")), list(ORDER[:4]))
            with self.assertRaises(ValueError):
                validate_state(Path("/fake"), complete=True)
            markers.update(ORDER[4:])
            self.assertEqual(validate_state(Path("/fake"), complete=True), list(ORDER))
        markers = {ORDER[0], ORDER[3]}
        with patch.object(Path, "is_file", lambda path: path.name in markers):
            with self.assertRaises(ValueError):
                validate_state(Path("/fake"))


if __name__ == "__main__":
    unittest.main()
