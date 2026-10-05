"""Synthetic protocol checks for SW0056 exact data/exposure accounting."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol import (exact_exposure_batches, official_learning_rate,
                      stable_weight_signature, training_ids, training_protocol)


class SW0056ProtocolTest(unittest.TestCase):
    def test_exact_ids_and_ten_real_exposures_with_eight_row_tail(self):
        ids = training_ids()
        self.assertEqual(len(ids), 2500)
        self.assertEqual((int(ids[0]), int(ids[999]), int(ids[1000]), int(ids[-1])), (0, 999, 1640, 3139))
        batches = exact_exposure_batches(seed=7)
        self.assertEqual(len(batches), 1563)
        self.assertTrue(all(len(batch) == 16 for batch in batches[:-1]))
        self.assertEqual(len(batches[-1]), 8)
        np.testing.assert_array_equal(np.unique(np.concatenate(batches), return_counts=True)[0], ids)
        np.testing.assert_array_equal(np.unique(np.concatenate(batches), return_counts=True)[1], np.full(2500, 10))
        self.assertEqual(len(np.concatenate(batches)), 25000)

    def test_schedule_and_protocol_report_actual_partial_batch(self):
        self.assertEqual(official_learning_rate(0), 0.0)
        self.assertGreater(official_learning_rate(156), 0.0)
        p = training_protocol(1)
        self.assertEqual(p["effective_image_exposures"], 25000)
        self.assertEqual(p["optimizer_updates"], 1563)
        self.assertEqual(p["full_batch_updates"], 1562)
        self.assertEqual(p["final_batch_size"], 8)

    def test_determinism_and_validation_range_separation(self):
        a = exact_exposure_batches(seed=3)
        b = exact_exposure_batches(seed=3)
        np.testing.assert_array_equal(np.concatenate(a), np.concatenate(b))
        self.assertFalse(set(training_ids()).intersection(range(1320, 1640)))

    def test_weight_signature_preserves_order_and_normalizes_keras_layer_indices(self):
        class V:
            def __init__(self, name, shape):
                self.name, self.shape = name, shape
        primary = [V("model/conv2d_1/kernel:0", (5, 5, 3, 64)),
                   V("model/conv2d_1/bias:0", (64,))]
        auxiliary = [V("model_4/conv2d_9/kernel:0", (5, 5, 3, 64)),
                     V("model_4/conv2d_9/bias:0", (64,))]
        self.assertEqual(stable_weight_signature(primary), stable_weight_signature(auxiliary))
        with self.assertRaises(AssertionError):
            self.assertEqual(stable_weight_signature(primary), stable_weight_signature(auxiliary[::-1]))


if __name__ == "__main__":
    unittest.main()
