"""Pure NumPy tests for finite-budget schedule and complete pass coverage."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from protocol import make_epoch_balanced_batches, official_learning_rate, training_protocol


class MatchedSlotProtocolTest(unittest.TestCase):
    def test_learning_rate_matches_official_warmup_and_exponential_decay(self):
        self.assertEqual(official_learning_rate(0), 0.0)
        self.assertAlmostEqual(official_learning_rate(125), 2e-4 * (0.5 ** (125 / 2500)))
        self.assertAlmostEqual(official_learning_rate(250), 4e-4 * (0.5 ** 0.1))
        self.assertAlmostEqual(official_learning_rate(2500), 2e-4)
        with self.assertRaises(ValueError):
            official_learning_rate(-1)

    def test_seeded_batches_cover_every_training_id_exactly_40_times(self):
        import numpy as np
        batches = make_epoch_balanced_batches(1000, batch_size=16, steps=2500, seed=1)
        self.assertEqual(batches.shape, (2500, 16))
        counts = np.bincount(batches.reshape(-1), minlength=1000)
        self.assertTrue((counts == 40).all())

    def test_batch_stream_is_deterministic_and_seed_specific(self):
        import numpy as np
        a = make_epoch_balanced_batches(40, batch_size=4, steps=12, seed=2)
        b = make_epoch_balanced_batches(40, batch_size=4, steps=12, seed=2)
        c = make_epoch_balanced_batches(40, batch_size=4, steps=12, seed=3)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_protocol_declares_complete_pass_budget(self):
        protocol = training_protocol(0)
        self.assertEqual(protocol["effective_image_passes"], 40.0)
        self.assertEqual(protocol["per_image_occurrences"], {"min": 40, "max": 40, "mean": 40.0})


if __name__ == "__main__":
    unittest.main()
