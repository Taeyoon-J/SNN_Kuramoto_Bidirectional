import unittest
import numpy as np
from .analyze import TRAIN_INDICES, paired_changes


class GeneralizationTests(unittest.TestCase):
    def test_raw_train_dev_gap_does_not_establish_overfit(self):
        train = np.full((16, 3), .95)
        dev = np.full((32, 3), .70)
        result = paired_changes(train, train, dev, dev)
        self.assertFalse(result['overfit_consistent_FG_signal'])

    def test_opposite_paired_changes_are_only_consistent_signal(self):
        train = np.full((16, 3), .7)
        dev = np.full((32, 3), .7)
        result = paired_changes(train, train + .05, dev, dev - .03)
        self.assertTrue(result['overfit_consistent_FG_signal'])
        self.assertFalse(result['causal_overfitting_proved'])
        self.assertAlmostEqual(result['TRAIN_minus_development_change']['fg_ari']['mean'], .08)

    def test_both_improve_is_not_development_deterioration(self):
        train = np.full((16, 3), .7)
        dev = np.full((32, 3), .7)
        self.assertFalse(paired_changes(train, train + .1, dev, dev + .01)['overfit_consistent_FG_signal'])

    def test_invalid_or_nonfinite_cohorts_rejected(self):
        train = np.zeros((16, 3)); dev = np.zeros((32, 3))
        with self.assertRaises(ValueError):paired_changes(train[:8], train[:8], dev, dev)
        train[0, 0] = np.nan
        with self.assertRaises(ValueError):paired_changes(train, train, dev, dev)

    def test_fixed_indices_cover_every_committed_proposal_once(self):
        self.assertEqual(TRAIN_INDICES, tuple(4*i for i in range(16)))


if __name__ == '__main__': unittest.main()
