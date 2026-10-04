import unittest

import numpy as np

from hybrid import (compact_labels, gamma_local_rows, generic_spectral_partition,
                    preserve_foreground, restricted_spectral_labels)


class HybridReadoutTest(unittest.TestCase):
    def test_freeze_preserves_exact_mask_and_compacts_labels(self):
        fg = np.array([1, 0, 1, 1, 0, 1], dtype=bool)
        labels = np.array([9, 2, 9, 4, 8, 4])
        pred = preserve_foreground(labels, fg)
        np.testing.assert_array_equal(pred != 0, fg)
        np.testing.assert_array_equal(pred, [2, 0, 2, 1, 0, 1])

    def test_restricted_clustering_never_changes_frozen_mask(self):
        fg = np.array([1, 0, 1, 0, 1], dtype=bool)
        affinity = np.eye(5)
        def fake_spectral(sub, k):
            self.assertEqual(sub.shape, (3, 3))
            self.assertEqual(k, 3)
            return np.array([7, 7, 12])
        pred = restricted_spectral_labels(affinity, fg, 10, fake_spectral)
        np.testing.assert_array_equal(pred != 0, fg)
        np.testing.assert_array_equal(pred, [1, 0, 1, 0, 2])

    def test_invalid_zero_fg_label_rejected(self):
        with self.assertRaises(ValueError):
            preserve_foreground(np.array([0, 1]), np.array([True, False]))

    def test_real_generic_spectral_partition_supports_arbitrary_size(self):
        # Seven foreground nodes deliberately exercise a size that is not a
        # 16x16 grid and therefore cannot pass through the legacy reshaper.
        affinity = np.exp(-np.abs(np.arange(7)[:, None] - np.arange(7)[None, :]))
        labels = generic_spectral_partition(affinity, 3)
        self.assertEqual(labels.shape, (7,))
        self.assertTrue(np.issubdtype(labels.dtype, np.integer))
        self.assertTrue(np.all(labels >= 1))
        self.assertLessEqual(np.unique(labels).size, 3)

    def test_gamma_row_mapping_aligned_full_and_invalid(self):
        self.assertEqual(gamma_local_rows(320, 1320, 320), list(range(320)))
        self.assertEqual(gamma_local_rows(320, 1320, 4), list(range(4)))
        self.assertEqual(gamma_local_rows(10000, 1320, 320), list(range(1320, 1640)))
        with self.assertRaises(ValueError):
            gamma_local_rows(300, 1320, 320)


if __name__ == "__main__":
    unittest.main()
