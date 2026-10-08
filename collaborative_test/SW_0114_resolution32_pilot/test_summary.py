import unittest

import numpy as np

from summarize import paired_bootstrap


class SummaryTests(unittest.TestCase):
    def test_fixed_bootstrap_is_reproducible_on_seed_by_shared_image_matrix(self):
        x = np.arange(3 * 320, dtype=np.float64).reshape(3, 320) / 1000.
        a = paired_bootstrap(x, n=50, seed=114)
        b = paired_bootstrap(x, n=50, seed=114)
        self.assertEqual(a, b)
        self.assertEqual(a["sampling_unit"], "shared_image_ids_across_fixed_seeds")

    def test_bootstrap_rejects_wrong_or_nonfinite_shape(self):
        with self.assertRaises(ValueError):
            paired_bootstrap(np.zeros((320, 3)))
        x = np.zeros((3, 320)); x[1, 0] = np.nan
        with self.assertRaises(ValueError):
            paired_bootstrap(x)


if __name__ == "__main__":
    unittest.main()
