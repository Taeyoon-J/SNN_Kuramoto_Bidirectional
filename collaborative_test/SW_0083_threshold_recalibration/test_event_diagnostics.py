import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from event_diagnostics import reshape_binary_crossings


class BinaryCrossingShapeTests(unittest.TestCase):
    def test_stepwise_thresholds_reshape_to_component_grid_and_are_binary(self):
        # B=2, D=2, N=3, T=4 encoded as [B*D,N] at each hook call.
        captured = [np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
                    for _ in range(4)]
        result = reshape_binary_crossings(captured, 2, 2, 3, 4)
        self.assertEqual(result.shape, (2, 2, 3, 4))
        self.assertEqual(result.dtype, np.bool_)
        self.assertTrue(np.isin(result, (False, True)).all())

    def test_rejects_missing_steps_wrong_shape_and_continuous_spikes(self):
        valid = np.zeros((1, 4), dtype=bool)
        with self.assertRaises(ValueError):
            reshape_binary_crossings([valid], 1, 1, 4, 2)
        with self.assertRaises(ValueError):
            reshape_binary_crossings([np.zeros((1, 4), dtype=bool)] * 2, 1, 2, 2, 2)
        with self.assertRaises(ValueError):
            reshape_binary_crossings([np.full((1, 4), .2)] * 2, 1, 1, 4, 2)


if __name__ == "__main__":
    unittest.main()
