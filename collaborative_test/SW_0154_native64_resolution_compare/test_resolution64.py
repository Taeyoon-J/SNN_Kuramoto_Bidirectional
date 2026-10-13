from __future__ import annotations

import unittest

import numpy as np

from collaborative_test.SW_0154_native64_resolution_compare.resolution64 import (
    GRID, NODES, OLD_NODES, map_node_parameter, map_pair_parameter, parent_index,
)


class Native64MappingTests(unittest.TestCase):
    def test_parent_map_is_exact_row_major_four_by_four_replication(self):
        p = parent_index().reshape(-1)
        self.assertEqual(p.shape, (NODES,))
        counts = np.bincount(p, minlength=OLD_NODES)
        np.testing.assert_array_equal(counts, np.full(OLD_NODES, 16))
        self.assertEqual(int(p[0]), 0)
        self.assertEqual(int(p[3]), 0)
        self.assertEqual(int(p[4]), 1)
        self.assertEqual(int(p[GRID * 4]), 16)
        self.assertEqual(int(p[-1]), 255)

    def test_node_parameter_preserves_parent_values_and_rejects_wrong_source_grid(self):
        values = np.arange(OLD_NODES * 3, dtype=np.float32).reshape(OLD_NODES, 3)
        result = map_node_parameter(values, name="tau_m")
        self.assertEqual(result.shape, (NODES, 3))
        np.testing.assert_array_equal(result, values[parent_index().reshape(-1)])
        with self.assertRaises(ValueError):
            map_node_parameter(np.zeros((1024, 2), dtype=np.float32))

    def test_pair_parameter_uses_same_parent_on_both_axes(self):
        values = np.arange(OLD_NODES * OLD_NODES, dtype=np.int32).reshape(OLD_NODES, OLD_NODES)
        result = map_pair_parameter(values)
        parent = parent_index().reshape(-1)
        self.assertEqual(result.shape, (NODES, NODES))
        np.testing.assert_array_equal(result, values[np.ix_(parent, parent)])
        with self.assertRaises(ValueError):
            map_pair_parameter(np.zeros((OLD_NODES, OLD_NODES - 1), dtype=np.float32))

    def test_modal_native64_direct_2x2_tie_uses_production_smallest_label(self):
        from collaborative_test.SW_0154_native64_resolution_compare.resolution64 import modal_native64
        # Lazy import of the production contract also verifies the actual helper path.
        pixels = np.zeros((320, 128, 128), dtype=np.int16)
        pixels[:, 0, 0] = 3
        pixels[:, 0, 1] = 2
        pixels[:, 1, 0] = 3
        pixels[:, 1, 1] = 2
        pooled = modal_native64(pixels)
        self.assertEqual(pooled.shape, (320, 64, 64))
        self.assertTrue(np.all(pooled[:, 0, 0] == 2))
        # A 2x2 vote is used: the neighboring 2x2 block is untouched background.
        self.assertTrue(np.all(pooled[:, 0, 1] == 0))


if __name__ == "__main__":
    unittest.main()
