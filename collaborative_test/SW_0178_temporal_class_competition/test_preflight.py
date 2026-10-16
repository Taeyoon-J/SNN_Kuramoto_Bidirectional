import numpy as np  # Keep NumPy/OpenMP initialization before torch on Windows.
import torch
import unittest

from collaborative_test.SW_0178_temporal_class_competition.preflight import (
    flatten_modal_targets, settled_cache_rho_q,
)
from collaborative_test.SW_0178_temporal_class_competition.class_competition import native_component_rho_q


class TemporalCacheBoundaryTests(unittest.TestCase):
    def test_settled_cache_matches_full_rollout_slice_without_reslicing(self):
        generator = torch.Generator().manual_seed(17801)
        settled = torch.randn((2, 4, 1024, 512), generator=generator, dtype=torch.float32)
        rho_cache, q_cache = settled_cache_rho_q(settled)
        prefix = torch.randn((2, 4, 1024, 512), generator=generator, dtype=torch.float32)
        full = torch.cat((prefix, settled), dim=-1)
        rho_full, q_full = native_component_rho_q(full.mean(dim=1), full, settle=512)
        self.assertTrue(torch.equal(rho_cache, rho_full))
        self.assertTrue(torch.equal(q_cache, q_full))
        with self.assertRaisesRegex(ValueError, "settled cache rows"):
            settled_cache_rho_q(full)

    def test_modal_grid_flatten_is_explicit_and_preserves_full_ids(self):
        grid = np.zeros((2, 32, 32), dtype=np.int32)
        grid[0, 1, 3] = 9
        grid[1, 31, 31] = 10
        flattened = flatten_modal_targets(grid)
        self.assertEqual(tuple(flattened.shape), (2, 1024))
        self.assertEqual(int(flattened[0, 35]), 9)
        self.assertEqual(int(flattened[1, 1023]), 10)
        with self.assertRaisesRegex(ValueError, r"\[B,32,32\]"):
            flatten_modal_targets(np.zeros((1, 1024), dtype=np.int64))
        with self.assertRaisesRegex(TypeError, "integer dtype"):
            flatten_modal_targets(np.zeros((1, 32, 32), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
