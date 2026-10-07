"""Small synthetic checks for the fixed triangle-pruning rule."""
import unittest
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from .diagnose import groups_from_adj, triangle_support


class TriangleSupportTests(unittest.TestCase):
    def test_exact_size_two_raw_component_edge_is_restored(self):
        q = np.zeros((5, 5), dtype=np.float32)
        q[0, 1] = q[1, 0] = .7
        for u, v in ((2, 3), (3, 4), (2, 4)):
            q[u, v] = q[v, u] = .7
        raw, pruned, restored = triangle_support(torch.from_numpy(q))
        self.assertTrue(raw[0, 1])
        self.assertTrue(pruned[0, 1])
        self.assertEqual(restored, [(0, 1)])
        self.assertEqual(groups_from_adj(pruned), [[0, 1]])

    def test_one_common_neighbor_retains_all_triangle_edges(self):
        q = np.zeros((4, 4), dtype=np.float32)
        for u, v in ((0, 1), (1, 2), (0, 2)):
            q[u, v] = q[v, u] = .7
        _, pruned, restored = triangle_support(torch.from_numpy(q))
        self.assertEqual(int(np.triu(pruned, 1).sum()), 3)
        self.assertEqual(restored, [])

    def test_path_component_larger_than_two_is_pruned_without_restoration(self):
        q = np.zeros((3, 3), dtype=np.float32)
        q[0, 1] = q[1, 0] = .7
        q[1, 2] = q[2, 1] = .7
        raw, pruned, restored = triangle_support(torch.from_numpy(q))
        count, components = connected_components(csr_matrix(raw), directed=False)
        self.assertEqual(int(np.bincount(components, minlength=count).max()), 3)
        self.assertEqual(int(np.triu(pruned, 1).sum()), 0)
        self.assertEqual(restored, [])

    def test_common_neighbor_count_uses_int32_not_uint8(self):
        n = 260
        q = np.zeros((n, n), dtype=np.float32)
        q[0, 1] = q[1, 0] = .7
        for k in range(2, 258):  # 256 common neighbors of nodes zero and one
            q[0, k] = q[k, 0] = .7
            q[1, k] = q[k, 1] = .7
        raw, pruned, _ = triangle_support(torch.from_numpy(q))
        self.assertTrue(raw[0, 1])
        self.assertTrue(pruned[0, 1])


if __name__ == "__main__":
    unittest.main()
