"""Synthetic-only checks for SW0102 maximin forest and active-gradient guards."""

import unittest

import torch

from .forest_loss import (cannot_link_hinge, maximin_forest_edge_map,
                          same_component_pairs)


def brute_widest(q, threshold=0.40):
    """All-pairs widest paths via Floyd-Warshall over max/min semiring."""
    n = q.size(0)
    best = torch.full((n, n), float("-inf"), dtype=q.dtype)
    for i in range(n):
        best[i, i] = float("inf")
    for i in range(n):
        for j in range(n):
            if i != j and q[i, j] > threshold:
                best[i, j] = q[i, j]
    for k in range(n):
        best = torch.maximum(best, torch.minimum(best[:, k:k+1], best[k:k+1, :]))
    return best


class ForestDraftTests(unittest.TestCase):
    def test_forest_map_matches_all_widest_path_values(self):
        gen = torch.Generator().manual_seed(812)
        raw = torch.rand((12, 12), generator=gen, dtype=torch.float64)
        q = (raw + raw.T) * 0.5
        q.fill_diagonal_(1.0)
        index_map = maximin_forest_edge_map(q[None])[0]
        widest = brute_widest(q)
        for i in range(12):
            for j in range(12):
                if i == j or widest[i, j] <= 0.40:
                    self.assertEqual(int(index_map[i, j]), -1)
                    continue
                flat = int(index_map[i, j])
                self.assertGreaterEqual(flat, 0)
                u, v = divmod(flat, 12)
                self.assertAlmostEqual(float(q[u, v]), float(widest[i, j]), places=12)

    def test_indirect_counterexample_gathers_only_selected_live_bottleneck(self):
        # q01=.1, q02=.9, q21=.8: negative pair 01 has c=.8 by path 0-2-1.
        q = torch.tensor([[[1., .1, .9], [.1, 1., .8], [.9, .8, 1.]]],
                         dtype=torch.float64, requires_grad=True)
        negative = torch.zeros_like(q, dtype=torch.bool)
        negative[0, 0, 1] = True
        loss, counts = cannot_link_hinge(q, [negative])
        self.assertAlmostEqual(float(loss.detach()), (.8 - .4) ** 2, places=12)
        self.assertEqual(counts["connected_directed_negatives"], 1)
        loss.backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertGreater(float(q.grad[0, 1, 2].abs()), 0.0)
        self.assertGreater(float(q.grad[0, 2, 1].abs()), 0.0)
        self.assertEqual(float(q.grad[0, 0, 2]), 0.0)

    def test_equal_weight_order_is_lexicographic_and_deterministic(self):
        q = torch.eye(4, dtype=torch.float64)
        for i, j in ((0, 1), (0, 2), (1, 3), (2, 3)):
            q[i, j] = q[j, i] = .8
        first = maximin_forest_edge_map(q[None])
        second = maximin_forest_edge_map(q[None])
        self.assertTrue(torch.equal(first, second))
        # Lex order accepts (0,1), then (0,2), which first joins pair (1,2).
        self.assertEqual(int(first[0, 1, 2]), 0 * 4 + 2)

    def test_disconnected_pairs_are_zero_and_classifier_half_cc_agrees(self):
        q = torch.tensor([[[1., .50, .2, .0], [.50, 1., .1, .0],
                           [.2, .1, 1., .50], [.0, .0, .50, 1.]]],
                         dtype=torch.float64, requires_grad=True)
        all_pairs = torch.ones_like(q, dtype=torch.bool)
        same_half = same_component_pairs(q, all_pairs, threshold=.50)
        # Components at >=.50 are {0,1} and {2,3}; map edges at >.40 have
        # exactly the same component partition.
        edge_map = maximin_forest_edge_map(q.detach(), edge_threshold=.40)
        forest_cc = edge_map >= 0
        expected = torch.tensor([[[False, True, False, False],
                                  [True, False, False, False],
                                  [False, False, False, True],
                                  [False, False, True, False]]])
        self.assertTrue(torch.equal(same_half.cpu(), expected))
        self.assertTrue(torch.equal(forest_cc, expected))

        negative = torch.zeros_like(q, dtype=torch.bool)
        negative[0, 0, 3] = True  # disconnected under the >.40 graph
        loss, counts = cannot_link_hinge(q, [negative])
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(counts["connected_directed_negatives"], 0)

    def test_symmetry_assertion_and_empty_batch_zero(self):
        asymmetric = torch.tensor([[[1., .7], [.6, 1.]]])
        with self.assertRaises(ValueError):
            maximin_forest_edge_map(asymmetric)
        q = torch.eye(3).unsqueeze(0).requires_grad_()
        empty = torch.zeros_like(q, dtype=torch.bool)
        loss, stats = cannot_link_hinge(q, [empty])
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(stats["selected_directed_negatives"], 0)
        loss.backward()
        self.assertTrue(torch.isfinite(q.grad).all())


if __name__ == "__main__":
    unittest.main()
