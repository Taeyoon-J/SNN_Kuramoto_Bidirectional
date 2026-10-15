"""Independent tiny-grid checks for the SW0168 maximin and local objectives."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from collaborative_test.SW_0168_end_to_end_connectivity import connectivity_objective as objective


class ConnectivityObjectiveTests(unittest.TestCase):
    def test_pair_count_weighted_maximin_matches_hand_computation(self):
        q = torch.tensor([[0., .2, .1, .1], [.2, 0., .8, .1],
                          [.1, .8, 0., .9], [.1, .1, .9, 0.]], requires_grad=True)
        target = np.array([[1, 1, 1, 2]], dtype=np.int32)
        support = np.ones((1, 4), dtype=bool)
        plan = objective.forest_routes(q, target, support)
        self.assertEqual(plan["positive_edges"], [(1, 2, 1), (0, 1, 2)])
        self.assertEqual(plan["positive_pair_mass"], 3)
        self.assertEqual(plan["negative_edges"], [(2, 3, 3)])
        self.assertEqual(plan["negative_pair_mass"], 3)
        loss = objective.constrained_maximin_loss(q, target, support)
        self.assertAlmostEqual(float(loss.detach()), .625, places=7)
        loss.backward()
        self.assertAlmostEqual(float(q.grad[0, 1]), -2 * .8 * 2 / 3 / 2, places=7)
        self.assertAlmostEqual(float(q.grad[1, 2]), -2 * .2 / 3 / 2, places=7)
        self.assertAlmostEqual(float(q.grad[2, 3]), 2 * .9 / 2, places=7)

    def test_local_control_is_uniform_class_mean_and_gradient(self):
        q = torch.tensor([[0., .2, .1, .1], [.2, 0., .8, .1],
                          [.1, .8, 0., .9], [.1, .1, .9, 0.]], requires_grad=True)
        target = np.array([[1, 1, 1, 2]], dtype=np.int64)
        support = np.ones((1, 4), dtype=bool)
        loss = objective.matched_local_edge_loss(q, target, support)
        self.assertAlmostEqual(float(loss.detach()), (.34 + .81) / 2, places=7)
        loss.backward()
        self.assertAlmostEqual(float(q.grad[0, 1]), -2 * .8 / 2 / 2, places=7)
        self.assertAlmostEqual(float(q.grad[1, 2]), -2 * .2 / 2 / 2, places=7)
        self.assertAlmostEqual(float(q.grad[2, 3]), 2 * .9 / 2, places=7)

    def test_tie_order_is_canonical_and_background_class_is_supervised(self):
        q = torch.tensor([[0., .5, .1], [.5, 0., .5], [.1, .5, 0.]])
        # Target ID zero is a real class on supported patches; support, not the
        # target number, defines membership. Equal edge scores use patch order.
        target = np.array([[0, 0, 1]], dtype=np.int32)
        support = np.ones((1, 3), dtype=bool)
        plan = objective.forest_routes(q, target, support)
        self.assertEqual(plan["positive_edges"], [(0, 1, 1)])
        self.assertEqual(plan["negative_edges"], [(1, 2, 2)])
        self.assertEqual(plan["positive_pair_mass"], 1)
        self.assertEqual(plan["negative_pair_mass"], 2)

    def test_disconnected_fixed_support_does_not_create_routes(self):
        q = torch.full((4, 4), .5)
        q.fill_diagonal_(0)
        target = np.array([[7, 7, 0, 8]], dtype=np.int32)
        support = np.array([[1, 1, 0, 1]], dtype=bool)
        plan = objective.forest_routes(q, target, support)
        self.assertEqual(plan["positive_edges"], [(0, 1, 1)])
        self.assertEqual(plan["positive_pair_mass"], 1)
        self.assertEqual(plan["negative_pair_mass"], 0)
        self.assertEqual(plan["excluded_unreachable_negative_pair_mass"], 2)

    def test_infeasible_target_reused_across_disconnected_support_is_rejected(self):
        q = torch.full((4, 4), .5)
        q.fill_diagonal_(0)
        target = np.array([[7, 8, 0, 7]], dtype=np.int32)
        support = np.array([[1, 1, 0, 1]], dtype=bool)
        with self.assertRaisesRegex(ValueError, "4-connected"):
            objective.constrained_maximin_loss(q, target, support)

    def test_infeasible_disconnected_target_inside_support_component_is_rejected(self):
        q = torch.full((3, 3), .5)
        q.fill_diagonal_(0)
        target = np.array([[1, 2, 1]], dtype=np.int32)
        support = np.ones((1, 3), dtype=bool)
        with self.assertRaisesRegex(ValueError, "4-connected"):
            objective.constrained_maximin_loss(q, target, support)

    def test_empty_support_has_live_zero_and_bad_geometry_rejects(self):
        q = torch.zeros((4, 4), requires_grad=True)
        loss = objective.constrained_maximin_loss(q, np.zeros((2, 2), np.int32),
                                                  np.zeros((2, 2), bool))
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        with self.assertRaises(ValueError):
            objective.matched_local_edge_loss(q, np.zeros((1, 4), np.int32),
                                              np.zeros((2, 2), bool))
        with self.assertRaises(ValueError):
            objective.constrained_maximin_loss(torch.full((4, 4), 1.1),
                np.zeros((2, 2), np.int32), np.zeros((2, 2), bool))


if __name__ == "__main__":
    unittest.main()
