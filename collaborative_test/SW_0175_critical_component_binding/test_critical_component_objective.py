from __future__ import annotations

import unittest

import numpy as np  # Initialize NumPy/OpenMP before torch on Windows.
import torch

from collaborative_test.SW_0168_end_to_end_connectivity import connectivity_objective as old_objective
from collaborative_test.SW_0175_critical_component_binding import critical_component_objective as objective
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity


class CriticalComponentObjectiveTests(unittest.TestCase):
    def test_live_rho_and_q_match_production_component_affinity_exactly(self):
        generator = torch.Generator(device="cpu").manual_seed(175001)
        spikes = torch.randint(0, 2, (2, 4, 19), generator=generator).float()
        component_spikes = torch.randint(0, 2, (2, 4, 4, 19), generator=generator).float()
        rho, q = objective.native_component_rho_q(spikes, component_spikes, settle=5)
        production_q = spike_synchrony_affinity(
            spikes, components=component_spikes, settle=5, affinity_mode="spike")
        self.assertEqual(tuple(rho.shape), (2, 4, 4, 4))
        self.assertTrue(torch.equal(q, production_q))
        self.assertTrue(torch.equal(q, rho.clamp_min(0).prod(dim=1)))

    def test_dead_product_positive_route_has_gradient_to_signed_bottleneck(self):
        # One positive pair. Its first rho component is zero, so native Q is
        # zero and the old product penalty has no path through the active gate.
        correlations = torch.full((4, 2, 2), .5)
        correlations[:, 0, 1] = torch.tensor([0.0, .6, .7, .8])
        correlations[:, 1, 0] = correlations[:, 0, 1]
        rho = correlations.requires_grad_()
        q = rho.detach().clamp_min(0).prod(dim=0)
        support = np.ones((1, 2), dtype=bool)
        target = np.ones((1, 2), dtype=np.int64)
        loss = objective.critical_component_loss(q, rho, target, support)
        grad = torch.autograd.grad(loss, rho)[0]
        self.assertEqual(float(q[0, 1]), 0.0)
        self.assertAlmostEqual(float(loss.detach()), 1.0, places=7)
        self.assertAlmostEqual(float(grad[0, 0, 1]), -2.0, places=7)
        # The frozen forest stores canonical i<j edge orientation only.
        self.assertAlmostEqual(float(grad[0, 1, 0]), 0.0, places=7)
        self.assertTrue(torch.equal(grad[1:, 0, 1], torch.zeros(3)))

    def test_amax_tie_uses_equal_component_subgradient(self):
        rho = torch.full((4, 2, 2), .25, requires_grad=True)
        q = rho.detach().clamp_min(0).prod(dim=0)
        target = np.ones((1, 2), dtype=np.int64)
        support = np.ones((1, 2), dtype=bool)
        loss = objective.critical_component_loss(q, rho, target, support)
        grad = torch.autograd.grad(loss, rho)[0]
        # d(1-rho)^2/d rho = -1.5; four-way amax tie divides credit equally.
        self.assertAlmostEqual(float(loss.detach()), .5625, places=7)
        self.assertTrue(torch.allclose(grad[:, 0, 1], torch.full((4,), -.375)))

    def test_negative_q_squared_loss_and_gradient_are_unchanged(self):
        q1 = torch.tensor([[1.0, .2, .3, .4], [.2, 1.0, .5, .6],
                           [.3, .5, 1.0, .7], [.4, .6, .7, 1.0]], requires_grad=True)
        q2 = q1.detach().clone().requires_grad_()
        rho = torch.full((4, 4, 4), .3, requires_grad=True)
        target = np.asarray([[1, 2], [3, 4]], dtype=np.int64)
        support = np.ones((2, 2), dtype=bool)
        candidate = objective.critical_component_loss(q1, rho, target, support)
        control = old_objective.constrained_maximin_loss(q2, target, support)
        candidate_grad = torch.autograd.grad(candidate, q1)[0]
        control_grad = torch.autograd.grad(control, q2)[0]
        self.assertEqual(float(candidate.detach()), float(control.detach()))
        self.assertTrue(torch.equal(candidate_grad, control_grad))
        self.assertIsNone(torch.autograd.grad(candidate, rho, allow_unused=True)[0])

    def test_pair_mass_routes_are_identical_and_weight_bottleneck_edges(self):
        q = torch.full((4, 4), .25)
        q.fill_diagonal_(1.0)
        rho = torch.full((4, 4, 4), .2, requires_grad=True)
        target = np.ones((2, 2), dtype=np.int64)
        support = np.ones((2, 2), dtype=bool)
        plan = old_objective.forest_routes(q, target, support)
        self.assertEqual(plan["positive_pair_mass"], 6)
        self.assertEqual(plan["positive_edges"], old_objective.forest_routes(
            q, target, support)["positive_edges"])
        loss = objective.critical_component_loss(q, rho, target, support, plan=plan)
        self.assertAlmostEqual(float(loss.detach()), .64, places=7)

    def test_malformed_component_geometry_is_rejected(self):
        spikes = torch.zeros(1, 4, 10)
        with self.assertRaises(ValueError):
            objective.native_component_rho_q(spikes, torch.zeros(1, 3, 4, 10), settle=2)
        with self.assertRaises(ValueError):
            objective.native_component_rho_q(spikes, torch.zeros(1, 4, 4, 10), settle=10)
        with self.assertRaises(ValueError):
            objective.critical_component_loss(torch.eye(4), torch.zeros(3, 4, 4),
                                              np.ones((2, 2), dtype=np.int64),
                                              np.ones((2, 2), dtype=bool))


if __name__ == "__main__":
    unittest.main()

