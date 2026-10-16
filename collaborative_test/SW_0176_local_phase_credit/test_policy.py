"""CPU checks for SW0176 local-phase gradient routing."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from collaborative_test.SW_0176_local_phase_credit.policy import detach_previous_theta
from collaborative_test.SW_0134_native_spike_binding.rollout import sinusoidal_gating
from snn_kuramoto_bidirectional.kuramoto_layer import graphVectorKuramoto


class LocalPhasePolicyTests(unittest.TestCase):
    def _model(self):
        torch.manual_seed(17)
        return graphVectorKuramoto(N=4, D=2, K=0.3, device="cpu", freq_gain=0.2,
                                  kuramoto_backend="pairwise")

    def test_forward_values_are_exact_and_theta_history_output_remains_live(self):
        model = self._model()
        theta0 = torch.tensor([[[0.2, 0.6], [0.7, 0.1], [1.0, 0.3], [0.4, 0.9]]],
                              dtype=torch.float32, requires_grad=True)
        gamma0 = torch.full((1, 4), 0.35)
        gamma1 = torch.full((1, 4), 0.8)
        adjacency = torch.tensor([[[0.0, .2, .4, .1], [.2, 0.0, .3, .5],
                                   [.4, .3, 0.0, .6], [.1, .5, .6, 0.0]]])
        ref1 = model(theta0, gamma0, A=adjacency)
        ref2 = model(ref1, gamma1, A=adjacency)
        with detach_previous_theta(model) as observed:
            local1 = model(theta0, gamma0, A=adjacency)
            local2 = model(local1, gamma1, A=adjacency)
            carrier, gate = sinusoidal_gating([local1, local2], 1, 1, gate_mode="sigmoid")
            self.assertEqual(observed["count"], 2)
        torch.testing.assert_close(local1, ref1, rtol=0, atol=0)
        torch.testing.assert_close(local2, ref2, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(carrier).all())
        self.assertTrue(torch.isfinite(gate).all())
        self.assertTrue(local1.requires_grad, "returned theta output must remain live for delayed reads")
        self.assertEqual(len(model._forward_pre_hooks), 0)

    def test_successive_input_jacobian_stops_but_local_and_delayed_read_credit_remain(self):
        model = self._model()
        theta0 = torch.tensor([[[0.2, 0.6], [0.7, 0.1], [1.0, 0.3], [0.4, 0.9]]],
                              dtype=torch.float32, requires_grad=True)
        gamma0 = torch.full((1, 4), 0.35)
        gamma1 = torch.full((1, 4), 0.8)
        adjacency = torch.ones((1, 4, 4)) - torch.eye(4).unsqueeze(0)
        with detach_previous_theta(model):
            theta1 = model(theta0, gamma0, A=adjacency)
            theta2 = model(theta1, gamma1, A=adjacency)
            successive = torch.autograd.grad(theta2.sum(), theta1, retain_graph=True,
                                             allow_unused=True)[0]
            # The delayed carrier/gate reads the still-live first output.
            carrier, gate = sinusoidal_gating([theta1, theta2], 1, 1, gate_mode="sigmoid")
            delayed = torch.autograd.grad((carrier.sum() + gate.sum()), theta1,
                                          retain_graph=True, allow_unused=True)[0]
            param_grad = torch.autograd.grad((theta2.square().sum() + carrier.sum()),
                                             model.omega, allow_unused=True)[0]
        self.assertTrue(successive is None or torch.count_nonzero(successive).item() == 0)
        self.assertIsNotNone(delayed)
        self.assertGreater(float(delayed.abs().sum().detach()), 0.0)
        self.assertIsNotNone(param_grad)
        self.assertGreater(float(param_grad.abs().sum().detach()), 0.0)
        self.assertEqual(len(model._forward_pre_hooks), 0)

    def test_keyword_theta_and_other_inputs_are_preserved(self):
        model = self._model()
        theta = torch.ones((1, 4, 2), requires_grad=True)
        gamma = torch.zeros((1, 4))
        adjacency = torch.eye(4).unsqueeze(0)
        expected = model(theta, gamma, A=adjacency)
        with detach_previous_theta(model) as observed:
            actual = model(theta_prev=theta, gamma=gamma, A=adjacency, spike=None, coupling=None)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(observed["count"], 1)
        self.assertEqual(len(model._forward_pre_hooks), 0)

    def test_hook_is_removed_even_if_forward_raises(self):
        model = self._model()
        theta = torch.ones((1, 4, 2), requires_grad=True)
        with self.assertRaises(ValueError):
            with detach_previous_theta(model):
                model(theta, torch.zeros((1, 3)))
        self.assertEqual(len(model._forward_pre_hooks), 0)
        # The original module path remains callable after the exceptional scope.
        value = model(torch.ones((1, 4, 2)), torch.zeros((1, 4)))
        self.assertEqual(tuple(value.shape), (1, 4, 2))


if __name__ == "__main__":
    unittest.main()
