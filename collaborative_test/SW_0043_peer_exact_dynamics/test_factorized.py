"""Numerical and compatibility tests for peer-exact factorized dynamics."""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "snn_kuramoto_bidirectional"))

from snn_kuramoto_bidirectional.hyperparameter import S2NetHyperparameters
from snn_kuramoto_bidirectional.kuramoto_layer import graphVectorKuramoto
from snn_kuramoto_bidirectional.s2net_cls import S2NetCore


class FactorizedKuramotoTest(unittest.TestCase):
    def test_default_pairwise_keys_rng_and_explicit_mode_are_unchanged(self):
        torch.manual_seed(800)
        default = graphVectorKuramoto(7, D=3, K=1.2, freq_gain=0.4)
        expected_next = torch.rand(4)
        torch.manual_seed(800)
        explicit = graphVectorKuramoto(7, D=3, K=1.2, freq_gain=0.4,
                                       kuramoto_backend="pairwise")
        actual_next = torch.rand(4)
        self.assertEqual(set(default.state_dict()), set(explicit.state_dict()))
        for key in default.state_dict():
            torch.testing.assert_close(default.state_dict()[key], explicit.state_dict()[key],
                                       rtol=0, atol=0)
        torch.testing.assert_close(expected_next, actual_next, rtol=0, atol=0)

    def test_pairwise_factorized_forward_and_gradients_close(self):
        torch.manual_seed(123)
        pairwise = graphVectorKuramoto(6, D=3, K=1.7, dt=0.13,
                                       freq_gain=0.2, kuramoto_backend="pairwise")
        factorized = graphVectorKuramoto(6, D=3, K=1.7, dt=0.13,
                                         freq_gain=0.2, kuramoto_backend="factorized")
        factorized.load_state_dict(pairwise.state_dict(), strict=True)
        theta0 = torch.randn(2, 6, 3)
        gamma0 = torch.randn(2, 6, 1)
        graph0 = torch.rand(2, 6, 6) + 0.1

        def run(layer):
            theta = theta0.clone().requires_grad_()
            gamma = gamma0.clone().requires_grad_()
            graph = graph0.clone().requires_grad_()
            output = layer(theta, gamma, A=graph)
            loss = output.square().mean()
            inputs = [theta, gamma, graph, *layer.parameters()]
            grads = torch.autograd.grad(loss, inputs, allow_unused=True)
            return output, grads

        expected, expected_grads = run(pairwise)
        actual, actual_grads = run(factorized)
        torch.testing.assert_close(expected, actual, rtol=2e-5, atol=2e-6)
        self.assertEqual(len(expected_grads), len(actual_grads))
        for left, right in zip(expected_grads, actual_grads):
            self.assertIsNotNone(left)
            self.assertIsNotNone(right)
            self.assertTrue(torch.isfinite(left).all())
            self.assertTrue(torch.isfinite(right).all())
            torch.testing.assert_close(left, right, rtol=5e-4, atol=2e-6)

    def test_fixed_graph_kernels_can_be_reused_across_steps(self):
        torch.manual_seed(41)
        layer = graphVectorKuramoto(5, D=2, kuramoto_backend="factorized")
        graph = torch.rand(2, 5, 5)
        theta = torch.randn(2, 5, 2)
        gamma = torch.randn(2, 5)
        coupling = layer.prepare_coupling(graph)
        direct = layer(theta, gamma, A=graph)
        reused = layer(theta, gamma, A=graph, coupling=coupling)
        torch.testing.assert_close(direct, reused, rtol=0, atol=0)

    def test_full_core_rollout_same_checkpoint_pairwise_vs_factorized(self):
        def core(backend):
            hp = S2NetHyperparameters(
                num_feature_maps=2, num_regions=8, sc=torch.eye(8), osc_dim=2,
                gamma_drive_mode="static", gamma_phase_mode="standardize_tanh",
                graph_mode="static", num_time_steps=5, spike_per_component=True,
                kuramoto_backend=backend,
            ).validate()
            return S2NetCore(hp, device="cpu")

        torch.manual_seed(98)
        pairwise = core("pairwise")
        factorized = core("factorized")
        self.assertEqual(set(pairwise.state_dict()), set(factorized.state_dict()))
        factorized.load_state_dict(pairwise.state_dict(), strict=True)
        gamma = torch.randn(2, 8)
        _, spikes_a, membrane_a, theta_a = pairwise(
            gamma, return_core_out=True, return_theta=True)
        _, spikes_b, membrane_b, theta_b = factorized(
            gamma, return_core_out=True, return_theta=True)
        torch.testing.assert_close(theta_a, theta_b, rtol=2e-5, atol=3e-6)
        torch.testing.assert_close(membrane_a, membrane_b, rtol=2e-5, atol=3e-6)
        torch.testing.assert_close(spikes_a, spikes_b, rtol=1e-6, atol=2e-7)

    def test_invalid_backend_rejected(self):
        with self.assertRaisesRegex(ValueError, "kuramoto_backend"):
            S2NetHyperparameters(kuramoto_backend="other").validate()


if __name__ == "__main__":
    unittest.main()
