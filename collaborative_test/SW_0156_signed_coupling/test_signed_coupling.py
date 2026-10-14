import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from collaborative_test.SW_0156_signed_coupling.signed_coupling import (
    topk_signed_perturbation,
    install_instance_signed_interaction,
    native_logits_and_signed_u,
    prepare_signed_coupling,
)
from snn_kuramoto_bidirectional.graph_generator import ImageConditionedGraph
from snn_kuramoto_bidirectional.kuramoto_layer import graphVectorKuramoto


class NativeLag(nn.Module):
    def __init__(self):
        super().__init__()
        self.direction_learner = nn.Parameter(torch.tensor([
            [0.0, 0.8, -0.2], [-0.3, 0.0, 0.4], [0.7, -0.5, 0.0]
        ]))
        self.alpha_scale = 0.7

    def prepare_coupling(self, a, batch_size=None, num_units=None, device=None):
        a0 = 0.5 * (a + a.transpose(1, 2))
        a0 = torch.relu(a0) + 1e-6
        cost = a0.reciprocal()
        lo = cost.amin(dim=(1, 2), keepdim=True)
        hi = cost.amax(dim=(1, 2), keepdim=True)
        norm = (cost - lo) / (hi - lo + 1e-6)
        d = self.direction_learner - self.direction_learner.T
        alpha = self.alpha_scale * torch.tanh(d)[None] * norm
        return a0, a0 * alpha.cos(), a0 * alpha.sin()


class SignedCouplingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(156)
        self.layer = NativeLag()
        self.a = torch.tensor([[[0.1, 0.8, 0.1], [0.2, 0.1, 0.7],
                                [0.7, 0.4, 0.1]]], dtype=torch.float32)
        self.u = torch.tensor([[[0.0, 2.0, 0.0], [2.0, 0.0, 2.0],
                                [0.0, 2.0, 0.0]]], dtype=torch.float32)

    def test_beta_zero_returns_original_tuple_bitwise(self):
        expected = self.layer.prepare_coupling(self.a)
        actual = prepare_signed_coupling(self.layer, self.a, self.u, 0.0)
        self.assertEqual(len(actual), 3)
        for x, y in zip(expected, actual):
            self.assertTrue(torch.equal(x, y))

    def test_signed_edge_can_be_negative_and_pulse_graph_stays_positive(self):
        a0, lagcos, lagsin = prepare_signed_coupling(self.layer, self.a, self.u, 0.5)
        signed = a0 - 0.5 * self.u
        self.assertTrue(torch.equal(a0, self.layer.prepare_coupling(self.a)[0]))
        self.assertLess(float(signed[0, 0, 1]), 0.0)
        self.assertTrue((a0 > 0).all())
        c0 = self.layer.prepare_coupling(self.a)
        cosalpha, sinalpha = c0[1] / c0[0], c0[2] / c0[0]
        self.assertTrue(torch.allclose(lagcos, signed * cosalpha))
        self.assertTrue(torch.allclose(lagsin, signed * sinalpha))

    def test_directed_topk_scatter_gain_over_k_then_symmetry(self):
        logits = torch.tensor([[[3., 2., 1.], [1., 4., 3.], [4., 1., 2.]]])
        got = topk_signed_perturbation(logits, top_k=1, gain=2.0)
        top_idx = logits.topk(1, -1).indices
        top_values = torch.full_like(logits.topk(1, -1).values, 2.0)
        directed = torch.zeros_like(logits).scatter_(-1, top_idx, top_values)
        want = 0.5 * (directed + directed.transpose(-1, -2))
        self.assertTrue(torch.equal(got, want))
        self.assertTrue(torch.equal(got, got.transpose(-1, -2)))

    def test_control_and_signed_candidate_keep_same_native_alpha(self):
        base = self.layer.prepare_coupling(self.a)
        self.a[0, 0, 2] = 0.0
        self.a[0, 2, 0] = 0.0
        base = self.layer.prepare_coupling(self.a)
        control = prepare_signed_coupling(self.layer, self.a, torch.zeros_like(self.u), 0.5,
                                          base_scale=0.5)
        self.assertTrue(torch.equal(base[0], control[0]))
        alpha_cos, alpha_sin = base[1] / base[0], base[2] / base[0]
        control_a = 0.5 * base[0] + 0.5e-6
        self.assertTrue(torch.allclose(control[1], control_a * alpha_cos))
        self.assertTrue(torch.allclose(control[2], control_a * alpha_sin))
        self.assertAlmostEqual(float(torch.sqrt(control[1][0, 0, 2] ** 2
                                               + control[2][0, 0, 2] ** 2).detach()), 1e-6, places=10)
        candidate = prepare_signed_coupling(self.layer, self.a, self.u, 0.5)
        self.assertFalse(torch.equal(control[1], candidate[1]))

    def test_invalid_u_or_logits_fail_closed(self):
        with self.assertRaises(ValueError):
            prepare_signed_coupling(self.layer, self.a, torch.full_like(self.u, float('nan')), .5)
        with self.assertRaises(ValueError):
            topk_signed_perturbation(torch.ones(1, 3, 4), 1, 2.)

    def test_actual_graph_logits_and_learned_gain_define_u(self):
        torch.manual_seed(31)
        graph = ImageConditionedGraph(3, hidden_dim=4, top_k=2, coupling_gain=3.0,
                                     temperature=.2, grid_size=2, spatial_decay=.7,
                                     geodesic_steps=1, geodesic_contrast=8.0)
        gamma = torch.randn(1, 3, 4)
        native = graph(gamma)
        logits, u = native_logits_and_signed_u(graph, gamma, native)
        indices = logits.topk(2, -1).indices
        values = torch.ones_like(indices, dtype=logits.dtype) * graph.log_coupling_gain.exp() / 2
        directed = torch.zeros_like(logits).scatter_(-1, indices, values)
        self.assertTrue(torch.equal(u, .5 * (directed + directed.transpose(-1, -2))))
        with self.assertRaises(ValueError):
            native_logits_and_signed_u(graph, gamma, native + .01)

    def test_instance_install_matches_native_zero_and_restores(self):
        torch.manual_seed(33)
        graph = ImageConditionedGraph(3, hidden_dim=4, top_k=2, coupling_gain=3.0,
                                     temperature=.2, grid_size=2, spatial_decay=.7,
                                     geodesic_steps=1, geodesic_contrast=8.0)
        layer = graphVectorKuramoto(4, D=2, kuramoto_backend="factorized")
        gamma = torch.randn(1, 3, 4)
        expected_graph = graph(gamma)
        expected_coupling = layer.prepare_coupling(expected_graph)
        restore = install_instance_signed_interaction(graph, layer, beta=0.0)
        try:
            actual_graph = graph(gamma)
            actual_coupling = layer.prepare_coupling(actual_graph)
            self.assertTrue(torch.equal(expected_graph, actual_graph))
            for left, right in zip(expected_coupling, actual_coupling):
                self.assertTrue(torch.equal(left, right))
        finally:
            restore()
        self.assertIs(graph.forward.__func__, ImageConditionedGraph.forward)

    def test_actual_native_kuramoto_preparation_keeps_signed_lag_and_floor(self):
        layer = graphVectorKuramoto(3, D=2, kuramoto_backend="factorized")
        with torch.no_grad():
            layer.direction_learner.zero_()
        adjacency = torch.tensor([[[0., .2, 0.], [.2, 0., .4], [0., .4, 0.]]])
        u = torch.tensor([[[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]]])
        native = layer.prepare_coupling(adjacency)
        candidate = prepare_signed_coupling(layer, adjacency, u, beta=.5)
        control = prepare_signed_coupling(layer, adjacency, torch.zeros_like(u), beta=0.,
                                          base_scale=.5)
        self.assertLess(float(candidate[1][0, 0, 1].detach()), 0.0)
        self.assertEqual(float(candidate[2].abs().max().detach()), 0.0)
        self.assertTrue(torch.equal(native[0], candidate[0]))
        # The diagonal/off-support floor is kept at 1e-6 in the half graph.
        self.assertAlmostEqual(float(control[1][0, 0, 0].detach()), 1e-6, places=10)

    def test_signed_interaction_can_repel_pair(self):
        # For a two-node phase difference, positive A attracts under zero lag;
        # replacing its lag multiplier by a negative value reverses the sign.
        theta = torch.tensor([[[0.0], [0.4]]])
        a_beta = torch.tensor([[[0.0, -0.5], [-0.5, 0.0]]])
        alpha = torch.zeros_like(a_beta)
        delta = theta[:, None, :, 0] - theta[:, :, None, 0] - alpha
        term = (a_beta * delta.sin()).sum(-1)
        self.assertLess(float(term[0, 0]), 0.0)
        self.assertGreater(float(term[0, 1]), 0.0)

    def test_factorized_signed_lags_match_pairwise_alpha_formula(self):
        a0, lagcos, lagsin = prepare_signed_coupling(self.layer, self.a, self.u, .5)
        theta = torch.tensor([[[.1, .7], [.4, -.2], [-.5, .3]]])
        sin_t, cos_t = theta.sin(), theta.cos()
        factored = (cos_t * torch.bmm(lagcos, sin_t)
                    - sin_t * torch.bmm(lagcos, cos_t)
                    - cos_t * torch.bmm(lagsin, cos_t)
                    - sin_t * torch.bmm(lagsin, sin_t))
        signed_a = a0 - .5 * self.u
        native = self.layer.prepare_coupling(self.a)
        alpha = torch.atan2(native[2] / native[0], native[1] / native[0])
        pairwise = (signed_a.unsqueeze(-1)
                    * torch.sin(theta.unsqueeze(1) - theta.unsqueeze(2) - alpha.unsqueeze(-1))).sum(2)
        self.assertTrue(torch.allclose(factored, pairwise, atol=2e-6, rtol=2e-6))


if __name__ == '__main__':
    unittest.main()




