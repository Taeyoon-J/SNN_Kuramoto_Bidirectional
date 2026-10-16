import math
from pathlib import Path
import sys
import unittest

import numpy as np  # Must precede Torch on this Windows runtime.
import torch

PATCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PATCH_ROOT))
from snn_kuramoto_bidirectional.graph_generator import ImageConditionedGraph
from snn_kuramoto_bidirectional.kuramoto_layer import graphVectorKuramoto
from collaborative_test.SW_0177_learned_coupling_polarity.polarity_operator import (
    SymmetricEdgeMultiplier, apply_lag_multiplier, install_polarity_prepare,
    seeded_edge_multiplier,
)


class PolarityOperatorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(170176)

    def test_zero_final_layer_is_exact_native_tuple_and_support_floor_is_fixed(self):
        layer = graphVectorKuramoto(4, D=1, K=1.0, device="cpu", kuramoto_backend="factorized")
        structural = torch.tensor([[[0., 1., 0., 0.],
                                    [1., 0., 1., 0.],
                                    [0., 1., 0., 0.],
                                    [0., 0., 0., 0.]]])
        native = layer.prepare_coupling(structural)
        z = torch.randn(1, 4, 16)
        for mode in ("signed", "nonnegative"):
            head = SymmetricEdgeMultiplier(grid_size=2, mode=mode)
            self.assertEqual(sum(isinstance(layer, torch.nn.Linear)
                                for layer in head.layers), 2)
            multiplier = head(z, structural)
            self.assertTrue(torch.equal(multiplier, torch.ones_like(multiplier)))
            changed = apply_lag_multiplier(native, structural, z, head)
            self.assertIs(changed[0], native[0])
            self.assertTrue(torch.equal(changed[1], native[1]))
            self.assertTrue(torch.equal(changed[2], native[2]))

            with torch.no_grad():
                head.layers[-1].bias.fill_(0.8)
            multiplier = head(z, structural)
            self.assertTrue(torch.equal(multiplier, multiplier.transpose(1, 2)))
            self.assertTrue(torch.equal(multiplier.diagonal(dim1=1, dim2=2),
                                         torch.ones(1, 4)))
            self.assertEqual(float(multiplier[0, 0, 2].detach()), 1.0)  # unadmitted floor
            self.assertEqual(float(multiplier[0, 3, 3].detach()), 1.0)
            scaled = apply_lag_multiplier(native, structural, z, head)
            self.assertTrue(torch.equal(scaled[0], native[0]))
            self.assertTrue(torch.allclose(scaled[1], native[1] * multiplier))
            self.assertTrue(torch.allclose(scaled[2], native[2] * multiplier))

    def test_signed_and_abs_magnitude_and_zero_initial_gradient_match(self):
        structural = torch.tensor([[[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]]])
        z_signed = torch.randn(1, 3, 16, requires_grad=True)
        z_control = z_signed.detach().clone().requires_grad_(True)
        signed = SymmetricEdgeMultiplier(grid_size=2, mode="signed")
        control = SymmetricEdgeMultiplier(grid_size=2, mode="nonnegative")
        control.load_state_dict(signed.state_dict())
        weight = torch.randn(1, 3, 3)
        ms = signed(z_signed, structural)
        mc = control(z_control, structural)
        (ms * weight).sum().backward()
        (mc * weight).sum().backward()
        for (name_s, p_s), (name_c, p_c) in zip(signed.named_parameters(),
                                                control.named_parameters()):
            self.assertEqual(name_s, name_c)
            self.assertTrue(torch.equal(p_s.grad, p_c.grad), name_s)
        self.assertTrue(torch.equal(z_signed.grad, z_control.grad))

        with torch.no_grad():
            signed.layers[-1].bias.fill_(1.0)  # m is negative, not merely small
            control.layers[-1].bias.fill_(1.0)
        ms = signed(z_signed.detach(), structural)
        mc = control(z_control.detach(), structural)
        admitted = torch.triu(structural > 0, diagonal=1)
        self.assertTrue((ms[admitted] < 0).all())
        self.assertTrue(torch.allclose(ms.abs()[admitted], mc[admitted]))
        self.assertTrue((mc[admitted] >= 0).all())

    def test_negative_multiplier_survives_native_factorized_four_term_path(self):
        layer = graphVectorKuramoto(3, D=1, K=1.0, alpha_scale=0.7, device="cpu",
                                    kuramoto_backend="factorized")
        with torch.no_grad():
            layer.direction_learner.copy_(torch.tensor([[0.0, 0.9, -0.2],
                                                        [-0.5, 0.0, 0.7],
                                                        [0.3, -0.8, 0.0]]))
            layer.omega.zero_()
            layer.kappa.zero_()
        structural = torch.tensor([[[0., 1., 0.5],
                                    [1., 0., 0.25],
                                    [0.5, 0.25, 0.]]])
        native = layer.prepare_coupling(structural)
        self.assertGreater(float(native[1][0, 0, 1].detach()), 0.0)
        self.assertGreater(float(native[2].abs().max().detach()), 0.0)
        head = SymmetricEdgeMultiplier(grid_size=2, mode="signed")
        with torch.no_grad():
            head.layers[-1].bias.fill_(1.0)
        modified = apply_lag_multiplier(native, structural, torch.randn(1, 3, 16), head)
        self.assertGreater(float(modified[0][0, 0, 1].detach()), 0.0)  # native A_lat stays positive
        self.assertLess(float(modified[1][0, 0, 1].detach()), 0.0)  # signed lag survives native ReLU
        self.assertGreater(float(native[2][0, 1, 2].detach()), 0.0)
        self.assertLess(float(modified[2][0, 1, 2].detach()), 0.0)

        theta = torch.tensor([[[0.0], [math.pi / 2], [math.pi / 3]]])
        gamma = torch.zeros_like(theta)
        base_out = layer(theta, gamma, structural, coupling=native)
        signed_out = layer(theta, gamma, structural, coupling=modified)
        def four_term(coupling):
            _a, lag_cos, lag_sin = coupling
            sin_t, cos_t = torch.sin(theta), torch.cos(theta)
            return (cos_t * torch.bmm(lag_cos, sin_t)
                    - sin_t * torch.bmm(lag_cos, cos_t)
                    - cos_t * torch.bmm(lag_sin, cos_t)
                    - sin_t * torch.bmm(lag_sin, sin_t))

        expected_base = theta + (layer.K / 3.0) * four_term(native)
        expected_signed = theta + (layer.K / 3.0) * four_term(modified)
        self.assertTrue(torch.allclose(base_out, expected_base, atol=1e-7, rtol=1e-7))
        self.assertTrue(torch.allclose(signed_out, expected_signed, atol=1e-7, rtol=1e-7))
        # This isolates the changed signed lag-kernel contribution while
        # retaining all four production factorized terms.
        delta = tuple(modified[k] - native[k] for k in range(3))
        expected_delta = (layer.K / 3.0) * four_term(delta)
        self.assertTrue(torch.allclose(signed_out - base_out, expected_delta,
                                       atol=1e-7, rtol=1e-7))

    def test_actual_projection_and_head_receive_gradient(self):
        graph = ImageConditionedGraph(in_channels=4, hidden_dim=16, top_k=2,
                                      coupling_gain=1.0, learn_gain=False)
        layer = graphVectorKuramoto(4, D=1, K=0.7, device="cpu",
                                    kuramoto_backend="factorized")
        head = SymmetricEdgeMultiplier(grid_size=2, mode="signed")
        with torch.no_grad():
            head.layers[-1].weight.normal_(0.0, 0.01)
            head.layers[-1].bias.fill_(0.2)
        gamma = torch.randn(1, 4, 4)
        theta = torch.randn(1, 4, 1)
        with install_polarity_prepare(graph, layer, head) as captured:
            structural = graph(gamma)
            result = layer(theta, torch.zeros_like(theta), structural)
            (result.square() * torch.tensor([[[1.], [2.], [3.], [5.]]])).sum().backward()
            self.assertIn("z", captured)
            expected_z = torch.nn.functional.normalize(graph.projection(gamma), dim=-1)
            self.assertTrue(torch.equal(captured["z"], expected_z))
        self.assertIsNotNone(graph.projection.weight.grad)
        self.assertGreater(float(graph.projection.weight.grad.norm().detach()), 0.0)
        self.assertGreater(float(head.layers[-1].weight.grad.norm().detach()), 0.0)
        self.assertTrue(torch.isfinite(graph.projection.weight.grad).all())

    def test_instance_wrapper_restores_exactly_on_exception(self):
        graph = ImageConditionedGraph(in_channels=3, hidden_dim=16, top_k=2,
                                      coupling_gain=1.0, learn_gain=False)
        layer = graphVectorKuramoto(4, D=1, device="cpu", kuramoto_backend="factorized")
        head = SymmetricEdgeMultiplier(grid_size=2)
        original_keys = set(layer.__dict__)
        original_projection_hooks = len(graph.projection._forward_hooks)
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            with install_polarity_prepare(graph, layer, head):
                self.assertIn("prepare_coupling", layer.__dict__)
                self.assertEqual(len(graph.projection._forward_hooks),
                                 original_projection_hooks + 1)
                raise RuntimeError("fixture failure")
        self.assertEqual(set(layer.__dict__), original_keys)
        self.assertNotIn("prepare_coupling", layer.__dict__)
        self.assertEqual(len(graph.projection._forward_hooks), original_projection_hooks)

    def test_seeded_head_preserves_cpu_rng_and_repeats_exact_state(self):
        torch.manual_seed(8881)
        before = torch.random.get_rng_state().clone()
        signed = seeded_edge_multiplier(seed=177001, mode="signed")
        after = torch.random.get_rng_state().clone()
        control = seeded_edge_multiplier(seed=177001, mode="nonnegative")
        self.assertTrue(torch.equal(before, after))
        self.assertTrue(torch.equal(signed.state_dict()["layers.0.weight"],
                                    control.state_dict()["layers.0.weight"]))
        self.assertTrue(torch.equal(signed.state_dict()["layers.0.bias"],
                                    control.state_dict()["layers.0.bias"]))
        self.assertEqual(sum(isinstance(module, torch.nn.Linear) for module in signed.layers), 2)


if __name__ == "__main__":
    unittest.main()
