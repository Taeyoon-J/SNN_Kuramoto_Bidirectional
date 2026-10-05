import unittest

import torch
from torch import nn

from snn_kuramoto_bidirectional.hyperparameter import S2NetHyperparameters
from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
from rgb_graph import blend_adjacencies, rgb_bilateral_graph


class FakeGraph(nn.Module):
    def __init__(self, top_k=16, gain=6.0):
        super().__init__()
        self.top_k = top_k
        self.register_buffer("log_coupling_gain", torch.tensor(gain).log())


class RGBGraphTests(unittest.TestCase):
    def test_rgb_graph_shape_symmetry_nonnegative_and_gain(self):
        images = torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8)
        graph = rgb_bilateral_graph(images, FakeGraph(), grid_size=4)
        self.assertEqual(tuple(graph.shape), (2, 16, 16))
        self.assertTrue(torch.isfinite(graph).all())
        self.assertTrue((graph >= 0).all())
        self.assertTrue(torch.allclose(graph, graph.transpose(1, 2), atol=1e-7))
        self.assertTrue(torch.allclose(
            graph.sum(dim=-1).mean(dim=-1), torch.full((2,), 6.0), atol=1e-5
        ))

    def test_alpha_zero_is_exact_and_blends_are_valid(self):
        learned = torch.rand(2, 16, 16)
        rgb = torch.rand(2, 16, 16)
        self.assertIs(blend_adjacencies(learned, rgb, 0.0), learned)
        for alpha in (0.25, 0.5, 1.0):
            result = blend_adjacencies(learned, rgb, alpha)
            self.assertEqual(tuple(result.shape), tuple(learned.shape))
            self.assertTrue(torch.isfinite(result).all())
            self.assertTrue((result >= 0).all())

    def test_invalid_graph_parameters_rejected(self):
        with self.assertRaisesRegex(ValueError, "alpha"):
            blend_adjacencies(torch.ones(1, 2, 2), torch.ones(1, 2, 2), 1.1)
        with self.assertRaisesRegex(ValueError, "top_k"):
            rgb_bilateral_graph(torch.ones(1, 3, 8, 8), FakeGraph(top_k=0), grid_size=2)


class CoreOverrideTests(unittest.TestCase):
    def make_core(self):
        hp = S2NetHyperparameters(
            num_feature_maps=2, num_regions=16, osc_dim=2,
            gamma_drive_mode="static", num_time_steps=4,
            gamma_phase_mode="standardize_tanh", theta_init="gamma",
            graph_mode="learned", graph_top_k=4, graph_spatial_decay=None,
            kuramoto_backend="factorized", k=8.0, freq_gain=1.0,
            branch=2, low_n=-2.0, high_n=0.0, spike_per_component=True,
            gate_mode="raw", spike_spatial_grid_size=4,
        ).validate()
        return S2NetCore(hp, device="cpu").eval()

    def test_none_override_matches_default_call_bitwise(self):
        torch.manual_seed(7)
        core = self.make_core()
        gamma = torch.rand(1, 2, 16)
        with torch.no_grad():
            base = core(gamma, return_core_out=True, return_theta=True)
            explicit_none = core(gamma, return_core_out=True, return_theta=True, graph_override=None)
        self.assertEqual(base[0], explicit_none[0])
        for left, right in zip(base[1:], explicit_none[1:]):
            self.assertTrue(torch.equal(left, right))

    def test_override_shape_and_finite_nonnegative_contract(self):
        core = self.make_core()
        gamma = torch.rand(1, 2, 16)
        learned = core.graph_generator(gamma)
        with torch.no_grad():
            result = core(gamma, return_core_out=True, graph_override=learned)
        self.assertEqual(tuple(result[1].shape[:2]), (1, 16))
        with self.assertRaisesRegex(ValueError, "shape"):
            core(gamma, graph_override=torch.ones(1, 15, 15))
        with self.assertRaisesRegex(ValueError, "finite nonnegative"):
            bad = torch.ones(1, 16, 16); bad[0, 0, 0] = -1
            core(gamma, graph_override=bad)


if __name__ == "__main__":
    unittest.main()
