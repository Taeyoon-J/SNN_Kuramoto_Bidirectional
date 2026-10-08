import unittest

import torch
from torch import nn

from snn_kuramoto_bidirectional.graph_generator import ImageConditionedGraph
from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
from collaborative_test.SW_0094_aligned_joint_pilot.run import hparams
from collaborative_test.SW_0110_xy_graph_route.xy_graph import XYGraphAdapter, attach_xy_graph


class XYGraphAdapterTests(unittest.TestCase):
    def make_graph(self):
        return ImageConditionedGraph(
            in_channels=8, hidden_dim=16, top_k=32, coupling_gain=8.0,
            temperature=0.1, grid_size=16, spatial_decay=0.35,
            geodesic_steps=3, geodesic_radius=1.5, geodesic_contrast=2.0,
            geodesic_temperature=0.5, geodesic_cap=16.0,
        ).eval()

    def test_zero_projection_is_exact_legacy_graph(self):
        torch.manual_seed(11)
        baseline = self.make_graph()
        candidate = XYGraphAdapter(self.make_graph()).eval()
        candidate.base_graph.load_state_dict(baseline.state_dict(), strict=True)
        gamma = torch.randn(2, 8, 256)
        self.assertTrue(torch.equal(baseline(gamma), candidate(gamma)))

    def test_xy_projection_has_finite_nonzero_gradient(self):
        torch.manual_seed(12)
        graph = XYGraphAdapter(self.make_graph())
        gamma = torch.randn(2, 8, 256)
        weight = torch.randn(2, 256, 256)
        objective = (graph(gamma) * weight).sum()
        grad, = torch.autograd.grad(objective, (graph.xy_projection,))
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.norm()), 0.0)

    def test_adapter_parameters_and_coordinates_follow_graph_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        holder = nn.Module()
        holder.graph_generator = self.make_graph().to(device)
        attach_xy_graph(holder)
        self.assertEqual(holder.graph_generator.xy_projection.device, device)
        self.assertEqual(holder.graph_generator.xy.device, device)
        gamma = torch.randn(2, 8, 256, device=device)
        self.assertEqual(holder.graph_generator(gamma).device, device)

    def test_roundtrip_preserves_legacy_graph_and_xy_parameter(self):
        torch.manual_seed(13)
        first = XYGraphAdapter(self.make_graph())
        with torch.no_grad():
            first.xy_projection[0, 0] = 0.125
        state = first.state_dict()
        second = XYGraphAdapter(self.make_graph())
        second.load_state_dict(state, strict=True)
        self.assertTrue(torch.equal(first.xy_projection, second.xy_projection))
        for key, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[key]), key)

    def test_zero_xy_initialization_preserves_real_s2net_rollout(self):
        torch.manual_seed(14)
        hp = hparams("raw")
        hp.num_time_steps = 8
        base = S2NetCore(hp.validate(), device="cpu").eval()
        candidate = S2NetCore(hp.validate(), device="cpu").eval()
        candidate.load_state_dict(base.state_dict(), strict=True)
        candidate.graph_generator = XYGraphAdapter(candidate.graph_generator)
        base._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.size(0))]
        candidate._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.size(0))]
        gamma = torch.randn(2, 8, 256)
        with torch.no_grad():
            expected = base(gamma, return_core_out=True, return_theta=True,
                            num_time_steps=8)
            actual = candidate(gamma, return_core_out=True, return_theta=True,
                               num_time_steps=8)
        for index, label in zip((1, 2, 3), ("actual spikes", "membrane", "phase")):
            self.assertTrue(torch.equal(expected[index], actual[index]), label)


if __name__ == "__main__":
    unittest.main()
