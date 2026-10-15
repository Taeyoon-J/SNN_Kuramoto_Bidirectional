"""Focused SW0172 CPU contracts; import NumPy before Torch on Windows."""
import numpy as np  # noqa: F401
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F

from collaborative_test.SW_0172_graph_drive_separation import dual_gamma_rollout as dg
from collaborative_test.SW_0171_contextual_instance_graph.contextual_encoder import (
    ContextualFeatureEncoder,
)
from collaborative_test.SW_0130_phase_state_integration.model import PhaseStateIntegration
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from collaborative_test.SW_0162_connected_signed_partition.preflight import capture_source_coupling
from snn_kuramoto_bidirectional.hyperparameter import S2NetHyperparameters
from snn_kuramoto_bidirectional.gamma_initializer import (
    FeaturePatchGammaInitializer, GammaToDrive,
)
from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity


class _Graph(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(8, 4, bias=False)
        self.seen = []
        self.uses_feedback = False

    def forward(self, gamma):
        self.seen.append(gamma)
        z = F.normalize(self.projection(gamma.transpose(1, 2)), dim=-1)
        return torch.bmm(z, z.transpose(1, 2)).clamp_min(0)


class _Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.graph_generator = _Graph()
        self.gamma_to_drive = GammaToDrive(8, 1024, 4, drive_mode="static")
        self.gamma_channel_proj = nn.Linear(8, 4, bias=False)
        self.gamma_phase_gain = None
        self.kuramoto_backend = "factorized"
        self.kuramoto = _Coupling()


class _Coupling:
    def prepare_coupling(self, graph, **_kwargs):
        return (graph.clamp_min(0) + 1e-6, graph.new_zeros(()), graph.new_ones(()))


class _Wrapped(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = _Core()


def _tiny_native_rollout(wrapped, gamma, *, total_steps=None, live_tail_steps=64):
    """Exercise real graph/drive modules while keeping this contract test tiny."""
    core = wrapped.core
    adjacency = core.graph_generator(gamma)
    drive = core.gamma_to_drive(gamma, core.gamma_channel_proj, core.gamma_phase_gain)
    coupling = core.kuramoto.prepare_coupling(adjacency, batch_size=gamma.shape[0],
                                              num_units=gamma.shape[-1],
                                              device=gamma.device)[0]
    # Six source trace fields retain independent graph/drive paths as production does.
    return {
        "component_membrane": drive.unsqueeze(1),
        "component_spikes": (drive + coupling.mean()).unsqueeze(1),
        "component_gates": torch.sigmoid(drive).unsqueeze(1),
        "membrane": drive,
        "spikes": drive,
        "theta": coupling.mean(dim=-1),
        "live_tail_steps": live_tail_steps,
    }


class _BaseEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=0)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.conv(x)


class _ZeroResidual(nn.Module):
    def forward(self, x):
        return x.new_zeros((x.shape[0], 8, 128, 128))


class DualGammaRolloutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(172)
        self.wrapped = _Wrapped()
        self.gamma = torch.randn(1, 8, 1024, dtype=torch.float32)

    def test_shared_gamma_matches_frozen_rollout_trace_and_native_z(self):
        with patch.object(dg, "late_rollout", side_effect=_tiny_native_rollout):
            expected = _tiny_native_rollout(self.wrapped, self.gamma)
            actual = dg.run_dual_gamma_rollout(
                self.wrapped, self.gamma, self.gamma, live_tail_steps=64)
        for name in ("component_membrane", "component_spikes", "component_gates",
                     "membrane", "spikes", "theta"):
            torch.testing.assert_close(actual["trace"][name], expected[name], rtol=0, atol=0)
        z = F.normalize(self.wrapped.core.graph_generator.projection(
            self.gamma.transpose(1, 2)), dim=-1)
        torch.testing.assert_close(actual["z"], z, rtol=0, atol=0)
        self.assertEqual(tuple(actual["adjacency"].shape), (1, 1024, 1024))
        torch.testing.assert_close(actual["adjacency"],
            actual["graph_adjacency"].clamp_min(0) + 1e-6, rtol=0, atol=0)
        self.assertEqual(len(self.wrapped.core.graph_generator.seen), 2)
        self.assertIs(self.wrapped.core.graph_generator.seen[-1], self.gamma)
        self.assertEqual(len(self.wrapped.core.gamma_to_drive._forward_pre_hooks), 0)
        self.assertEqual(len(self.wrapped.core.graph_generator._forward_hooks), 0)
        self.assertEqual(len(self.wrapped.core.graph_generator.projection._forward_hooks), 0)

    def test_separated_route_keeps_context_graph_credit_and_base_drive_only(self):
        base = self.gamma.clone().requires_grad_(True)
        context = (self.gamma + 0.1).clone().requires_grad_(True)
        with patch.object(dg, "late_rollout", side_effect=_tiny_native_rollout):
            result = dg.run_dual_gamma_rollout(self.wrapped, context, base)
        self.assertIs(self.wrapped.core.graph_generator.seen[-1], context)
        drive_grad = torch.autograd.grad(
            result["trace"]["spikes"].sum(), (base, context), allow_unused=True,
            retain_graph=True)
        self.assertIsNotNone(drive_grad[0])
        self.assertGreater(float(drive_grad[0].abs().sum()), 0.0)
        self.assertIsNone(drive_grad[1])
        graph_grad = torch.autograd.grad(result["z"].sum(), context, allow_unused=True)[0]
        self.assertIsNotNone(graph_grad)
        self.assertGreater(float(graph_grad.abs().sum()), 0.0)

    def test_exception_restores_hooks_and_registered_module_state(self):
        core = self.wrapped.core
        before_state = {key: value.detach().clone() for key, value in core.state_dict().items()}
        before_drive_dict = set(core.gamma_to_drive.__dict__)
        with patch.object(dg, "late_rollout", side_effect=RuntimeError("injected rollout failure")):
            with self.assertRaisesRegex(RuntimeError, "injected rollout failure"):
                dg.run_dual_gamma_rollout(self.wrapped, self.gamma, self.gamma)
        self.assertEqual(set(core.gamma_to_drive.__dict__), before_drive_dict)
        self.assertEqual(len(core.gamma_to_drive._forward_pre_hooks), 0)
        self.assertEqual(len(core.graph_generator._forward_hooks), 0)
        self.assertEqual(len(core.graph_generator.projection._forward_hooks), 0)
        after_state = core.state_dict()
        for key, value in before_state.items():
            torch.testing.assert_close(after_state[key], value, rtol=0, atol=0)

    def test_gamma_pair_reuses_frozen_base_once_and_zero_residual_is_exact(self):
        base = _BaseEncoder()
        encoder = ContextualFeatureEncoder(base, seed=172001)
        encoder.context = _ZeroResidual()
        found = SimpleNamespace(
            encoder=encoder,
            patcher=FeaturePatchGammaInitializer(grid_size=32),
            feature_mean=torch.zeros(8), feature_std=torch.ones(8), feature_clip=8.0,
        )
        rgb = torch.randint(0, 256, (1, 128, 128, 3), dtype=torch.uint8)
        pair = dg.encode_gamma_pair(found, rgb)
        self.assertEqual(base.calls, 1)
        self.assertEqual(tuple(pair["gamma_base"].shape), (1, 8, 1024))
        torch.testing.assert_close(pair["gamma_context"], pair["gamma_base"], rtol=0, atol=0)
        torch.testing.assert_close(pair["context_features"], pair["base_features"], rtol=0, atol=0)

    def test_actual_native_rollout_short_geometry_shared_path_parity(self):
        """Exercise the real S2NetCore, GammaToDrive, graph, and coupling APIs."""
        torch.manual_seed(17202)
        n = 1024
        hp = S2NetHyperparameters(
            num_feature_maps=8, num_regions=n, sc=torch.eye(n), osc_dim=4,
            gamma_drive_mode="static", gamma_phase_mode="standardize_tanh",
            theta_init="gamma", graph_mode="learned", graph_top_k=2,
            graph_spatial_decay=.35, geodesic_steps=0, geodesic_radius=1.5,
            geodesic_contrast=2.0, geodesic_temperature=.5, geodesic_cap=16.0,
            kuramoto_backend="factorized", k=256.0, freq_gain=2.0,
            num_time_steps=4, spike_per_component=True, gate_mode="raw",
            membrane_vth=.06, spike_pulse_gain=0.0, dendritic_projection="shared",
            spike_spatial_grid_size=32,
        ).validate()
        wrapped = PhaseStateIntegration(S2NetCore(hp, device="cpu"), "phase")
        gamma = torch.randn(1, 8, n)
        raw_projection = []
        ph = wrapped.core.graph_generator.projection.register_forward_hook(
            lambda _module, _inputs, output: raw_projection.append(output))
        try:
            torch.manual_seed(17203)
            expected, expected_a = capture_source_coupling(
                wrapped.core.kuramoto,
                lambda: late_rollout(wrapped, gamma, total_steps=4, live_tail_steps=2))
            expected_z = F.normalize(raw_projection[-1], dim=-1)
            expected_projection = raw_projection[-1]
            torch.manual_seed(17203)
            actual = dg.run_dual_gamma_rollout(
                wrapped, gamma, gamma, total_steps=4, live_tail_steps=2)
        finally:
            ph.remove()
        self.assertEqual(len(raw_projection), 2)
        for name in ("theta", "membrane", "spikes", "component_membrane",
                     "component_spikes", "component_gates"):
            self.assertTrue(torch.equal(expected[name], actual["trace"][name]), name)
        self.assertTrue(torch.equal(expected_a, actual["adjacency"]))
        self.assertTrue(torch.equal(expected_projection, actual["projection_raw"]))
        self.assertTrue(torch.equal(expected_z, actual["z"]))
        q_expected = spike_synchrony_affinity(
            expected["spikes"], expected["component_spikes"], settle=0,
            affinity_mode="spike")
        q_actual = spike_synchrony_affinity(
            actual["trace"]["spikes"], actual["trace"]["component_spikes"],
            settle=0, affinity_mode="spike")
        self.assertTrue(torch.equal(q_expected, q_actual))

    def test_invalid_gamma_geometry_and_dtype_rejected(self):
        with self.assertRaises(ValueError):
            dg.run_dual_gamma_rollout(self.wrapped, self.gamma[:, :, :8], self.gamma[:, :, :8])
        with self.assertRaises(TypeError):
            dg.run_dual_gamma_rollout(self.wrapped, self.gamma.double(), self.gamma.double())


if __name__ == "__main__":
    unittest.main()
