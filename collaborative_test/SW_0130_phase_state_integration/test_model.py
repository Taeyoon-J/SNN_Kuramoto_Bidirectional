"""Focused SW0130 adapter parity and parameter-path regressions."""
import copy
import sys
import unittest
from pathlib import Path

# Import NumPy before PyTorch in the Windows CPU environment. This avoids
# loading a second OpenMP runtime through NumPy after torch has initialized.
import numpy as np  # noqa: F401
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'snn_kuramoto_bidirectional')]

from snn_kuramoto_bidirectional.hyperparameter import S2NetHyperparameters
from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
from collaborative_test.SW_0130_phase_state_integration.model import (
    PhaseStateIntegration, strict_load_then_attach,
)


def make_core(steps=64):
    hp = S2NetHyperparameters(
        num_feature_maps=4, num_regions=8, sc=torch.eye(8), osc_dim=4,
        gamma_drive_mode='static', gamma_phase_mode='standardize_tanh',
        graph_mode='static', num_time_steps=steps, spike_per_component=True,
        kuramoto_backend='factorized', gate_mode='raw', theta_init='zeros',
        membrane_vth=0.06, spike_pulse_gain=0.0,
    ).validate()
    core = S2NetCore(hp, device='cpu')
    core._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.size(0))]
    return core


def make_source_style_core(steps):
    """Small learned-graph/gamma-theta core matching source97 dynamics flags."""
    hp = S2NetHyperparameters(
        num_feature_maps=8, num_regions=64, sc=torch.eye(64), osc_dim=4,
        gamma_drive_mode='static', gamma_phase_mode='standardize_tanh',
        theta_init='gamma', graph_mode='learned', graph_top_k=4,
        graph_spatial_decay=.35, geodesic_steps=3, geodesic_radius=1.5,
        geodesic_contrast=2.0, geodesic_temperature=.5,
        geodesic_cap=16.0, kuramoto_backend='factorized', k=256.0,
        freq_gain=2.0, num_time_steps=steps, spike_per_component=True,
        gate_mode='raw', membrane_vth=.06, spike_pulse_gain=0.0,
        dendritic_projection='shared', spike_spatial_grid_size=8,
    ).validate()
    core = S2NetCore(hp, device='cpu')
    core._detect_object_groups = lambda out, spikes: [[] for _ in range(spikes.size(0))]
    return core


class PhaseStateAdapterTests(unittest.TestCase):
    def test_strict_source_load_then_zero_initialized_arms_match_native(self):
        torch.manual_seed(130)
        source = make_core()
        state = copy.deepcopy(source.state_dict())
        target = make_core()
        wrapped = strict_load_then_attach(target, state, 'phase')
        self.assertEqual(set(target.state_dict()), set(state))
        self.assertTrue(torch.equal(wrapped.a_d, torch.zeros(4)))
        self.assertTrue(torch.equal(wrapped.a_m, torch.zeros(4)))
        self.assertTrue(torch.equal(wrapped.b, torch.zeros(4)))
        gamma = torch.randn(2, 4, 8)
        native = source(gamma, return_core_out=True, return_theta=True, num_time_steps=64)
        adapted = wrapped(gamma, return_core_out=True, return_theta=True, num_time_steps=64,
                          capture_state=True)
        self.assertEqual(native[0], adapted[0])
        for actual, expected in zip(adapted[1:], native[1:]):
            self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(source.last_component_spikes, target.last_component_spikes))
        self.assertTrue(torch.equal(source.last_component_out, target.last_component_out))
        self.assertTrue(torch.equal(wrapped.last_state_trace['integrated_h'],
                                    wrapped.last_state_trace['native_h']))
        self.assertTrue(torch.equal(wrapped.last_state_trace['integrated_membrane'],
                                    wrapped.last_state_trace['native_membrane']))

    def test_constant_arm_keeps_native_gate_and_uses_constant_integration_input(self):
        torch.manual_seed(131)
        core = make_core()
        wrapped = PhaseStateIntegration(core, 'constant')
        wrapped(torch.randn(2, 4, 8), num_time_steps=64, capture_state=True)
        trace = wrapped.last_state_trace
        self.assertTrue(torch.equal(trace['integration_gate'], torch.full_like(trace['gate'], 0.5)))
        self.assertTrue(torch.equal(trace['emitted_spikes'], trace['events'] * trace['gate']))
        self.assertFalse(torch.equal(trace['integration_gate'], trace['gate']))

    def test_both_arms_have_finite_nonzero_a_d_a_m_and_threshold_gradients(self):
        torch.manual_seed(132)
        gamma = torch.randn(2, 4, 8)
        for arm in ('phase', 'constant'):
            wrapped = PhaseStateIntegration(make_core(), arm)
            _, _, core_out = wrapped(gamma, return_core_out=True, num_time_steps=64)
            weights = torch.linspace(0.2, 1.3, core_out.numel()).reshape_as(core_out)
            loss = (core_out * weights).sum()
            grads = torch.autograd.grad(loss, (wrapped.a_d, wrapped.a_m, wrapped.b))
            for grad in grads:
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(float(grad.norm()), 0.0, arm)

    def test_state_resets_each_rollout_and_projection_is_post_step_only(self):
        torch.manual_seed(133)
        wrapped = PhaseStateIntegration(make_core(), 'phase')
        gamma = torch.randn(1, 4, 8)
        first = wrapped(gamma, return_core_out=True, return_theta=True, num_time_steps=64)
        wrapped(torch.randn(1, 4, 8), num_time_steps=64)
        second = wrapped(gamma, return_core_out=True, return_theta=True, num_time_steps=64)
        for left, right in zip(first[1:], second[1:]):
            self.assertTrue(torch.equal(left, right))
        with torch.no_grad():
            wrapped.a_d.copy_(torch.tensor([-0.1, 0.2, 1.1, 0.7]))
            wrapped.a_m.copy_(torch.tensor([0.2, -0.2, 0.4, 1.3]))
        self.assertLess(float(wrapped.a_d[0]), 0.0)  # no hidden forward-time clamp
        wrapped.project_integrations_()
        self.assertTrue(torch.equal(wrapped.a_d, torch.tensor([0., .2, 1., .7])))
        self.assertTrue(torch.equal(wrapped.a_m, torch.tensor([.2, 0., .4, 1.])))

    def test_rejects_unsupported_core_layout(self):
        hp = S2NetHyperparameters(num_feature_maps=4, num_regions=8, sc=torch.eye(8),
            osc_dim=2, gamma_drive_mode='static', graph_mode='static',
            spike_per_component=True, spike_spatial_grid_size=2).validate()
        with self.assertRaisesRegex(ValueError, 'four oscillator components'):
            PhaseStateIntegration(S2NetCore(hp, device='cpu'), 'phase')
        with self.assertRaisesRegex(ValueError, 'arm'):
            PhaseStateIntegration(make_core(), 'other')

    def test_source_style_gamma_theta_matches_native_at_both_horizons(self):
        torch.manual_seed(134)
        for steps in (64, 1024):
            source = make_source_style_core(steps)
            state = copy.deepcopy(source.state_dict())
            gamma = torch.randn(1, 8, 64)
            native = source(gamma, return_core_out=True, return_theta=True,
                            num_time_steps=steps)
            for arm in ('phase', 'constant'):
                target = make_source_style_core(steps)
                wrapped = strict_load_then_attach(target, state, arm)
                adapted = wrapped(gamma, return_core_out=True, return_theta=True,
                                  num_time_steps=steps)
                self.assertEqual(native[0], adapted[0], (steps, arm))
                for actual, expected in zip(adapted[1:], native[1:]):
                    self.assertTrue(torch.equal(actual, expected), (steps, arm))
                self.assertTrue(torch.equal(source.last_component_spikes,
                                            target.last_component_spikes), (steps, arm))
                self.assertTrue(torch.equal(source.last_component_out,
                                            target.last_component_out), (steps, arm))


if __name__ == '__main__':
    unittest.main()
