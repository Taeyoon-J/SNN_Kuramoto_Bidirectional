"""CPU parity and credit-path tests for the SW0134 late rollout helper."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "snn_kuramoto_bidirectional")]

import torch

from collaborative_test.SW_0130_phase_state_integration.model import (
    PhaseStateIntegration, strict_load_then_attach,
)
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from snn_kuramoto_bidirectional.hyperparameter import S2NetHyperparameters
from snn_kuramoto_bidirectional.s2net_cls import S2NetCore


def _core(steps=64):
    hp = S2NetHyperparameters(
        num_feature_maps=8, num_regions=64, sc=torch.eye(64), osc_dim=4,
        gamma_drive_mode="static", gamma_phase_mode="standardize_tanh",
        theta_init="gamma", graph_mode="learned", graph_top_k=4,
        graph_spatial_decay=.35, geodesic_steps=3, geodesic_radius=1.5,
        geodesic_contrast=2.0, geodesic_temperature=.5, geodesic_cap=16.0,
        kuramoto_backend="factorized", k=256.0, freq_gain=2.0,
        num_time_steps=steps, spike_per_component=True, gate_mode="raw",
        membrane_vth=.06, spike_pulse_gain=0.0, dendritic_projection="shared",
        spike_spatial_grid_size=8,
    ).validate()
    core = S2NetCore(hp, device="cpu")
    core._detect_object_groups = lambda _out, spikes: [[] for _ in range(spikes.size(0))]
    return core


class SW0134LateRolloutTests(unittest.TestCase):
    def test_zero_initialized_late_rollout_matches_native_and_full_adapter_traces(self):
        torch.manual_seed(1341)
        for arm in ("phase", "constant"):
            source = _core()
            source_state = copy.deepcopy(source.state_dict())
            gamma = torch.randn(1, 8, 64)
            native = source(gamma, return_core_out=True, return_theta=True,
                            num_time_steps=64)
            expected_mem = source.last_component_out.detach().clone()
            expected_spikes = source.last_component_spikes.detach().clone()
            target = _core()
            wrapped = strict_load_then_attach(target, source_state, arm)
            full = wrapped(gamma, return_core_out=True, return_theta=True,
                            num_time_steps=64, capture_state=True)
            expected_gates = target.last_gate_history.detach().clone()
            actual = late_rollout(wrapped, gamma, total_steps=64, live_tail_steps=32)
            self.assertEqual(native[0], full[0])
            self.assertTrue(torch.equal(actual["component_membrane"], expected_mem))
            self.assertTrue(torch.equal(actual["component_spikes"], expected_spikes))
            self.assertTrue(torch.equal(actual["component_membrane"], target.last_component_out))
            self.assertTrue(torch.equal(actual["component_spikes"], target.last_component_spikes))
            self.assertTrue(torch.equal(actual["theta"], full[3]))
            self.assertTrue(torch.equal(actual["component_gates"], expected_gates))
            self.assertEqual(tuple(actual["component_spikes"].shape), (1, 4, 64, 64))
            self.assertEqual(tuple(actual["component_gates"].shape), (1, 4, 64, 64))

    def test_zero_live_tail_is_no_grad_evaluation_not_truncated_bptt(self):
        torch.manual_seed(1344)
        wrapped = PhaseStateIntegration(_core(), "phase")
        gamma = torch.randn(1, 8, 64, requires_grad=True)
        output = late_rollout(wrapped, gamma, total_steps=64, live_tail_steps=0)
        self.assertFalse(output["truncated_bptt"])
        self.assertEqual(output["prefix_steps"], 64)
        self.assertEqual(output["live_tail_steps"], 0)
        for key in ("component_membrane", "component_spikes", "component_gates",
                    "membrane", "spikes", "theta"):
            self.assertFalse(output[key].requires_grad, key)

    def test_nonzero_phase_and_constant_integration_match_untruncated_adapter(self):
        torch.manual_seed(1342)
        gamma = torch.randn(1, 8, 64)
        for arm in ("phase", "constant"):
            wrapped = PhaseStateIntegration(_core(), arm)
            with torch.no_grad():
                wrapped.a_d.copy_(torch.tensor([.2, .3, .1, .25]))
                wrapped.a_m.copy_(torch.tensor([.15, .1, .2, .05]))
                wrapped.b.copy_(torch.tensor([.1, -.1, .15, -.05]))
            full = wrapped(gamma, return_core_out=True, return_theta=True,
                           num_time_steps=64, capture_state=True)
            expected_mem = wrapped.core.last_component_out.detach().clone()
            expected_spikes = wrapped.core.last_component_spikes.detach().clone()
            expected_gates = wrapped.core.last_gate_history.detach().clone()
            actual = late_rollout(wrapped, gamma, total_steps=64, live_tail_steps=32)
            self.assertTrue(torch.equal(actual["component_membrane"], expected_mem))
            self.assertTrue(torch.equal(actual["component_spikes"], expected_spikes))
            self.assertTrue(torch.equal(actual["component_gates"], expected_gates))
            self.assertTrue(torch.equal(actual["theta"], full[3]))
            self.assertTrue(torch.equal(actual["membrane"], full[2]))
            self.assertTrue(torch.equal(actual["spikes"], full[1]))

    def test_only_tail_has_parameter_credit_and_gates_have_no_membrane_credit(self):
        torch.manual_seed(1343)
        wrapped = PhaseStateIntegration(_core(), "phase")
        with torch.no_grad():
            wrapped.a_d.fill_(.2)
            wrapped.a_m.fill_(.15)
            wrapped.b.fill_(.1)
        raw_gamma = torch.randn(1, 8, 64)
        encoder_scale = torch.nn.Parameter(torch.ones(()))
        gamma = raw_gamma * encoder_scale
        output = late_rollout(wrapped, gamma, total_steps=64, live_tail_steps=32)
        tail_spikes = output["component_spikes"][..., 32:]
        weights = torch.linspace(.2, 1.7, tail_spikes.numel()).reshape_as(tail_spikes)
        tail_loss = (tail_spikes * weights).sum()
        prefix_loss = output["component_spikes"][..., :32].sum()
        params = (encoder_scale, wrapped.a_d, wrapped.a_m, wrapped.b,
                  wrapped.core.dendric_layer.oscillator_dense.weight,
                  wrapped.core.membrane_layer.tau_m)
        tail_grads = torch.autograd.grad(tail_loss, params, allow_unused=True, retain_graph=True)
        self.assertTrue(torch.isfinite(tail_grads[0]).all())
        self.assertGreater(float(tail_grads[0].norm()), 0.0)
        self.assertGreater(float(tail_grads[1].norm()), 0.0)
        self.assertGreater(float(tail_grads[2].norm()), 0.0)
        self.assertGreater(float(tail_grads[3].norm()), 0.0)
        self.assertIsNotNone(tail_grads[4])
        self.assertIsNotNone(tail_grads[5])
        self.assertGreater(float(tail_grads[4].norm()), 0.0)
        self.assertGreater(float(tail_grads[5].norm()), 0.0)
        prefix_grads = torch.autograd.grad(prefix_loss, params[1:], allow_unused=True,
                                           retain_graph=True)
        self.assertTrue(all(grad is None or not bool(grad.abs().any()) for grad in prefix_grads))

        gate_loss = output["component_gates"][..., 32:].sum()
        gate_grads = torch.autograd.grad(gate_loss, (wrapped.a_d, wrapped.a_m, wrapped.b,
            wrapped.core.dendric_layer.oscillator_dense.weight,
            wrapped.core.membrane_layer.tau_m), allow_unused=True)
        self.assertTrue(all(grad is None for grad in gate_grads))
        self.assertTrue(torch.equal(output["component_gates"][:, 0],
                                    output["component_gates"][:, 3]))

    def test_rejects_unsupported_or_malformed_source_rollout(self):
        wrapped = PhaseStateIntegration(_core(), "phase")
        with self.assertRaisesRegex(ValueError, "static source shape"):
            late_rollout(wrapped, torch.randn(1, 7, 64), total_steps=64, live_tail_steps=32)
        with self.assertRaisesRegex(ValueError, "total or live-tail"):
            late_rollout(wrapped, torch.randn(1, 8, 64), total_steps=64, live_tail_steps=65)


if __name__ == "__main__":
    unittest.main()
