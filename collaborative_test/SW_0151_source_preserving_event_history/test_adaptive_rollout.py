from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for import_root in (ROOT, ROOT / "snn_kuramoto_bidirectional"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from collaborative_test.SW_0130_phase_state_integration.model import PhaseStateIntegration
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from collaborative_test.SW_0151_source_preserving_event_history.adaptive_rollout import adaptive_rollout
from collaborative_test.SW_0151_source_preserving_event_history.event_history import AdaptiveEventHistory
from snn_kuramoto_bidirectional.hyperparameter import S2NetHyperparameters
from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity
from snn_kuramoto_bidirectional.sinusoidal_gating import sinusoidal_gating


STEPS = 1024


def _core():
    hp = S2NetHyperparameters(
        num_feature_maps=8, num_regions=4, sc=torch.eye(4), osc_dim=4,
        gamma_drive_mode="static", gamma_phase_mode="standardize_tanh",
        theta_init="gamma", graph_mode="learned", graph_top_k=4,
        graph_spatial_decay=.35, geodesic_steps=3, geodesic_radius=1.5,
        geodesic_contrast=2.0, geodesic_temperature=.5, geodesic_cap=16.0,
        kuramoto_backend="factorized", k=256.0, freq_gain=2.0,
        num_time_steps=STEPS, spike_per_component=True, gate_mode="raw",
        membrane_vth=.06, spike_pulse_gain=0.0, dendritic_projection="shared",
        spike_spatial_grid_size=2,
    ).validate()
    core = S2NetCore(hp, device="cpu")
    core._detect_object_groups = lambda _out, spikes: [[] for _ in range(spikes.size(0))]
    return core


def _wrapped_pair(seed=15101):
    torch.manual_seed(seed)
    template = PhaseStateIntegration(_core(), "phase")
    with torch.no_grad():
        template.a_d.copy_(torch.tensor([.20, .13, .27, .08]))
        template.a_m.copy_(torch.tensor([.11, .24, .09, .17]))
        template.b.copy_(torch.tensor([.10, -.12, .18, -.05]))
    state = copy.deepcopy(template.state_dict())
    ref, candidate = PhaseStateIntegration(_core(), "phase"), PhaseStateIntegration(_core(), "phase")
    ref.load_state_dict(state, strict=True)
    candidate.load_state_dict(state, strict=True)
    return ref, candidate


class AdaptiveRolloutTests(unittest.TestCase):
    def test_kappa_zero_matches_all_six_native_outputs_at_tbptt_boundaries(self):
        torch.manual_seed(15102)
        gamma = torch.randn(2, 8, 4, dtype=torch.float32)
        rng = torch.random.get_rng_state()
        for tail in (0, 64, STEPS):
            with self.subTest(live_tail_steps=tail):
                reference, candidate = _wrapped_pair(15103)
                torch.random.set_rng_state(rng)
                expected = late_rollout(reference, gamma, total_steps=STEPS, live_tail_steps=tail)
                torch.random.set_rng_state(rng)
                helper = AdaptiveEventHistory()
                actual = adaptive_rollout(candidate, gamma, helper, total_steps=STEPS,
                                          live_tail_steps=tail, scale=.25)
                for key in ("component_membrane", "component_spikes", "component_gates",
                            "membrane", "spikes", "theta"):
                    self.assertTrue(torch.equal(actual[key], expected[key]), key)
                expected_q = spike_synchrony_affinity(
                    expected["spikes"], expected["component_spikes"],
                    settle=512, affinity_mode="spike")
                actual_q = spike_synchrony_affinity(
                    actual["spikes"], actual["component_spikes"],
                    settle=512, affinity_mode="spike")
                self.assertTrue(torch.equal(actual_q, expected_q), "production spike Q")
                self.assertEqual(actual["prefix_steps"], STEPS - tail)
                self.assertEqual(actual["live_tail_steps"], tail)
                self.assertEqual(actual["sw0134_rollout_sha256"],
                                 "66c08404af91b50fb4cbfa63fe44650dd6a698ea669cd3d6a672dfc5c78503f3")

    def test_nonzero_history_changes_events_and_receives_live_tail_credit(self):
        torch.manual_seed(15104)
        gamma = torch.randn(2, 8, 4, dtype=torch.float32)
        reference, candidate = _wrapped_pair(15105)
        expected = late_rollout(reference, gamma, total_steps=STEPS, live_tail_steps=64)
        helper = AdaptiveEventHistory()
        with torch.no_grad():
            helper.kappa_raw.fill_(0.5)
        actual = adaptive_rollout(candidate, gamma, helper, total_steps=STEPS,
                                  live_tail_steps=64, scale=.5)
        self.assertTrue(torch.equal(actual["theta"], expected["theta"]))
        self.assertTrue(torch.isfinite(actual["component_membrane"]).all())
        self.assertTrue(torch.isfinite(actual["component_spikes"]).all())
        self.assertTrue(torch.isfinite(actual["component_event_history"]).all())
        self.assertTrue((actual["component_spikes"] != expected["component_spikes"]).any())
        grad_lambda, grad_kappa = torch.autograd.grad(
            actual["component_spikes"][..., -64:].sum(),
            (helper.lambda_logit, helper.kappa_raw), allow_unused=False)
        self.assertTrue(torch.isfinite(grad_lambda).all())
        self.assertTrue(torch.isfinite(grad_kappa).all())
        self.assertGreater(float(grad_lambda.abs().sum().detach()), 0.0)
        self.assertGreater(float(grad_kappa.abs().sum().detach()), 0.0)
        self.assertTrue(torch.equal(candidate.core.last_component_spikes,
                                    actual["component_spikes"]))

    def test_adaptive_emitted_spike_is_next_dendrite_and_reset_input(self):
        torch.manual_seed(15106)
        gamma = torch.randn(1, 8, 4, dtype=torch.float32)
        reference, candidate = _wrapped_pair(15107)
        source_trace = late_rollout(reference, gamma, total_steps=STEPS, live_tail_steps=0)
        helper = AdaptiveEventHistory()
        with torch.no_grad():
            helper.kappa_raw.fill_(20.0)
        original = candidate._dense_dendrite
        captured = []

        def capture(layer, carrier, previous_spike):
            dense = original(layer, carrier, previous_spike)
            captured.append({
                "previous_h": layer.h.detach().clone(),
                "previous_spike": previous_spike.detach().clone(),
                "dense": dense.detach().clone(),
            })
            return dense

        with patch.object(candidate, "_dense_dendrite", side_effect=capture):
            trace = adaptive_rollout(candidate, gamma, helper, total_steps=STEPS,
                                     live_tail_steps=0, scale=.5)
        # At t=2, the native dendrite receives t=1's adaptive emitted spikes.
        step = 2
        previous_spike = captured[step]["previous_spike"]
        self.assertTrue(torch.equal(previous_spike.reshape(1, 4, 4),
                                    trace["component_spikes"][..., step - 1]))
        # Independently reconstruct the t=2 source membrane equation, including
        # native reset by the actual adaptive emitted spike.
        theta_history = [trace["theta"][:, t] for t in range(step + 1)]
        _, gate = sinusoidal_gating(theta_history, step, candidate.core.phase_delay_steps,
                                    gate_mode=candidate.core.gate_mode)
        folded_gate = gate.repeat_interleave(4, dim=0)
        previous_h = captured[step]["previous_h"]
        beta = torch.sigmoid(candidate.core.dendric_layer.tau_n).unsqueeze(0)
        native_h = beta * previous_h + (1.0 - beta) * captured[step]["dense"]
        ids = torch.arange(4)
        u = folded_gate.unsqueeze(-1)
        integrated_h = native_h + candidate.a_d[ids].view(4, 1, 1) * (u - 1.0) * (
            native_h - previous_h)
        h_wave = integrated_h.sum(dim=2)
        previous_mem = trace["component_membrane"][..., step - 1].reshape(4, 4)
        threshold = .06 * torch.exp(candidate.b[ids].view(4, 1))
        alpha = torch.sigmoid(candidate.core.membrane_layer.tau_m).unsqueeze(0)
        native_mem = previous_mem * alpha + (1.0 - alpha) * h_wave - threshold * previous_spike
        a_m = candidate.a_m[ids].view(4, 1)
        expected_mem = native_mem + a_m * (folded_gate - 1.0) * (native_mem - previous_mem)
        expected_mem = torch.where(folded_gate == 0, previous_mem, expected_mem)
        actual_mem = trace["component_membrane"][..., step].reshape(4, 4)
        self.assertTrue(torch.equal(actual_mem, expected_mem))
        source_spike = source_trace["component_spikes"][..., step - 1].reshape(4, 4)
        source_reset_mem = previous_mem * alpha + (1.0 - alpha) * h_wave - threshold * source_spike
        source_reset_mem = source_reset_mem + a_m * (folded_gate - 1.0) * (source_reset_mem - previous_mem)
        source_reset_mem = torch.where(folded_gate == 0, previous_mem, source_reset_mem)
        self.assertTrue((previous_spike != source_spike).any())
        self.assertTrue((actual_mem != source_reset_mem).any())
        self.assertTrue(torch.equal(candidate.core.membrane_layer.spike,
                                    trace["component_spikes"][..., -1].reshape(4, 4)))


if __name__ == "__main__":
    unittest.main()
