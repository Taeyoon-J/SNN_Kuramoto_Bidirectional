import sys
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "snn_kuramoto_bidirectional")]

from collaborative_test.SW_0130_phase_state_integration.model import PhaseStateIntegration
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from collaborative_test.SW_0158_membrane_spike_credit import spike_credit
from snn_kuramoto_bidirectional.hyperparameter import S2NetHyperparameters
from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity


class MembraneSpikeCreditTests(unittest.TestCase):
    @staticmethod
    def tiny_wrapped():
        hp = S2NetHyperparameters(
            num_feature_maps=8, num_regions=4, num_time_steps=16,
            gamma_drive_mode="static", gamma_phase_mode="standardize_tanh",
            theta_init="gamma", theta_init_noise=0.0, graph_mode="learned",
            graph_hidden_dim=4, graph_top_k=2, graph_coupling_gain=2.0,
            graph_temperature=0.2, graph_spatial_decay=0.2,
            spike_per_component=True, spike_spatial_grid_size=2,
            kuramoto_backend="pairwise", geodesic_steps=0, spike_pulse_gain=0.0)
        return PhaseStateIntegration(S2NetCore(hp, device="cpu"), "phase")

    def test_registered_source_t_scale_is_report_bound_and_pooled(self):
        values = spike_credit.registered_margin_rms()
        self.assertEqual(values, spike_credit.SOURCE_T_MARGIN_RMS)
        self.assertEqual(spike_credit.sha256_file(spike_credit.SW157_REPORT),
                         spike_credit.SW157_REPORT_SHA256)

    def test_zero_eta_matches_all_native_traces_and_actual_spike_readouts(self):
        torch.manual_seed(15801)
        wrapped = self.tiny_wrapped()
        gamma = torch.randn(1, 8, 4)
        reference = late_rollout(wrapped, gamma, total_steps=16, live_tail_steps=16)
        spike_credit.install_spike_credit(wrapped, spike_credit.SOURCE_T_MARGIN_RMS)
        actual = spike_credit.late_rollout_credit(
            wrapped, gamma, total_steps=16, live_tail_steps=16)
        for key in ("theta", "membrane", "spikes", "component_membrane",
                    "component_spikes", "component_gates"):
            self.assertTrue(torch.equal(reference[key], actual[key]), key)
        reference_q = spike_synchrony_affinity(reference["spikes"],
            reference["component_spikes"], settle=8, affinity_mode="spike")
        actual_q = spike_synchrony_affinity(actual["spikes"],
            actual["component_spikes"], settle=8, affinity_mode="spike")
        self.assertTrue(torch.equal(reference_q, actual_q))
        self.assertTrue(torch.equal(wrapped.sw0158_eta, torch.zeros(4)))

    def test_emitter_has_exact_zero_path_and_live_only_added_margin_gradient(self):
        events = torch.tensor([[[1.], [0.]], [[1.], [1.]],
                               [[0.], [1.]], [[1.], [1.]]])
        gates = torch.full_like(events, .5)
        membrane = torch.tensor([[[.4], [.3]], [[.2], [.1]],
                                 [[.5], [.4]], [[.3], [.2]]], requires_grad=True)
        threshold = torch.full_like(events, .05, requires_grad=True)
        scales = torch.tensor(spike_credit.SOURCE_T_MARGIN_RMS)
        ids = torch.arange(4, dtype=torch.long)
        zero = torch.nn.Parameter(torch.zeros(4))
        native = events * gates
        zero_emitted = spike_credit.emit_spikes(events, gates, membrane, threshold,
                                                zero, scales, ids)
        self.assertTrue(torch.equal(zero_emitted, native))

        eta = torch.nn.Parameter(torch.tensor([.3, -.2, .4, .1]))
        live = spike_credit.emit_spikes(events, gates, membrane, threshold, eta, scales, ids)
        detached = spike_credit.emit_spikes(events, gates, membrane, threshold, eta, scales,
                                            ids, detached_control=True)
        self.assertTrue(torch.equal(live, detached))
        self.assertTrue(torch.isfinite(live).all())
        self.assertTrue(torch.all(live >= 0))
        active = native != 0
        ratio = live[active] / native[active]
        self.assertTrue(torch.all((ratio >= .5) & (ratio <= 1.5)))
        live_margin_grad, live_threshold_grad = torch.autograd.grad(
            live.sum(), (membrane, threshold), retain_graph=True)
        detached_grad, detached_threshold_grad = torch.autograd.grad(
            detached.sum(), (membrane, threshold), allow_unused=True)
        self.assertTrue(torch.isfinite(live_margin_grad).all())
        self.assertGreater(float(live_margin_grad.norm()), 0.0)
        self.assertIsNone(detached_grad)
        self.assertGreater(float(live_threshold_grad.norm()), 0.0)
        self.assertGreater(float(detached_threshold_grad.norm()), 0.0)

    def test_positive_eta_recurrence_uses_same_emitted_spike_for_dendrite_and_reset(self):
        torch.manual_seed(15802)
        wrapped = self.tiny_wrapped()
        scales = torch.tensor(spike_credit.SOURCE_T_MARGIN_RMS)
        spike_credit.install_spike_credit(wrapped, scales,
                                          eta_initial=torch.tensor([.4, -.2, .3, .1]))
        gamma = torch.randn(1, 8, 4)
        trace = spike_credit.late_rollout_credit(wrapped, gamma, total_steps=16,
                                                live_tail_steps=16, capture_state=True)
        state = trace["credit_state"]
        def folded(name):
            value = state[name]
            if value.ndim == 3 and value.shape[1] == 1:
                value = value.squeeze(1)
            return value.reshape(1, 4, 4, 16)
        emitted = trace["component_spikes"]
        self.assertTrue(torch.equal(folded("previous_spike")[..., 1:], emitted[..., :-1]))
        self.assertTrue(torch.equal(folded("dendrite_previous_spike"),
                                    folded("previous_spike")))
        previous_mem = folded("previous_membrane")
        previous_spike = folded("previous_spike")
        h_wave = folded("h_wave")
        alpha = folded("alpha")
        threshold = folded("threshold")
        native_mem = folded("native_membrane")
        expected = previous_mem * alpha + (1 - alpha) * h_wave - threshold * previous_spike
        self.assertTrue(torch.equal(native_mem, expected))
        self.assertTrue(torch.isfinite(trace["component_spikes"]).all())

    def test_two_step_actual_spike_shape_is_recurrent_reset_input(self):
        torch.manual_seed(15804)
        wrapped = self.tiny_wrapped()
        spike_credit.install_spike_credit(wrapped, spike_credit.SOURCE_T_MARGIN_RMS,
                                          eta_initial=torch.tensor([.5, -.3, .2, .1]))
        trace = spike_credit.late_rollout_credit(
            wrapped, torch.randn(1, 8, 4), total_steps=2, live_tail_steps=2,
            capture_state=True)
        state = trace["credit_state"]
        self.assertEqual(tuple(trace["component_spikes"].shape), (1, 4, 4, 2))
        first = trace["component_spikes"][..., 0].reshape(4, 4)
        self.assertTrue(torch.equal(state["previous_spike"][:, :, 1], first))
        self.assertTrue(torch.equal(state["dendrite_previous_spike"][:, :, 1], first))
        expected_native = (state["previous_membrane"][:, :, 1]
            * state["alpha"][:, :, 1]
            + (1 - state["alpha"][:, :, 1]) * state["h_wave"][:, :, 1]
            - state["threshold"][:, :, 1] * first)
        self.assertTrue(torch.equal(state["native_membrane"][:, :, 1], expected_native))

    def test_live_and_detached_rollouts_have_equal_forwards_at_same_state(self):
        torch.manual_seed(15803)
        wrapped = self.tiny_wrapped()
        spike_credit.install_spike_credit(wrapped, spike_credit.SOURCE_T_MARGIN_RMS,
                                          eta_initial=torch.tensor([.2, .3, -.1, .4]))
        gamma = torch.randn(1, 8, 4)
        live = spike_credit.late_rollout_credit(wrapped, gamma, total_steps=16,
                                                live_tail_steps=16)
        detached = spike_credit.late_rollout_credit(wrapped, gamma, total_steps=16,
            live_tail_steps=16, detached_control=True)
        for key in ("theta", "membrane", "spikes", "component_membrane",
                    "component_spikes", "component_gates"):
            self.assertTrue(torch.equal(live[key], detached[key]), key)
        self.assertTrue(torch.equal(live["component_spikes"], detached["component_spikes"]))


if __name__ == "__main__":
    unittest.main()
