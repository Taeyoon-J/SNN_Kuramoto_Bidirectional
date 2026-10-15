from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from collaborative_test.SW_0169_normalized_hard_reset import hard_reset as hr
from collaborative_test.SW_0130_phase_state_integration.model import PhaseStateIntegration
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from snn_kuramoto_bidirectional.dendric_layer import DendricLayer
from snn_kuramoto_bidirectional.membrane_layer import MembraneLayer


class _TinyKuramoto(nn.Module):
    spike_pulse_gain = None

    def forward(self, theta, drive, *, A, spike=None, coupling=None):
        return theta + 0.01 * torch.sin(theta) + 0.001 * drive


class _TinyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = 2
        self.in_dim = 3
        self.osc_dim = 4
        self.gamma_drive_mode = "static"
        self.spike_per_component = True
        self.kuramoto_backend = "pairwise"
        self.device = "cpu"
        self.num_time_steps = 6
        self.phase_delay_steps = 2
        self.gate_mode = "raw"
        self.graph_generator = None
        self.register_buffer("sc", torch.eye(self.in_dim))
        self.gamma_channel_proj = nn.Linear(self.T, self.osc_dim, bias=False)
        self.gamma_phase_gain = None
        self.kuramoto = _TinyKuramoto()
        self.dendric_layer = DendricLayer(
            self.in_dim, self.in_dim, low_n=0.1, high_n=0.1, branch=2,
            device="cpu", input_vector_dim=1)
        self.membrane_layer = MembraneLayer(
            self.in_dim, tau_minitializer="constant", low_m=0.4, high_m=0.4,
            vth=hr.BASE_THRESHOLD, device="cpu")

    def gamma_to_drive(self, gamma, channel_projection, phase_gain):
        return channel_projection(gamma.transpose(1, 2))

    def _init_theta(self, drive, batch):
        return drive


class _TinyWrapped(PhaseStateIntegration):
    def __init__(self):
        super().__init__(_TinyCore(), "phase")


class HardResetStepTests(unittest.TestCase):
    def test_hard_reset_uses_binary_history_and_actual_emission(self):
        # Eprev=1 resets V's carry regardless of the previous emitted amplitude.
        vprev = torch.tensor([[9.0, 2.0]], requires_grad=True)
        eprev = torch.tensor([[1.0, 0.0]])
        current = torch.tensor([[3.0, 4.0]], requires_grad=True)
        gate = torch.ones_like(current)

        def sigmoid_event(x):
            return torch.sigmoid(8.0 * x)

        out = hr.normalized_membrane_step(
            voltage=vprev, event_history=eprev,
            previous_membrane=vprev, previous_event=eprev,
            current=current, gate=gate, retention=torch.tensor(0.5),
            threshold=torch.tensor(0.5), conductance_scale=torch.tensor(1.0),
            event_fn=sigmoid_event)
        # First node: reset term is binary and its current contribution is 1.5.
        # Second node: no reset; carry 1.0 + current contribution 2.0.
        torch.testing.assert_close(out.voltage, torch.tensor([[1.5, 3.0]]))
        torch.testing.assert_close(out.emitted_spike, out.raw_event * gate)
        torch.testing.assert_close(out.history_event, out.raw_event)
        self.assertTrue(out.emitted_spike.requires_grad)

    def test_zero_gate_holds_voltage_and_binary_history_but_emits_zero(self):
        v = torch.tensor([[0.75, -0.25]])
        e = torch.tensor([[1.0, 0.0]])
        out = hr.normalized_membrane_step(
            voltage=v, event_history=e, previous_membrane=v,
            previous_event=e, current=torch.tensor([[100.0, -100.0]]),
            gate=torch.tensor([[0.0, 0.0]]), retention=torch.full_like(v, 0.2),
            threshold=torch.full_like(v, 0.5), conductance_scale=torch.ones_like(v))
        torch.testing.assert_close(out.voltage, v)
        torch.testing.assert_close(out.history_event, e)
        torch.testing.assert_close(out.emitted_spike, torch.zeros_like(v))

    def test_detached_control_detaches_only_carry_operands(self):
        v = torch.tensor([[0.9]], requires_grad=True)
        e = torch.tensor([[0.2]], requires_grad=True)
        theta = torch.tensor(0.3, requires_grad=True)
        current = torch.tensor([[0.4]], requires_grad=True)
        out = hr.normalized_membrane_step(
            voltage=v, event_history=e, previous_membrane=v,
            previous_event=e, current=current, gate=torch.ones_like(v),
            retention=torch.full_like(v, 0.5), threshold=theta,
            conductance_scale=torch.ones_like(v), detach_history=True,
            event_fn=lambda x: torch.sigmoid(4.0 * x))
        out.emitted_spike.sum().backward()
        self.assertIsNone(v.grad)
        self.assertIsNone(e.grad)
        self.assertIsNotNone(theta.grad)
        self.assertIsNotNone(current.grad)

    def test_calibration_is_active_gate_rms_with_cpu_float64_accumulation(self):
        h = torch.ones((2, 4, 3, 4), dtype=torch.float32)
        gate = torch.ones_like(h)
        gate[:, :, :, 2:] = 0
        h[:, :, :, 1] = 3
        result = hr.source_calibration(h, gate, settle=0)
        expected = float(hr.R_m) * np.sqrt((1.0 + 9.0) / 2.0)
        self.assertEqual(result["active_counts"], [12] * 4)
        self.assertEqual(result["r"].dtype, torch.float32)
        np.testing.assert_allclose(result["r"].numpy(), expected, rtol=1e-6)

    def test_calibration_rejects_missing_active_support_and_zero_rms(self):
        h = torch.zeros((1, 4, 2, 3))
        gate = torch.ones_like(h)
        with self.assertRaisesRegex(ValueError, "RMS"):
            hr.source_calibration(h, gate, settle=0)
        h.fill_(1.0)
        gate[:, 2] = 0
        with self.assertRaisesRegex(ValueError, "no active"):
            hr.source_calibration(h, gate, settle=0)

    def test_initialization_uses_registered_e_fold_and_normalized_threshold(self):
        from collaborative_test.SW_0169_normalized_hard_reset.calibration import _ordered_calibration_rows
        wrapped = _TinyWrapped()
        tau_before = wrapped.core.membrane_layer.tau_m.detach().clone()
        report = hr.initialize_slow_membrane_(wrapped)
        target = np.log(np.exp(-1.0 / 16.0) / (1.0 - np.exp(-1.0 / 16.0)))
        expected = tau_before - tau_before.median() + target
        torch.testing.assert_close(wrapped.core.membrane_layer.tau_m, expected)
        self.assertAlmostEqual(report["median_retention"], np.exp(-1.0 / 16.0), places=6)
        torch.testing.assert_close(0.06 * torch.exp(wrapped.b), torch.full((4,), 0.5))
        ids, rows = _ordered_calibration_rows(list(range(5000, 9096)), np.arange(4096))
        self.assertEqual(ids, list(range(5000, 5008)))
        np.testing.assert_array_equal(rows, np.arange(8))
        with self.assertRaisesRegex(ValueError, "4096"):
            _ordered_calibration_rows([1, 2], np.array([0, 1]))

    def test_opt_in_source_mode_exactly_replays_native_six_trace(self):
        torch.manual_seed(9)
        wrapped = _TinyWrapped()
        gamma = torch.randn(1, 2, 3)
        original = late_rollout(wrapped, gamma, total_steps=6, live_tail_steps=2)
        replay = hr.rollout(wrapped, gamma, torch.ones(4), total_steps=6,
                            live_tail_steps=2, source_mode=True)
        for key in ("theta", "membrane", "spikes", "component_membrane",
                    "component_spikes", "component_gates"):
            self.assertTrue(torch.equal(original[key], replay[key]), key)
        self.assertEqual(tuple(replay["h_wave"].shape), (1, 4, 3, 6))

    def test_live_and_detached_arms_have_same_forward_trace_and_event_identity(self):
        torch.manual_seed(17)
        wrapped = _TinyWrapped()
        hr.initialize_slow_membrane_(wrapped)
        gamma = torch.randn(1, 2, 3)
        live = hr.rollout(wrapped, gamma, torch.ones(4), total_steps=6,
                          live_tail_steps=2, detach_history=False)
        detached = hr.rollout(wrapped, gamma, torch.ones(4), total_steps=6,
                              live_tail_steps=2, detach_history=True)
        for key in ("theta", "membrane", "spikes", "component_membrane",
                    "component_spikes", "component_gates", "raw_events", "event_history"):
            self.assertTrue(torch.equal(live[key], detached[key]), key)
        self.assertTrue(torch.equal(live["component_spikes"],
                                    live["raw_events"] * live["component_gates"]))
        self.assertEqual(tuple(live["normalized_current"].shape), (1, 4, 3, 6))


if __name__ == "__main__":
    unittest.main()
