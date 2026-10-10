from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test"), str(ROOT / "snn_kuramoto_bidirectional")]

from collaborative_test.SW_0130_phase_state_integration.model import PhaseStateIntegration
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from collaborative_test.SW_0140_native32_causal_screen.causal_rollout import (
    CONDITIONS, INTERVENTIONS, assert_none_parity, causal_rollout,
)
from collaborative_test.SW_0140_native32_causal_screen.screen import first_train_records
from snn_kuramoto_bidirectional.kuramoto_layer import graphVectorKuramoto
from snn_kuramoto_bidirectional.membrane_layer import R_m, act_fun_adp


class TinyDendrite(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.tau_n = nn.Parameter(torch.linspace(-.8, .8, n * 4).reshape(n, 4))
        self.oscillator_dense = nn.Linear(2, 4)
        self.dendritic_projection = "shared"
        self.h = None

    def set_neuron_state(self, batch):
        self.h = torch.zeros(batch, self.tau_n.shape[0], 4)


class TinyMembrane(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.tau_m = nn.Parameter(torch.linspace(-.7, .7, n))
        self.vth = .06
        self.mem = None
        self.spike = None

    def set_neuron_state(self, batch):
        self.mem = torch.zeros(batch, self.tau_m.shape[0])
        self.spike = torch.zeros_like(self.mem)


class TinyNativeCore(nn.Module):
    def __init__(self, n=8):
        super().__init__()
        self.in_dim = self.T = n
        self.osc_dim = 4
        self.spike_per_component = True
        self.gamma_drive_mode = "static"
        self.phase_delay_steps = 2
        self.gate_mode = "raw"
        self.kuramoto_backend = "pairwise"
        self.num_time_steps = 16
        self.device = torch.device("cpu")
        self.gamma_channel_proj = None
        self.gamma_phase_gain = None
        self.graph_generator = None
        self.register_buffer("sc", torch.eye(n))
        self.kuramoto = graphVectorKuramoto(n, D=4, K=.7, dt=.03,
                                             spike_pulse_gain=0.0,
                                             kuramoto_backend="pairwise", device="cpu")
        self.dendric_layer = TinyDendrite(n)
        self.membrane_layer = TinyMembrane(n)
        self.last_component_out = None
        self.last_component_spikes = None
        self.last_gate_history = None

    def gamma_to_drive(self, gamma, _proj, _gain):
        mean = gamma.mean(dim=-1, keepdim=True)
        return mean.expand(-1, -1, self.osc_dim)

    def _init_theta(self, drive, batch_size):
        return drive.expand(batch_size, self.in_dim, self.osc_dim).contiguous()


class SW0140CausalRolloutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(14001)
            self.wrapped = PhaseStateIntegration(TinyNativeCore(), "phase")
            self.gamma = torch.randn(1, 8, 8) * .3

    def test_none_is_bitwise_equal_to_frozen_late_rollout_and_resets_state(self):
        expected = late_rollout(self.wrapped, self.gamma, total_steps=16, live_tail_steps=0)
        got = assert_none_parity(self.wrapped, self.gamma, total_steps=16)
        second = causal_rollout(self.wrapped, self.gamma, intervention="none", total_steps=16)
        for name in ("theta", "membrane", "spikes", "component_membrane",
                     "component_spikes", "component_gates"):
            self.assertTrue(torch.equal(expected[name], got[name]), name)
            self.assertTrue(torch.equal(got[name], second[name]), name)
        self.assertEqual(set(CONDITIONS), {"none", *INTERVENTIONS})

    def test_six_interventions_are_finite_and_event_forcing_keeps_gate_feedback(self):
        original_k = self.wrapped.core.kuramoto.K
        outputs = {name: causal_rollout(self.wrapped, self.gamma, intervention=name,
                                        total_steps=16, capture_internal=True)
                   for name in INTERVENTIONS}
        self.assertEqual(self.wrapped.core.kuramoto.K, original_k)
        for name, row in outputs.items():
            for field in ("theta", "membrane", "spikes", "component_membrane",
                          "component_spikes", "component_gates", "component_events",
                          "component_carrier", "component_dense", "component_dendrite"):
                self.assertTrue(torch.isfinite(row[field]).all(), (name, field))
            self.assertTrue(torch.equal(row["component_spikes"],
                                        row["component_events"] * row["component_gates"]))
        forced = outputs["binary_events_forced_on"]
        self.assertTrue(torch.equal(forced["component_events"], torch.ones_like(forced["component_events"])))
        self.assertTrue(torch.equal(forced["component_spikes"], forced["component_gates"]))
        self.assertTrue(torch.isfinite(forced["component_natural_threshold_events"]).all())
        threshold = .06 * torch.exp(self.wrapped.b).reshape(1, 4, 1, 1)
        expected_natural = act_fun_adp(forced["component_membrane"] - threshold)
        self.assertTrue(torch.equal(forced["component_natural_threshold_events"], expected_natural))
        constant = outputs["delayed_gate_constant_half"]
        self.assertTrue(torch.equal(constant["component_gates"],
                                    torch.full_like(constant["component_gates"], .5)))
        expected_carrier = torch.sin(constant["theta"]).permute(0, 3, 2, 1) * .5
        self.assertTrue(torch.equal(constant["component_carrier"], expected_carrier))

    def test_history_knockouts_keep_registered_feedthrough_and_reset_terms(self):
        base = causal_rollout(self.wrapped, self.gamma, total_steps=16, capture_internal=True)
        dend_off = causal_rollout(self.wrapped, self.gamma, intervention="dendrite_history_off",
                                  total_steps=16, capture_internal=True)
        beta = torch.sigmoid(self.wrapped.core.dendric_layer.tau_n)[None, None, ..., None]
        expected_h = (1.0 - beta) * dend_off["component_dense"]
        self.assertTrue(torch.equal(dend_off["component_dendrite"], expected_h))

        reset_off = causal_rollout(self.wrapped, self.gamma, intervention="spike_reset_off",
                                   total_steps=16, capture_internal=True)
        threshold = .06 * torch.exp(self.wrapped.b).reshape(1, 4, 1, 1)
        # At the first frame there is no prior emitted spike; later frames
        # differ only by the removed reset term when prior S is nonzero.
        reset_delta = reset_off["component_membrane"] - base["component_membrane"]
        prior_s = base["component_spikes"][..., 0]
        if bool(prior_s.ne(0).any()):
            expected_reset = threshold.squeeze(-1) * prior_s
            self.assertTrue(torch.allclose(reset_delta[..., 1], expected_reset, atol=1e-7, rtol=1e-6))
        self.assertEqual(tuple(dend_off["component_dendrite"].shape), (1, 4, 8, 4, 16))

    def test_rejects_unknown_or_nonraw_source_configuration(self):
        with self.assertRaisesRegex(ValueError, "unregistered"):
            causal_rollout(self.wrapped, self.gamma, intervention="spike_dropout", total_steps=16)
        self.wrapped.core.gate_mode = "sigmoid"
        with self.assertRaisesRegex(ValueError, "raw sinusoidal"):
            causal_rollout(self.wrapped, self.gamma, total_steps=16)

    def test_first_train_records_preserve_registered_pool_to_id_order(self):
        pool = torch.arange(4096, dtype=torch.int64).numpy()
        ids = (torch.arange(4096, dtype=torch.int64) + 1000).numpy()
        got_pool, got_ids = first_train_records(pool, ids)
        self.assertEqual(got_pool.tolist(), [0, 1, 2, 3])
        self.assertEqual(got_ids.tolist(), [1000, 1001, 1002, 1003])
        with self.assertRaisesRegex(ValueError, "4096"):
            first_train_records(pool[:4], ids[:4])


if __name__ == "__main__":
    unittest.main()
