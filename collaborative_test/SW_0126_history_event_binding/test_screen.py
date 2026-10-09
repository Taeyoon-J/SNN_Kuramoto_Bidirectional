import unittest
import math
from unittest.mock import patch

import numpy as np  # initialize shared OpenMP runtime before torch on Windows
import torch
from torch import nn

from collaborative_test.SW_0126_history_event_binding.screen import (
    BATCH,
    _run_batch,
    activity_summary,
    assert_native_source_rollout_parity,
    assert_phase_carrier_gate_parity,
)
from collaborative_test.SW_0126_history_event_binding.binder import TemporalSlotRGBBinder
from collaborative_test.SW_0126_history_event_binding.history_event import HistoryBoundMembrane
from snn_kuramoto_bidirectional.membrane_layer import MembraneLayer


class ScreenGuardTests(unittest.TestCase):
    def test_real_batch_path_records_finite_binder_and_recurrent_gradient_credit(self):
        class Dendrite(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(1.0))

        class Core:
            def __init__(self, dendritic_layer=None):
                self.dendric_layer = dendritic_layer or Dendrite()

        layer = HistoryBoundMembrane(
            MembraneLayer(256, tau_minitializer="constant", low_m=0.0,
                          vth=0.06, dt=1, device="cpu"), capture_gate_trace=True)
        layer.set_neuron_state(4)
        unit = torch.arange(256, dtype=torch.float32).reshape(1, 256, 1)
        dendrite = Dendrite()
        # Drive membrane values across the history threshold repeatedly so the
        # actual hard event stream carries surrogate credit to every family.
        for step in range(512):
            phase = 2.0 * torch.pi * step / 24.0
            wave = (0.06 + 0.12 * torch.sin(phase + unit * 0.05)).reshape(1, 256)
            h_wave = dendrite.weight * wave.expand(4, -1)
            gate = torch.full((1, 256), 0.7 + 0.2 * math.sin(phase))
            layer(h_wave, gate)
        # Form emitted spikes from the captured forward values, preserving the
        # actual gate/event product and its surrogate-gradient graph.
        events = layer.component_event_trace(1)
        gates = layer.component_gate_trace(1)
        emitted = gates * events
        full_spikes = torch.cat((torch.zeros_like(emitted), emitted), dim=-1)
        full_membrane = torch.cat((torch.zeros_like(emitted), emitted), dim=-1)
        common = {"theta": torch.zeros(1), "carrier": torch.zeros(1), "gate": torch.zeros(1)}
        reference = {**common, "component_spikes": torch.zeros_like(full_spikes),
                     "component_membrane": torch.zeros_like(full_membrane)}
        adaptive = {**common, "component_spikes": full_spikes,
                    "component_membrane": full_membrane}
        binders = {"history_event": TemporalSlotRGBBinder(init_seed=18),
                   "gate_only": TemporalSlotRGBBinder(init_seed=18)}
        binders["gate_only"].load_state_dict(binders["history_event"].state_dict())
        gamma = torch.zeros(1, 8, 256)
        image = torch.full((1, 3, 128, 128), 127, dtype=torch.uint8)
        with patch("collaborative_test.SW_0126_history_event_binding.screen.BATCH", 1), \
             patch("collaborative_test.SW_0126_history_event_binding.screen.sw117.read_rgb",
                   return_value=image), \
             patch("collaborative_test.SW_0126_history_event_binding.screen.sw117.encode_rgb",
                   return_value=gamma), \
             patch("collaborative_test.SW_0126_history_event_binding.screen._full_rollout",
                   side_effect=(reference, adaptive)), \
             patch("collaborative_test.SW_0126_history_event_binding.screen.assert_native_source_rollout_parity",
                   return_value={"theta": True, "component_membrane": True,
                                 "component_spikes": True}):
            report, _ = _run_batch(
                0, 0, np.array([0]), np.array([1000]), gamma, gamma,
                encoder=None, patcher=None, mean=None, std=None, clip=None,
                source_core=Core(), adaptive_core=Core(dendrite), layer=layer,
                binders=binders, device="cpu")
        credit = report["arms"]["history_event"]["gradient_credit"]
        self.assertGreater(credit["assignment_norm"], 0)
        self.assertGreater(credit["beta_norm"], 0)
        self.assertGreater(credit["membrane_tau_m_norm"], 0)
        self.assertGreater(credit["dendritic_norm"], 0)
        self.assertTrue(report["arms"]["history_event"]["assignment_finite"])

    def test_native_rollout_comparison_requires_exact_production_traces(self):
        class NativeCore:
            def __call__(self, gamma, **kwargs):
                del kwargs
                self.last_component_out = torch.arange(24, dtype=gamma.dtype).reshape(1, 4, 2, 3)
                self.last_component_spikes = (self.last_component_out > 10).to(gamma.dtype)
                theta = torch.arange(24, dtype=gamma.dtype).reshape(1, 3, 2, 4)
                return [], torch.zeros(1), torch.zeros(1), theta

        gamma = torch.zeros(1)
        reference = {
            "theta": torch.arange(24, dtype=gamma.dtype).reshape(1, 3, 2, 4).permute(0, 2, 3, 1),
            "component_membrane": torch.arange(24, dtype=gamma.dtype).reshape(1, 4, 2, 3),
            "component_spikes": (torch.arange(24, dtype=gamma.dtype).reshape(1, 4, 2, 3) > 10).to(gamma.dtype),
        }
        self.assertTrue(all(assert_native_source_rollout_parity(NativeCore(), gamma, reference).values()))
        reference["component_spikes"][0, 0, 0, 0] = 1
        with self.assertRaisesRegex(AssertionError, "production source forward"):
            assert_native_source_rollout_parity(NativeCore(), gamma, reference)

    def test_activity_guard_requires_non_saturated_occupancy_and_mixed_units(self):
        events = torch.zeros(2, 4, 10, 512)
        events[:, :, :2, ::2] = 1.0
        record = activity_summary(events)
        self.assertTrue(record["pass"])
        for value in record["occupancy_by_component"]:
            self.assertAlmostEqual(value, 0.1, places=6)
        self.assertTrue(torch.as_tensor(record["mixed_unit_pass_by_image_component"]).all())
        saturated = activity_summary(torch.ones_like(events))
        self.assertFalse(saturated["pass"])

    def test_reference_parity_checks_only_fixed_phase_carrier_and_gate(self):
        ref = {"theta": torch.zeros(1, 2), "carrier": torch.ones(1, 2),
               "gate": torch.full((1, 2), 0.25),
               "component_membrane": torch.zeros(1, 4, 2)}
        candidate = {key: value.clone() for key, value in ref.items()}
        candidate["component_membrane"].add_(1.0)
        result = assert_phase_carrier_gate_parity(ref, candidate)
        self.assertEqual(result, {"theta": True, "carrier": True, "gate": True})
        candidate["gate"][0, 0] = 0.5
        with self.assertRaisesRegex(AssertionError, "parity failed"):
            assert_phase_carrier_gate_parity(ref, candidate)

    def test_activity_guard_rejects_nonbinary_or_wrong_horizon(self):
        with self.assertRaisesRegex(ValueError, "finite and binary"):
            activity_summary(torch.full((1, 4, 8, 512), 0.5))
        with self.assertRaisesRegex(ValueError, r"shape \[B,4,N,512\]"):
            activity_summary(torch.zeros(1, 4, 8, 64))


if __name__ == "__main__":
    unittest.main()
