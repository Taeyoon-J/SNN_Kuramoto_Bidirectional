import numpy as np
import torch
import unittest
import torch.nn.functional as F

from adaptive_dynamics import AdaptiveMembraneLayer, VTH, install_adaptive_dynamics
from snn_kuramoto_bidirectional.membrane_layer import MembraneLayer


class AdaptiveMembraneTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)

    def make_layer(self):
        source = MembraneLayer(3, tau_minitializer="constant", low_m=0.0, vth=VTH, device="cpu")
        return AdaptiveMembraneLayer(source, torch.tensor([0.1, 0.2, 0.3, 0.4]))

    def test_folded_component_kappa_gate_holds_and_actual_spike_reset(self):
        layer = self.make_layer()
        layer.set_neuron_state(8)
        layer.mem = torch.arange(24, dtype=torch.float32).reshape(8, 3) / 100
        layer.adaptation = torch.full((8, 3), 0.2)
        layer.previous_event = torch.ones(8, 3)
        layer.spike = torch.full((8, 3), 0.4)
        old_mem, old_a = layer.mem.clone(), layer.adaptation.clone()
        h = torch.full((8, 3), 0.8)
        gate = torch.tensor([[0.0], [0.5], [1.0], [0.0], [1.0], [0.0], [0.25], [1.0]])
        mem, actual = layer(h, gate)
        active = gate.expand_as(h) > 0
        a_candidate = 0.9 * old_a + 0.1
        expected_a = torch.where(active, a_candidate, old_a)
        alpha = torch.sigmoid(layer.tau_m).unsqueeze(0)
        mem_candidate = alpha * old_mem + (1 - alpha) * h - VTH * 0.4
        expected_mem = torch.where(active, mem_candidate, old_mem)
        row_kappa = layer.kappa[torch.arange(8) % 4].unsqueeze(1)
        threshold = F.softplus(layer.raw_b[torch.arange(8) % 4]).unsqueeze(1) + row_kappa * expected_a
        expected_event = (expected_mem - threshold > 0).float() * active.float()
        torch.testing.assert_close(layer.adaptation, expected_a, rtol=0, atol=0)
        torch.testing.assert_close(mem, expected_mem, rtol=0, atol=0)
        torch.testing.assert_close(actual, expected_event * gate.expand_as(h), rtol=0, atol=0)
        self.assertTrue(torch.equal(mem[~active], old_mem[~active]))
        self.assertTrue(torch.equal(layer.adaptation[~active], old_a[~active]))

    def test_rollout_reset_clears_membrane_spike_adaptation_and_event(self):
        layer = self.make_layer()
        layer.set_neuron_state(4)
        layer(torch.ones(4, 3), torch.ones(4, 1))
        self.assertTrue(bool(layer.previous_event.any()))
        layer.set_neuron_state(4)
        for value in (layer.mem, layer.spike, layer.adaptation, layer.previous_event):
            self.assertTrue(torch.equal(value, torch.zeros_like(value)))

    def test_surrogate_gradient_reaches_input_retention_and_adaptive_base(self):
        layer = self.make_layer()
        layer.set_neuron_state(4)
        h = torch.full((4, 3), 0.2, requires_grad=True)
        mem, spikes = layer(h, torch.ones(4, 1))
        (mem.sum() + spikes.sum()).backward()
        for grad in (h.grad, layer.tau_m.grad, layer.raw_b.grad):
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.norm()), 0.0)

    def test_adaptive_installation_changes_only_registered_retention_and_adds_raw_base(self):
        class DummyCore(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.spike_per_component = True
                self.osc_dim = 4
                self.dendric_layer = torch.nn.Module()
                self.dendric_layer.tau_n = torch.nn.Parameter(torch.zeros(3, 4))
                self.membrane_layer = MembraneLayer(3, tau_minitializer="constant", low_m=0,
                                                    vth=VTH, device="cpu")
        core = DummyCore()
        old_layer = core.membrane_layer
        install_adaptive_dynamics(core, torch.ones(4) * 0.2)
        expected = torch.full((3, 4), np.log(9), dtype=torch.float32)
        torch.testing.assert_close(core.dendric_layer.tau_n, expected)
        torch.testing.assert_close(core.membrane_layer.tau_m, torch.full((3,), np.log(9)))
        self.assertIsNot(core.membrane_layer, old_layer)
        torch.testing.assert_close(F.softplus(core.membrane_layer.raw_b), torch.full((4,), VTH))
        self.assertTrue(torch.equal(core.membrane_layer.kappa, torch.full((4,), 0.2)))


if __name__ == "__main__":
    unittest.main()
