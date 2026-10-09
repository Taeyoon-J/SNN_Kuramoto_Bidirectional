import unittest

import numpy as np  # initialize the shared OpenMP runtime before torch on Windows
import torch
from torch import nn

from snn_kuramoto_bidirectional.membrane_layer import MembraneLayer
from collaborative_test.SW_0126_history_event_binding.history_event import (
    HistoryBoundMembrane,
    S2NetCoreMembraneLayer,
    head_trace_for_arm,
    strict_load_source_then_attach,
)


class _TinyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.osc_dim = 4
        self.membrane_layer = MembraneLayer(3, tau_minitializer="constant",
                                            low_m=0.0, vth=0.06, device="cpu")
        self.readout = nn.Linear(3, 2)


class HistoryEventTests(unittest.TestCase):
    def _native(self):
        return MembraneLayer(3, tau_minitializer="constant", low_m=0.0,
                             vth=0.06, device="cpu")

    def test_source_state_strict_load_precedes_opt_in_replacement(self):
        source_core = _TinyCore()
        target_core = _TinyCore()
        source_state = source_core.state_dict()
        tau_reference = target_core.membrane_layer.tau_m
        adapted = strict_load_source_then_attach(target_core, source_state)
        core = target_core
        self.assertIs(core.membrane_layer, adapted)
        self.assertIs(adapted.tau_m, tau_reference)
        self.assertTrue(torch.equal(adapted.tau_m, source_state["membrane_layer.tau_m"]))
        self.assertEqual(set(adapted.state_dict()), {"tau_m", "beta_logits"})
        self.assertTrue(torch.allclose(torch.sigmoid(adapted.beta_logits),
                                       torch.full((4,), 0.95)))
        rejected = _TinyCore()
        with self.assertRaises(RuntimeError):
            strict_load_source_then_attach(rejected, {"membrane_layer.tau_m": source_state["membrane_layer.tau_m"]})
        self.assertIsInstance(rejected.membrane_layer, MembraneLayer)

    def test_actual_factory_membrane_identity_strict_loads_and_attaches(self):
        # S2NetCore imports `membrane_layer` as a top-level module, which is a
        # second module identity for the same registered source class file.
        from collaborative_test.SW_0110_xy_graph_route import run as base
        core = base.make_core("cpu", steps=64)
        source_state = {key: value.detach().clone() for key, value in core.state_dict().items()}
        source_membrane = core.membrane_layer
        self.assertIs(type(source_membrane), S2NetCoreMembraneLayer)
        tau_reference = source_membrane.tau_m
        names_before = set(dict(core.named_parameters()))
        adapted = strict_load_source_then_attach(core, source_state)
        self.assertIs(core.membrane_layer, adapted)
        self.assertIs(adapted.tau_m, tau_reference)
        self.assertEqual(set(dict(core.named_parameters())) - names_before,
                         {"membrane_layer.beta_logits"})
        self.assertEqual(tuple(adapted.beta_logits.shape), (4,))

    def test_gate_closed_preserves_membrane_and_adaptation_history(self):
        layer = HistoryBoundMembrane(self._native())
        layer.set_neuron_state(4)
        layer.mem.fill_(0.4)
        layer.spike.fill_(0.2)
        layer.event_history.fill_(0.17)
        old_mem = layer.mem.clone()
        old_history = layer.event_history.clone()
        mem, spike = layer(torch.ones_like(layer.mem), torch.zeros_like(layer.mem))
        self.assertTrue(torch.equal(mem, old_mem))
        self.assertTrue(torch.equal(layer.event_history, old_history))
        self.assertTrue(torch.equal(spike, torch.zeros_like(spike)))

    def test_history_event_forward_and_beta_gradient_are_finite(self):
        layer = HistoryBoundMembrane(self._native(), capture_gate_trace=True)
        layer.set_neuron_state(4)
        h = torch.ones(4, 3, requires_grad=True)
        _, first = layer(h, torch.ones_like(h))
        _, second = layer(torch.zeros_like(h), torch.ones_like(h))
        self.assertTrue(torch.equal(first, torch.ones_like(first)))
        self.assertTrue(torch.equal(second, torch.ones_like(second)))
        self.assertEqual(tuple(layer.component_gate_trace(1).shape), (1, 4, 3, 2))
        second.sum().backward()
        self.assertIsNotNone(layer.beta_logits.grad)
        self.assertTrue(torch.isfinite(layer.beta_logits.grad).all())
        self.assertGreater(float(layer.beta_logits.grad.abs().sum()), 0.0)
        self.assertTrue(torch.isfinite(h.grad).all())

    def test_recent_event_changes_next_hard_event_relative_to_native_threshold(self):
        layer = HistoryBoundMembrane(self._native())
        layer.set_neuron_state(4)
        # First frame creates positive adaptation history; second places u
        # between the native zero threshold and that history value.
        _, first = layer(torch.full((4, 3), 0.5), torch.ones(4, 3))
        self.assertTrue(torch.equal(first, torch.ones_like(first)))
        _, second = layer(torch.full((4, 3), -0.002), torch.ones(4, 3))
        self.assertTrue(torch.all((layer.mem - 0.06 > 0)))
        self.assertTrue(torch.equal(second, torch.zeros_like(second)))

    def test_state_resets_each_rollout_and_detach_keeps_values(self):
        layer = HistoryBoundMembrane(self._native())
        layer.set_neuron_state(4)
        layer(torch.ones(4, 3), torch.ones(4, 3))
        state = layer.event_history.clone()
        layer.detach_rollout_boundary()
        self.assertTrue(torch.equal(layer.event_history, state))
        self.assertFalse(layer.event_history.requires_grad)
        layer.set_neuron_state(4)
        self.assertTrue(torch.equal(layer.event_history, torch.zeros_like(state)))

    def test_gate_only_changes_head_signal_not_event_signal(self):
        event = torch.tensor([[[[0.0, 1.0]]]])
        gate = torch.tensor([[[[0.4, 0.8]]]])
        self.assertIs(head_trace_for_arm(event, gate, "history_event"), event)
        self.assertIs(head_trace_for_arm(event, gate, "gate_only"), gate)
        with self.assertRaises(ValueError):
            head_trace_for_arm(event, gate, "other")


if __name__ == "__main__":
    unittest.main()
