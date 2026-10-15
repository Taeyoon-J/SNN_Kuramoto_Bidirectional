from __future__ import annotations

import unittest

import torch

from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from collaborative_test.SW_0169_normalized_hard_reset import hard_reset as hr
from collaborative_test.SW_0169_normalized_hard_reset.test_hard_reset import _TinyWrapped
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity


def _q(trace, settle: int):
    components = trace["component_spikes"]
    return spike_synchrony_affinity(
        components.mean(dim=1), components, settle=settle, affinity_mode="spike")


class RolloutContractTests(unittest.TestCase):
    def test_b2_source_replay_and_normalized_arm_forward_contract(self):
        torch.manual_seed(16901)
        wrapped = _TinyWrapped()
        gamma = torch.randn(2, 2, 3)

        native = late_rollout(wrapped, gamma, total_steps=10, live_tail_steps=3)
        replay = hr.rollout(
            wrapped, gamma, torch.ones(4), total_steps=10,
            live_tail_steps=3, source_mode=True)
        six = ("theta", "membrane", "spikes", "component_membrane",
               "component_spikes", "component_gates")
        for field in six:
            self.assertTrue(torch.equal(native[field], replay[field]), field)
        self.assertTrue(torch.equal(_q(native, 5), _q(replay, 5)))

        hr.initialize_slow_membrane_(wrapped)
        live = hr.rollout(
            wrapped, gamma, torch.ones(4), total_steps=10,
            live_tail_steps=3, detach_history=False)
        detached = hr.rollout(
            wrapped, gamma, torch.ones(4), total_steps=10,
            live_tail_steps=3, detach_history=True)
        fields = six + ("raw_events", "event_history", "normalized_current", "h_wave")
        for field in fields:
            self.assertTrue(torch.equal(live[field], detached[field]), field)
        self.assertTrue(torch.equal(_q(live, 5), _q(detached, 5)))
        self.assertEqual(tuple(live["normalized_current"].shape), (2, 4, 3, 10))
        self.assertEqual(tuple(live["h_wave"].shape), (2, 4, 3, 10))
        self.assertTrue(torch.equal(
            live["component_spikes"], live["raw_events"] * live["component_gates"]))

    def test_real_1024_boundary_and_tail_gradient(self):
        torch.manual_seed(16902)
        wrapped = _TinyWrapped()
        hr.initialize_slow_membrane_(wrapped)
        gamma = torch.randn(1, 2, 3, requires_grad=True)
        trace = hr.rollout(
            wrapped, gamma, torch.ones(4), total_steps=1024,
            live_tail_steps=64, detach_history=False)
        self.assertEqual(trace["prefix_steps"], 960)
        self.assertEqual(trace["live_tail_steps"], 64)
        self.assertTrue(trace["truncated_bptt"])
        self.assertEqual(tuple(trace["normalized_current"].shape), (1, 4, 3, 1024))
        prefix_grad = torch.autograd.grad(
            trace["raw_events"][..., :960].sum(), gamma,
            retain_graph=True, allow_unused=True)[0]
        self.assertTrue(prefix_grad is None or torch.count_nonzero(prefix_grad) == 0)
        loss = trace["raw_events"][..., 960:].sum()
        loss.backward()
        self.assertIsNotNone(gamma.grad)
        self.assertTrue(torch.isfinite(gamma.grad).all())
        self.assertGreater(float(gamma.grad.abs().sum()), 0.0)

    def test_actual_surrogate_carry_credit_is_removed_only_in_control(self):
        def two_steps(detach_history: bool):
            v0 = torch.tensor([[0.45]], requires_grad=True)
            e0 = torch.zeros_like(v0, requires_grad=True)
            first = hr.normalized_membrane_step(
                voltage=v0, event_history=e0,
                previous_membrane=v0, previous_event=e0,
                current=torch.tensor([[0.8]], requires_grad=True),
                gate=torch.ones_like(v0), retention=torch.tensor(0.8),
                threshold=torch.tensor(0.5, requires_grad=True),
                conductance_scale=torch.tensor(1.0, requires_grad=True))
            second = hr.normalized_membrane_step(
                voltage=first.voltage, event_history=first.history_event,
                previous_membrane=first.voltage, previous_event=first.history_event,
                current=torch.tensor([[0.7]], requires_grad=True),
                gate=torch.ones_like(v0), retention=torch.tensor(0.8),
                threshold=torch.tensor(0.5, requires_grad=True),
                conductance_scale=torch.tensor(1.0, requires_grad=True),
                detach_history=detach_history)
            grad = torch.autograd.grad(second.emitted_spike.sum(), first.voltage,
                                       allow_unused=True)[0]
            return first, second, grad

        live_first, live_second, live_grad = two_steps(False)
        detached_first, detached_second, detached_grad = two_steps(True)
        for left, right in zip(live_first, detached_first):
            self.assertTrue(torch.equal(left, right))
        for left, right in zip(live_second, detached_second):
            self.assertTrue(torch.equal(left, right))
        self.assertIsNotNone(live_grad)
        self.assertGreater(float(live_grad.abs().sum()), 0.0)
        self.assertIsNone(detached_grad)


if __name__ == "__main__":
    unittest.main()
