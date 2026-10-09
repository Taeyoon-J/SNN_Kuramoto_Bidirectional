import unittest

import numpy as np
import torch

from collaborative_test.SW_0120_degenerate_trace_gradient_guard.affinity import guarded_spike_affinity
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity


class GuardedAffinityTests(unittest.TestCase):
    def test_forward_is_bitwise_equal_with_normal_zero_and_tiny_traces(self):
        torch.manual_seed(120)
        components = torch.randn(2, 4, 7, 18)
        components[0, 0, 0] = 0
        components[0, 1, 1] = 3.0
        components[1, 2, 2] *= 1e-12
        activity = components.mean(dim=1)
        expected = spike_synchrony_affinity(activity, components=components, settle=4,
                                            affinity_mode="spike")
        actual = guarded_spike_affinity(activity, components=components, settle=4)
        self.assertTrue(torch.equal(actual, expected))

    def test_degenerate_norm_gradients_are_zero_and_valid_trace_gradients_remain(self):
        torch.manual_seed(121)
        components = torch.randn(1, 4, 5, 20, requires_grad=True)
        with torch.no_grad():
            components[0, 0, 0] = 0.0
            components[0, 1, 1] = 2.0
            components[0, 2, 2] *= 1e-12
            t = torch.arange(20, dtype=components.dtype)
            for d in range(4):
                wave = 0.2 + 0.03 * d
                components[0, d, 3] = torch.sin(t * wave)
                components[0, d, 4] = torch.sin(t * wave + 0.25)
        affinity = guarded_spike_affinity(components.mean(dim=1), components=components, settle=3)
        probe = affinity[0, 0, 1] + affinity[0, 1, 2] + affinity[0, 3, 4]
        grad, = torch.autograd.grad(probe, components)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertTrue(torch.equal(grad[0, 0, 0], torch.zeros_like(grad[0, 0, 0])))
        self.assertTrue(torch.equal(grad[0, 1, 1], torch.zeros_like(grad[0, 1, 1])))
        self.assertTrue(torch.equal(grad[0, 2, 2], torch.zeros_like(grad[0, 2, 2])))
        self.assertGreater(float(grad[0, 3].abs().sum()), 0.0)

        # On a loss path involving only valid traces, the intervention must be
        # an exact identity for their gradients, even in the same mixed batch.
        legacy_input = components.detach().clone().requires_grad_(True)
        guarded_input = components.detach().clone().requires_grad_(True)
        legacy = spike_synchrony_affinity(legacy_input.mean(dim=1),
                                          components=legacy_input, settle=3,
                                          affinity_mode="spike")
        guarded = guarded_spike_affinity(guarded_input.mean(dim=1),
                                         components=guarded_input, settle=3)
        legacy_grad, = torch.autograd.grad(legacy[0, 3, 4], legacy_input)
        guarded_grad, = torch.autograd.grad(guarded[0, 3, 4], guarded_input)
        self.assertTrue(torch.equal(legacy_grad[:, :, 3:5], guarded_grad[:, :, 3:5]))

    def test_legacy_zero_trace_can_have_large_gradient_but_guard_suppresses_it(self):
        time = torch.arange(16, dtype=torch.float32)
        base = torch.stack((torch.zeros_like(time), torch.sin(time), torch.cos(time),
                            torch.sin(time * 0.7)), dim=0).view(1, 1, 4, 16)
        # Two nodes: a constant first-node component and a valid second-node trace.
        components = base.expand(1, 1, 4, 16).clone()
        components = torch.cat((components, components.roll(1, dims=-1)), dim=2).requires_grad_()
        activity = components.mean(dim=1)
        old = spike_synchrony_affinity(activity, components=components, affinity_mode="spike")
        new = guarded_spike_affinity(activity, components=components)
        self.assertTrue(torch.equal(old, new))
        old_grad, = torch.autograd.grad(old[0, 0, 1], components, retain_graph=True)
        new_grad, = torch.autograd.grad(new[0, 0, 1], components)
        self.assertTrue(torch.isfinite(new_grad).all())
        self.assertGreater(float(old_grad.abs().max()), 1.0)
        self.assertLess(float(new_grad[0, 0, 0].abs().max()),
                        float(old_grad[0, 0, 0].abs().max()))

    def test_settle_precedes_centering_and_single_activity_matches_reference(self):
        torch.manual_seed(122)
        x = torch.randn(3, 9, 15, dtype=torch.float64)
        ref = spike_synchrony_affinity(x, settle=5, affinity_mode="spike")
        got = guarded_spike_affinity(x, settle=5)
        self.assertTrue(torch.equal(got, ref))

    def test_boundary_uses_existing_epsilon_without_changing_forward(self):
        eps = 1e-8
        x = torch.tensor([[[0.0, eps / 4, 0.0, 0.0],
                           [0.0, 0.0, eps / 4, 0.0]]])
        self.assertTrue(torch.equal(
            guarded_spike_affinity(x),
            spike_synchrony_affinity(x, affinity_mode="spike", eps=eps)))


if __name__ == "__main__":
    unittest.main()
