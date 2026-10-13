from __future__ import annotations

import unittest
from unittest import mock

import torch

from collaborative_test.SW_0150_spike_spatial_mixture import integration


class FakeFoundation:
    def __init__(self):
        self.wrapped = torch.nn.Identity()
        self.encoded = []
        self.rollouts = []
        self.trace = {"token": object()}

    def encode(self, rgb):
        self.encoded.append(rgb)
        return torch.zeros((rgb.shape[0], 8, 1024), dtype=torch.float32)

    def rollout(self, gamma, *, total_steps, live_tail_steps):
        self.rollouts.append((gamma, total_steps, live_tail_steps))
        return self.trace


class Native32AssignmentIntegrationTests(unittest.TestCase):
    def test_one_trace_drives_registered_qcc_and_actual_q_with_live_diffusion_gradient(self):
        foundation = FakeFoundation()
        rgb = torch.zeros((1, 128, 128, 3), dtype=torch.uint8)
        scale = torch.tensor(0.0, requires_grad=True)
        q = torch.full((1, 1024, 1024), 0.25) + 0.05 * scale
        qcc = torch.zeros((1, 32, 32), dtype=torch.int64)
        qcc.reshape(-1)[:512] = 1

        with (mock.patch.object(integration.sw145, "_actual_q", return_value=q) as actual_q,
              mock.patch.object(integration.sw136_eval, "_qcc_labels", return_value=qcc) as qcc_readout):
            result = integration.forward_assignment(foundation, rgb, live_tail_steps=64)

        self.assertEqual(len(foundation.encoded), 1)
        self.assertIs(foundation.encoded[0], rgb)
        self.assertEqual(len(foundation.rollouts), 1)
        gamma, steps, tail = foundation.rollouts[0]
        self.assertIs(result["gamma"], gamma)
        self.assertEqual((steps, tail), (1024, 64))
        self.assertIs(result["trace"], foundation.trace)
        actual_q.assert_called_once_with(foundation.trace)
        qcc_readout.assert_called_once_with(foundation.trace)
        self.assertEqual(result["q"].shape, (1, 1024, 1024))
        self.assertEqual(result["qcc"].shape, (1, 32, 32))
        self.assertEqual(result["seed_ids"][0][0], 512)  # largest label-0 region
        self.assertEqual(result["p"].shape, (1, 1024, 2))
        self.assertEqual(result["primary_labels"].shape, (1, 32, 32))
        objective = (result["p"][0, :, 1] * torch.linspace(0, 1, 1024)).sum()
        grad, = torch.autograd.grad(objective, scale)
        self.assertTrue(torch.isfinite(grad))
        self.assertGreater(abs(float(grad.detach())), 1e-8)

    def test_tail_zero_and_sixty_four_are_passed_through_exactly(self):
        for tail in (0, 64):
            foundation = FakeFoundation()
            rgb = torch.zeros((1, 128, 128, 3), dtype=torch.uint8)
            q = torch.full((1, 1024, 1024), 0.3)
            labels = torch.zeros((1, 32, 32), dtype=torch.int64)
            labels.reshape(-1)[512:] = 1
            with (mock.patch.object(integration.sw145, "_actual_q", return_value=q),
                  mock.patch.object(integration.sw136_eval, "_qcc_labels", return_value=labels)):
                result = integration.forward_assignment(
                    foundation, rgb, live_tail_steps=tail)
            self.assertEqual(foundation.rollouts[0][1:], (1024, tail))
            self.assertEqual(result["live_tail_steps"], tail)

    def test_decoder_gamma_view_transposes_feature_and_node_axes_with_live_credit(self):
        foundation = FakeFoundation()
        gamma = torch.arange(8 * 1024, dtype=torch.float32).reshape(1, 8, 1024)
        gamma.requires_grad_()
        foundation.encode = lambda _rgb: gamma
        rgb = torch.zeros((1, 128, 128, 3), dtype=torch.uint8)
        q = torch.full((1, 1024, 1024), 0.3)
        labels = torch.zeros((1, 32, 32), dtype=torch.int64)
        labels.reshape(-1)[512:] = 1
        with (mock.patch.object(integration.sw145, "_actual_q", return_value=q),
              mock.patch.object(integration.sw136_eval, "_qcc_labels", return_value=labels)):
            result = integration.forward_assignment(foundation, rgb)

        self.assertEqual(result["gamma"].shape, (1, 8, 1024))
        self.assertEqual(result["gamma_content"].shape, (1, 1024, 8))
        self.assertTrue(torch.equal(result["gamma_content"], gamma.transpose(1, 2)))
        weights = torch.arange(1024, dtype=torch.float32).reshape(1, 1024, 1)
        grad, = torch.autograd.grad((result["gamma_content"] * weights).sum(), gamma)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertTrue(torch.equal(grad, weights.expand_as(result["gamma_content"]).transpose(1, 2)))

    def test_rejects_unregistered_tail_or_input_before_rollout(self):
        for tail in (1, 512, True, -1):
            foundation = FakeFoundation()
            rgb = torch.zeros((1, 128, 128, 3), dtype=torch.uint8)
            with self.assertRaises(ValueError):
                integration.forward_assignment(foundation, rgb, live_tail_steps=tail)
            self.assertEqual(foundation.rollouts, [])

        foundation = FakeFoundation()
        with self.assertRaises(ValueError):
            integration.forward_assignment(
                foundation, torch.zeros((1, 3, 128, 128), dtype=torch.uint8))
        self.assertEqual(foundation.rollouts, [])

    def test_invalid_helper_shapes_fail_closed(self):
        foundation = FakeFoundation()
        rgb = torch.zeros((1, 128, 128, 3), dtype=torch.uint8)
        with mock.patch.object(integration.sw145, "_actual_q", return_value=torch.zeros(1, 4, 4)):
            with self.assertRaises(ValueError):
                integration.forward_assignment(foundation, rgb)
        self.assertEqual(len(foundation.rollouts), 1)


if __name__ == "__main__":
    unittest.main()
