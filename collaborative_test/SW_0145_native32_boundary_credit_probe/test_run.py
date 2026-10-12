from __future__ import annotations

import unittest

import numpy as np
import torch

from collaborative_test.SW_0145_native32_boundary_credit_probe import run


class BoundaryCreditProbeTests(unittest.TestCase):
    def test_rgb_seam_energy_locates_exact_native32_boundary(self):
        image = np.zeros((1, 128, 128, 3), dtype=np.uint8)
        image[:, :, 64:, :] = 255
        energy = run.boundary_energy(image)
        self.assertEqual(energy.shape, (1, 1984))
        expected = np.zeros(1984, dtype=np.float64)
        expected[np.arange(32) * 31 + 15] = 1.0
        np.testing.assert_array_equal(energy[0], expected)

    def test_registered_edge_target_loss_backpropagates_through_actual_q(self):
        gen = torch.Generator().manual_seed(145)
        shared_events = (torch.rand((2, 1, 1, 40), generator=gen) > 0.8).float()
        noise = torch.rand((2, 4, 12, 40), generator=gen) * 0.03
        spikes = (shared_events.expand(-1, 4, 12, -1) + noise).requires_grad_()
        from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity
        affinity = spike_synchrony_affinity(spikes.mean(dim=1), components=spikes,
                                            settle=10, affinity_mode="spike")
        pairs = torch.tensor([[0, 1], [1, 2], [3, 7], [8, 10]])
        targets = torch.tensor([[0.1, 0.4, 0.7, 0.9], [0.3, 0.5, 0.6, 0.8]])
        loss = run.boundary_q_loss(affinity, targets, pairs)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(spikes.grad)
        self.assertTrue(torch.isfinite(spikes.grad).all())
        self.assertGreater(float(spikes.grad.abs().sum()), 0.0)

    def test_source_q_preservation_detaches_teacher_and_keeps_student_credit(self):
        student = torch.rand((2, 7, 7), requires_grad=True)
        teacher = torch.rand((2, 7, 7), requires_grad=True)
        loss = run.source_q_preservation_loss(student, teacher)
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertGreater(float(student.grad.abs().sum()), 0.0)
        self.assertIsNone(teacher.grad)

    def test_weighted_microbatch_gradient_accumulation_matches_logical_batch(self):
        weight = torch.tensor(0.7, requires_grad=True)
        x = torch.tensor([0.2, 0.7, 1.1, -0.4, 0.6, 1.3, -0.8, 0.5])
        target = torch.tensor([0.1, 0.3, 0.9, 0.4, 0.6, -0.2, 0.5, 0.1])
        full = ((weight * x - target) ** 2).mean()
        expected = torch.autograd.grad(full, weight)[0]
        accumulated = [None]
        batch_size = 2
        for start in range(0, len(x), batch_size):
            loss = ((weight * x[start:start + batch_size] - target[start:start + batch_size]) ** 2).mean()
            grad = torch.autograd.grad(loss, weight)[0]
            run._accumulate_weighted_gradients(accumulated, (grad,), batch_size / len(x))
        torch.testing.assert_close(accumulated[0], expected, rtol=1e-6, atol=1e-7)

    def test_scale_requires_exact_registered_unique_row_count(self):
        fake = np.zeros((4096, 128, 128, 3), dtype=np.uint8)
        rows = np.arange(4095, dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "exactly4096"):
            run.boundary_scale_from_train(fake, rows)

    def test_source_foundation_must_carry_verified_rgb_asset_record(self):
        record = {"train_cache_sha256": "a" * 64, "val_cache_sha256": "b" * 64}
        self.assertEqual(run.require_validated_asset_record({"rgb_asset_validation": record}), record)
        with self.assertRaisesRegex(ValueError, "not loaded with verified RGB assets"):
            run.require_validated_asset_record({"rgb_asset_validation": None})

    def test_rejects_nonfinite_affinity(self):
        q = torch.zeros((1, 4, 4)); q[0, 0, 1] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "nonfinite"):
            run.boundary_q_loss(q, np.zeros((1, 2)), np.asarray([[0, 1], [1, 2]]))


if __name__ == "__main__":
    unittest.main()
