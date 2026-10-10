"""Focused tests for SW0133's genuine soft-assignment RGB path."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "snn_kuramoto_bidirectional"),
                str(ROOT / "collaborative_test")]
import numpy as np  # keep the registered environment's NumPy-before-Torch import order
import torch

from collaborative_test.SW_0132_partition_relative_rgb.relative_rgb import RelativeRGBDecoder
from collaborative_test.SW_0133_soft_partition_rgb.soft_partition import (
    reconstruct_soft, soft_weights,
)


def fixture(seed=130):
    torch.manual_seed(seed)
    labels = torch.arange(256) % 3
    hard = torch.nn.functional.one_hot(labels, num_classes=3).to(torch.float32)
    q = torch.rand(256, 256)
    q = (q + q.T) * 0.5
    q.fill_diagonal_(1.0)
    gamma = torch.randn(256, 8)
    target = torch.rand(128, 128, 3)
    return q, hard, gamma, target


class SoftPartitionTests(unittest.TestCase):
    def test_soft_p_matches_registered_row_dependent_count_formula(self):
        q, hard, _, _ = fixture()
        # Make unequal slot sizes so a global-count denominator cannot pass.
        hard = torch.nn.functional.one_hot(
            torch.tensor([0] * 120 + [1] * 80 + [2] * 56), num_classes=3
        ).to(torch.float32)
        weights, p = soft_weights(q, hard)
        q0 = q * (1.0 - torch.eye(256, dtype=q.dtype))
        count = hard.sum(dim=0)
        affinity = (q0 @ hard) / (count.unsqueeze(0) - hard).clamp_min(1.0)
        expected = torch.softmax(affinity / 0.10, dim=-1)
        self.assertTrue(torch.equal(p, expected))
        self.assertTrue(torch.equal(weights, p))
        self.assertTrue(torch.allclose(p.sum(dim=-1), torch.ones(256), atol=2e-7, rtol=0))

    def test_live_soft_path_backpropagates_through_assignment(self):
        q, hard, gamma, target = fixture(seed=131)
        q.requires_grad_(True)
        gamma.requires_grad_(True)
        decoder = RelativeRGBDecoder()
        prediction, loss, details = reconstruct_soft(
            q, hard, gamma, target, decoder, assignment_live=True,
            chunk_size=16384, checkpoint_chunks=False,
        )
        loss.backward()
        self.assertEqual(tuple(prediction.shape), (128, 128, 3))
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(q.grad)
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertGreater(float(q.grad.abs().sum()), 0.0)
        self.assertIsNone(gamma.grad)
        self.assertTrue(torch.equal(details["W"], details["P"]))

    def test_detached_control_has_same_forward_but_no_q_gradient(self):
        q, hard, gamma, target = fixture(seed=132)
        qa = q.clone().requires_grad_(True)
        qb = q.clone().requires_grad_(True)
        gamma.requires_grad_(True)
        decoder = RelativeRGBDecoder()
        pred_live, loss_live, _ = reconstruct_soft(
            qa, hard, gamma, target, decoder, assignment_live=True,
            chunk_size=16384, checkpoint_chunks=False,
        )
        pred_detached, loss_detached, details = reconstruct_soft(
            qb, hard, gamma, target, decoder, assignment_live=False,
            chunk_size=16384, checkpoint_chunks=False,
        )
        self.assertTrue(torch.equal(pred_live, pred_detached))
        self.assertTrue(torch.equal(loss_live, loss_detached))
        self.assertTrue(torch.equal(details["W"], details["P"].detach()))
        loss_live.backward()
        loss_detached.backward()
        self.assertIsNotNone(qa.grad)
        self.assertGreater(float(qa.grad.abs().sum()), 0.0)
        self.assertIsNone(qb.grad)

    def test_soft_rgb_directional_q_gradient_matches_fixed_h_finite_difference(self):
        q, hard, gamma, target = fixture(seed=133)
        q, hard, gamma, target = (value.to(torch.float64) for value in (q, hard, gamma, target))
        q.requires_grad_(True)
        decoder = RelativeRGBDecoder().double()
        pred, loss, _ = reconstruct_soft(
            q, hard, gamma, target, decoder, assignment_live=True,
            chunk_size=16384, checkpoint_chunks=False,
        )
        gradient, = torch.autograd.grad(loss, q)
        direction = gradient.detach() / gradient.detach().norm()
        analytic = (gradient * direction).sum()
        step = 1e-5
        with torch.no_grad():
            plus, plus_loss, _ = reconstruct_soft(
                q + step * direction, hard, gamma, target, decoder,
                assignment_live=True, chunk_size=16384, checkpoint_chunks=False,
            )
            minus, minus_loss, _ = reconstruct_soft(
                q - step * direction, hard, gamma, target, decoder,
                assignment_live=True, chunk_size=16384, checkpoint_chunks=False,
            )
        finite_difference = (plus_loss - minus_loss) / (2 * step)
        self.assertTrue(torch.isfinite(pred).all())
        self.assertTrue(torch.isfinite(analytic))
        self.assertGreater(abs(float(analytic)), 1e-9)
        self.assertGreater(abs(float(finite_difference)), 1e-9)
        self.assertAlmostEqual(float(analytic), float(finite_difference), delta=max(
            1e-9, abs(float(analytic)) * 2e-3))


if __name__ == "__main__":
    unittest.main()
