"""Loss/gradient checks for sample diversity on sigmoid membrane activity."""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from snn_kuramoto_bidirectional.loss_function import (
    UnsupervisedS2NetLoss,
    sample_activity_diversity_loss,
)
from snn_kuramoto_bidirectional.training.train_s2net_core import _select_loss_signal


class SampleDiversityLossTest(unittest.TestCase):
    def test_collapsed_samples_are_penalized_more_than_distinct_activity(self):
        torch.manual_seed(9)
        one = torch.rand(1, 8, 12)
        collapsed = one.repeat(4, 1, 1)
        distinct = torch.rand(4, 8, 12)
        collapsed_loss = sample_activity_diversity_loss(collapsed)
        distinct_loss = sample_activity_diversity_loss(distinct)
        self.assertGreater(float(collapsed_loss), float(distinct_loss))

    def test_weighted_term_has_finite_nonzero_gradient_through_sigmoid_membrane(self):
        torch.manual_seed(21)
        membrane = torch.randn(4, 8, 12, requires_grad=True)
        activity = _select_loss_signal(
            spikes=torch.zeros_like(membrane), core_out=membrane,
            loss_signal="sigmoid_membrane")
        criterion = UnsupervisedS2NetLoss(
            spike_rate_weight=0.0, spike_smooth_weight=0.0,
            spike_diversity_weight=0.0, structural_weight=0.0,
            sample_diversity_weight=0.05)
        total, parts = criterion(spikes=activity)
        raw = sample_activity_diversity_loss(activity)
        torch.testing.assert_close(parts["sample_diversity"], raw)
        torch.testing.assert_close(total, 0.05 * raw)
        total.backward()
        self.assertTrue(torch.isfinite(membrane.grad).all())
        self.assertGreater(float(membrane.grad.norm()), 0.0)

    def test_single_sample_returns_zero(self):
        value = torch.randn(1, 8, 12, requires_grad=True)
        loss = sample_activity_diversity_loss(value)
        self.assertEqual(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
