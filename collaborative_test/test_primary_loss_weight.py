import math
import unittest

import torch

from snn_kuramoto_bidirectional.training.train_s2net_core import (
    _scale_primary_loss,
    _scale_primary_loss_parts,
    _validate_loss_weight,
)


class PrimaryLossWeightTests(unittest.TestCase):
    def test_zero_primary_weight_removes_base_gradient_but_keeps_auxiliary(self):
        base_parameter = torch.tensor(2.0, requires_grad=True)
        spike_parameter = torch.tensor(3.0, requires_grad=True)
        base = base_parameter.square()
        spike_auxiliary = spike_parameter.square()

        actual_total = _scale_primary_loss(base, 0.0) + 5.0 * spike_auxiliary
        actual_total.backward()

        self.assertEqual(actual_total.item(), 45.0)
        self.assertEqual(base_parameter.grad.item(), 0.0)
        self.assertEqual(spike_parameter.grad.item(), 30.0)

    def test_default_weight_preserves_base_objective_and_parts_total(self):
        raw_total = torch.tensor(7.25, requires_grad=True)
        parts = {
            "base_term": torch.tensor(2.5),
            "total": raw_total,
        }
        logged = _scale_primary_loss_parts(parts, 1.0)
        self.assertEqual(_scale_primary_loss(raw_total, 1.0).item(), 7.25)
        self.assertEqual(logged["base_term"].item(), 2.5)
        self.assertEqual(logged["total"].item(), 7.25)
        self.assertEqual(logged["primary_unscaled_total"].item(), 7.25)

    def test_weight_validation_rejects_negative_and_nonfinite(self):
        self.assertEqual(_validate_loss_weight(0, "weight"), 0.0)
        self.assertEqual(_validate_loss_weight(1, "weight"), 1.0)
        for value in (-0.1, math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _validate_loss_weight(value, "weight")


if __name__ == "__main__":
    unittest.main()
