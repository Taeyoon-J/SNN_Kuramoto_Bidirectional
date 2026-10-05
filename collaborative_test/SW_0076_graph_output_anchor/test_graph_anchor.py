import sys
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from collaborative_test.SW_0068_joint_feature_core.train_joint import (
    compute_graph_output_anchor_loss,
    validate_graph_output_anchor,
)


class ToyGraph(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 4, bias=False)

    def forward(self, gamma):
        return torch.softmax(self.projection(gamma), dim=-1)


class ToyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.graph_generator = ToyGraph()


class GraphAnchorTests(unittest.TestCase):
    def test_identical_gamma_has_zero_loss(self):
        core = ToyCore()
        gamma = torch.randn(2, 5, 3)
        loss = compute_graph_output_anchor_loss(core, gamma, gamma)
        self.assertEqual(float(loss), 0.0)

    def test_candidate_gradient_finite_and_nonzero(self):
        core = ToyCore()
        core.requires_grad_(False)
        anchor = torch.randn(2, 5, 3)
        current = (anchor + 0.2 * torch.randn_like(anchor)).requires_grad_()
        loss = compute_graph_output_anchor_loss(core, current, anchor)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(current.grad).all())
        self.assertGreater(float(current.grad.norm()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in core.parameters()))

    def test_requires_graph_generator(self):
        with self.assertRaisesRegex(ValueError, "learned graph"):
            compute_graph_output_anchor_loss(nn.Identity(), torch.ones(1), torch.ones(1))

    def test_nonzero_weight_requires_frozen_core_and_finite_nonnegative(self):
        validate_graph_output_anchor(0.0, False)
        validate_graph_output_anchor(10.0, True)
        for weight in (-1.0, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                validate_graph_output_anchor(weight, True)
        with self.assertRaisesRegex(ValueError, "frozen core"):
            validate_graph_output_anchor(1.0, False)


if __name__ == "__main__":
    unittest.main()
