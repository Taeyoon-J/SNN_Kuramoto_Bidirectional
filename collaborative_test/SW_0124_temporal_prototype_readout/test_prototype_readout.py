"""Numerical and invariance checks for the GT-free temporal readout."""
import numpy as np  # Keep NumPy before Torch on the Windows development host.
import unittest

import torch

from collaborative_test.SW_0124_temporal_prototype_readout.prototype_readout import (
    temporal_prototype_assignment,
)
from collaborative_test.SW_0123_adaptive_temporal_assignment.readout import assignment_to_labels


def traces(batch=1, horizon=32, requires_grad=False):
    torch.manual_seed(244)
    # Multiple nonidentical binary temporal patterns across channels and patches.
    values = (torch.rand(batch, 4, 256, horizon) > .78).float()
    return values.requires_grad_(requires_grad)


def affinity(batch=1):
    torch.manual_seed(245)
    q = torch.rand(batch, 256, 256)
    q = (q + q.transpose(1, 2)) * .5
    q.diagonal(dim1=1, dim2=2).fill_(1.0)
    return q


class TemporalPrototypeTests(unittest.TestCase):
    def test_variable_temporal_horizons_and_static_gamma_contract(self):
        # Gamma is deliberately not accepted here: this readout consumes actual
        # time traces, unlike the core whose input gamma is static [B,8,256].
        for horizon, settle in ((32, 0), (512, 128)):
            x = traces(horizon=horizon)
            p, anchors, centers = temporal_prototype_assignment(
                x, torch.ones(4), affinity(), settle=settle)
            self.assertEqual(tuple(p.shape), (1, 256, 11))
            self.assertTrue(torch.isfinite(p).all())
            self.assertEqual(len(anchors), 1)
            self.assertLessEqual(len(anchors[0]), 11)
            self.assertEqual(tuple(centers[0].shape[1:]), (4, horizon - settle))

    def test_shared_time_permutation_preserves_readout(self):
        x = traces(horizon=32)
        q = affinity()
        order = torch.randperm(32)
        first, first_anchors, _ = temporal_prototype_assignment(x, torch.ones(4), q, settle=0)
        permuted, permuted_anchors, _ = temporal_prototype_assignment(
            x[..., order], torch.ones(4), q, settle=0)
        self.assertEqual(first_anchors, permuted_anchors)
        self.assertTrue(torch.allclose(first, permuted, rtol=2e-5, atol=2e-6))

    def test_individual_patch_time_permutation_changes_temporal_assignment(self):
        x = traces(horizon=32)
        q = torch.eye(256).unsqueeze(0)
        original, anchors, _ = temporal_prototype_assignment(x, torch.ones(4), q, settle=0)
        patch = next(index for index in range(256) if index not in anchors[0])
        altered = x.clone()
        altered[:, :, patch] = altered[:, :, patch].roll(1, dims=-1)
        changed, _, _ = temporal_prototype_assignment(altered, torch.ones(4), q, settle=0)
        self.assertFalse(torch.allclose(original, changed, rtol=1e-5, atol=1e-7))

    def test_slot_column_permutation_preserves_partition(self):
        probability = torch.full((1, 256, 11), 0.01)
        winning = torch.arange(256) % 11
        probability[0, torch.arange(256), winning] = 0.90
        permutation = torch.randperm(11)
        first = assignment_to_labels(probability).reshape(1, -1)
        second = assignment_to_labels(probability[:, :, permutation]).reshape(1, -1)
        first_equal = first[:, :, None] == first[:, None, :]
        second_equal = second[:, :, None] == second[:, None, :]
        self.assertTrue(torch.equal(first_equal, second_equal))

    def test_assignment_has_live_gradient_to_actual_spikes(self):
        x = traces(horizon=32, requires_grad=True)
        p, _, _ = temporal_prototype_assignment(x, torch.ones(4), affinity(), settle=0)
        weights = torch.linspace(-1, 1, 11).view(1, 1, 11)
        (p * weights).sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad.abs().sum()), 0.0)

    def test_identical_traces_stop_at_one_prototype_and_read_out_background(self):
        base = (torch.arange(32) % 2).float().view(1, 1, 1, 32)
        x = base.expand(1, 4, 256, 32).clone()
        q = torch.ones(1, 256, 256)
        p, anchors, _ = temporal_prototype_assignment(x, torch.ones(4), q, settle=0)
        self.assertEqual(len(anchors[0]), 1)
        self.assertTrue(torch.equal(p[..., 0], torch.ones_like(p[..., 0])))
        self.assertTrue(torch.equal(p[..., 1:], torch.zeros_like(p[..., 1:])))
        self.assertEqual(torch.unique(assignment_to_labels(p)).tolist(), [0])


if __name__ == "__main__":
    unittest.main()
