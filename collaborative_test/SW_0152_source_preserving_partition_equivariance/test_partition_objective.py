import sys
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from collaborative_test.SW_0152_source_preserving_partition_equivariance.partition_objective import (
    alignment_js, horizontal_flip_indices, partition_objective,
)


def flip_view(p):
    return p.reshape(p.shape[0], 32, 32, p.shape[-1]).flip(2).reshape_as(p)


class PartitionObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(150202)

    def test_flip_seed_map_is_bijection(self):
        ids = torch.arange(1024)
        mapped = horizontal_flip_indices(ids)
        self.assertTrue(torch.equal(horizontal_flip_indices(mapped), ids))
        self.assertEqual(mapped.unique().numel(), 1024)
        self.assertEqual(int(horizontal_flip_indices(torch.tensor([0]))[0]), 31)
        with self.assertRaises(ValueError):
            horizontal_flip_indices(torch.tensor([1024]))

    def test_identical_corresponding_partitions_have_zero_js(self):
        p = torch.softmax(torch.randn(2, 1024, 4), dim=-1)
        self.assertLess(float(alignment_js(p, flip_view(p))), 1e-7)
        perm = torch.tensor([2, 0, 3, 1])
        self.assertLess(float(alignment_js(p[..., perm], flip_view(p[..., perm]))), 1e-7)

    def test_within_color_assignment_permutation_increases_alignment_loss(self):
        # One constant-color image with a spatial two-way assignment.
        # Permuting rows remains entirely within the same RGB-color bin.
        labels = (torch.arange(1024).reshape(1, 1024) // 16) % 2
        p = torch.nn.functional.one_hot(labels.reshape(1, 1024), 2).float() * 0.98 + 0.01
        p = p / p.sum(-1, keepdim=True)
        exact = flip_view(p)
        permuted_unflipped = p.clone()
        # Every patch has the same color, so this cyclic row reassignment is a
        # within-color permutation and changes only spatial correspondence.
        permuted_unflipped = p.roll(8, dims=1)
        bad = flip_view(permuted_unflipped)
        self.assertGreater(float(alignment_js(p, bad)), float(alignment_js(p, exact)) + 0.1)

    def test_variable_slots_padding_k1_and_control(self):
        source = torch.zeros(2, 1024, 3)
        source[..., 0] = 1.0
        source[1, :, 0] = 0.5
        source[1, :, 1:] = 0.25
        one = torch.ones(1, 1024, 1)
        one_source = torch.ones(1, 1024, 1)
        result = partition_objective(one, one, one_source)
        self.assertEqual(float(result["total"]), 0.0)
        p = torch.softmax(torch.randn(2, 1024, 3), dim=-1).requires_grad_()
        q = torch.softmax(torch.randn(2, 1024, 3), dim=-1).requires_grad_()
        out = partition_objective(p, flip_view(q), source, mode="candidate")
        ctl = partition_objective(p, flip_view(q), source, mode="control")
        self.assertTrue(torch.allclose(out["total"], out["alignment_js"] + out["mass_kl"]))
        self.assertTrue(torch.allclose(ctl["total"], ctl["mass_kl"]))
        out["total"].backward()
        self.assertTrue(torch.isfinite(p.grad).all() and torch.isfinite(q.grad).all())
        self.assertGreater(float(p.grad.abs().sum()), 0)
        self.assertGreater(float(q.grad.abs().sum()), 0)

    def test_sorted_mass_is_slot_permutation_invariant(self):
        source = torch.softmax(torch.randn(2, 1024, 4), dim=-1)
        p = torch.softmax(torch.randn(2, 1024, 4), dim=-1)
        q = flip_view(torch.softmax(torch.randn(2, 1024, 4), dim=-1))
        expected = partition_objective(p, q, source, mode="control")["mass_kl"]
        # In the marginal-only objective, source and current slot names are irrelevant.
        got = partition_objective(p[..., [2, 0, 3, 1]], q[..., [1, 3, 0, 2]],
                                  source[..., [3, 1, 0, 2]], mode="control")["mass_kl"]
        self.assertTrue(torch.allclose(expected, got, atol=1e-7, rtol=1e-6))
        # The frozen area anchor may also be cached as a mass-only vector.
        source_mass = source.mean(dim=1)
        mass_only = partition_objective(p, q, source_mass, mode="control")["mass_kl"]
        self.assertTrue(torch.allclose(expected, mass_only, atol=1e-7, rtol=1e-6))

    def test_invalid_probability_contract_rejected(self):
        good = torch.full((1, 1024, 2), 0.5)
        bad = good.clone()
        bad[0, 0, 0] = -0.1
        with self.assertRaises(ValueError):
            alignment_js(good, bad)
        bad = good.clone()
        bad[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            alignment_js(good, bad)
        bad = good.clone()
        bad[0, 0, 0] = 0.7
        with self.assertRaises(ValueError):
            alignment_js(good, bad)


if __name__ == "__main__":
    unittest.main()
