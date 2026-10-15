"""Gold-partition tie semantics for the SW0167 structured objective."""
import numpy as np
import torch
import unittest

from collaborative_test.SW_0167_structured_relation_energy.training_objective import (
    structured_hinge_gold_ties,
)


class GoldTieTests(unittest.TestCase):
    def test_exact_augmented_score_tie_selects_gold_with_live_zero_gradient(self):
        # For two adjacent foreground nodes with one shared target, e_theta=1
        # gives augmented merge energy exactly zero. The greedy oracle merges
        # only strictly positive gains, so G_hat differs from G* at an exact
        # structured-score tie. The registered tie rule must choose G*.
        e0 = torch.zeros((2, 2), dtype=torch.float64)
        e_theta = torch.tensor([[0.0, 1.0], [1.0, 0.0]],
                               dtype=torch.float64, requires_grad=True)
        target = torch.tensor([[1, 1]])
        support = torch.tensor([[True, True]])
        loss = structured_hinge_gold_ties(e_theta, e0, target, support)
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertIsNotNone(e_theta.grad)
        self.assertTrue(torch.equal(e_theta.grad, torch.zeros_like(e_theta)))

    def test_positive_hinge_keeps_candidate_gradient(self):
        # A sufficiently unfavorable candidate edge leaves the predicted
        # partition split; this is not a tie and must retain hinge credit.
        e0 = torch.zeros((2, 2), dtype=torch.float64)
        e_theta = torch.tensor([[0.0, -1.0], [-1.0, 0.0]],
                               dtype=torch.float64, requires_grad=True)
        loss = structured_hinge_gold_ties(
            e_theta, e0, torch.tensor([[1, 1]]), torch.tensor([[True, True]])
        )
        self.assertEqual(float(loss.detach()), 2.0)
        loss.backward()
        self.assertEqual(float(e_theta.grad[0, 1]), -1.0)

    def test_malformed_energy_and_target_geometry_rejected(self):
        e = torch.zeros((4, 4))
        with self.assertRaises(ValueError):
            structured_hinge_gold_ties(e, e, np.zeros((1, 3), dtype=np.int32),
                                       np.ones((1, 3), dtype=bool))
        bad = e.clone(); bad[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            structured_hinge_gold_ties(bad, e, np.ones((2, 2), dtype=np.int32),
                                       np.ones((2, 2), dtype=bool))


if __name__ == "__main__":
    unittest.main()
