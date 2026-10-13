from __future__ import annotations

import unittest

import torch

from collaborative_test.SW_0149_spike_graph_modularity.modularity import (
    degree_corrected_modularity_loss, source_adjacency, validate_modularity_inputs,
)


class ModularityKernelTests(unittest.TestCase):
    def test_source_graph_is_symmetrized_detached_and_diagonal_cleared(self):
        raw = torch.tensor([[[7.0, 2.0, 0.0], [4.0, 9.0, 1.0], [0.0, 3.0, 8.0]]],
                           requires_grad=True)
        original = raw.detach().clone()
        a = source_adjacency(raw)
        self.assertTrue(torch.equal(a, torch.tensor([[[0.0, 3.0, 0.0],
                                                       [3.0, 0.0, 2.0],
                                                       [0.0, 2.0, 0.0]]])))
        self.assertFalse(a.requires_grad)
        self.assertTrue(torch.equal(raw.detach(), original))

    def test_all_ones_q_has_zero_modularity_and_finite_gradient(self):
        a = torch.tensor([[[0.0, 1.0, 0.5, 0.0],
                           [1.0, 0.0, 0.0, 0.5],
                           [0.5, 0.0, 0.0, 1.0],
                           [0.0, 0.5, 1.0, 0.0]]], dtype=torch.float32)
        q = torch.ones_like(a, requires_grad=True)
        loss = degree_corrected_modularity_loss(a, q)
        self.assertAlmostEqual(float(loss.detach()), 0.0, places=7)
        grad, = torch.autograd.grad(loss, q)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_degree_correction_matches_direct_registered_formula(self):
        a = torch.tensor([[[0.0, 2.0, 0.2, 0.0],
                           [2.0, 0.0, 0.0, 0.1],
                           [0.2, 0.0, 0.0, 3.0],
                           [0.0, 0.1, 3.0, 0.0]]], dtype=torch.float64)
        q = torch.tensor([[[1.0, .8, .2, .1], [.8, 1.0, .2, .1],
                           [.2, .2, 1.0, .9], [.1, .1, .9, 1.0]]], dtype=torch.float64,
                         requires_grad=True)
        degree = a.sum(-1)
        z = degree.sum(-1)
        m = a - degree.unsqueeze(-1) * degree.unsqueeze(-2) / z[:, None, None]
        expected = -(m * q).sum((-2, -1)).div(z).mean()
        actual = degree_corrected_modularity_loss(a, q)
        self.assertTrue(torch.equal(actual, expected))
        grad, = torch.autograd.grad(actual, q)
        self.assertTrue(torch.isfinite(grad).all())

    def test_modularity_favors_two_source_communities_and_is_scale_invariant(self):
        a = torch.tensor([[[0.0, 1.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0],
                           [0.0, 0.0, 1.0, 0.0]]], dtype=torch.float32)
        within = torch.tensor([[[1.0, .9, .9, .1], [.9, 1.0, .1, .9],
                                [.9, .1, 1.0, .9], [.1, .9, .9, 1.0]]], dtype=torch.float32,
                              requires_grad=True)
        cross = torch.tensor([[[1.0, .1, .1, .9], [.1, 1.0, .9, .1],
                               [.1, .9, 1.0, .1], [.9, .1, .1, 1.0]]], dtype=torch.float32,
                             requires_grad=True)
        all_ones = torch.ones_like(within, requires_grad=True)
        good = degree_corrected_modularity_loss(a, within)
        bad = degree_corrected_modularity_loss(a, cross)
        zero = degree_corrected_modularity_loss(a, all_ones)
        self.assertLess(float(good.detach()), float(zero.detach()))
        self.assertLess(float(zero.detach()), float(bad.detach()))
        scaled = degree_corrected_modularity_loss(a * 7.0, within.detach().requires_grad_())
        self.assertTrue(torch.allclose(good.detach(), scaled.detach(), rtol=1e-6, atol=1e-7))

    def test_batched_production_shape_is_accepted(self):
        n = 1024
        eye = torch.eye(n, dtype=torch.float32)
        a = (torch.ones((2, n, n), dtype=torch.float32) - eye) / (n - 1)
        q = torch.ones_like(a, requires_grad=True)
        z = validate_modularity_inputs(a, q)
        self.assertEqual(tuple(z.shape), (2,))
        self.assertTrue(torch.allclose(z, torch.full_like(z, float(n)), atol=1e-3))
        self.assertLess(abs(float(degree_corrected_modularity_loss(a, q).detach())), 1e-7)

    def test_invalid_graph_and_q_values_are_rejected(self):
        base = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], dtype=torch.float32)
        diagonal_a = base.clone()
        diagonal_a[0, 0, 0] = 1.0
        for bad_a, bad_q, pattern in (
            (diagonal_a, torch.ones_like(base), "diagonal"),
            (torch.tensor([[[0.0, -1.0], [-1.0, 0.0]]]), torch.ones_like(base), "nonnegative"),
            (torch.zeros_like(base), torch.ones_like(base), "positive finite total degree"),
            (base, torch.tensor([[[1.0, 2.0], [2.0, 1.0]]]), "lie in \\[0,1\\]"),
            (base, torch.tensor([[[1.0, float("nan")], [float("nan"), 1.0]]]), "nonfinite"),
        ):
            with self.subTest(pattern=pattern), self.assertRaisesRegex(ValueError, pattern):
                degree_corrected_modularity_loss(bad_a, bad_q.requires_grad_())

    def test_nonfinite_or_negative_source_graph_is_rejected_before_preparation(self):
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            source_adjacency(torch.tensor([[[0.0, float("inf")], [1.0, 0.0]]]))
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            source_adjacency(torch.tensor([[[0.0, -1.0], [-1.0, 0.0]]]))
        with self.assertRaisesRegex(ValueError, "raw source adjacency"):
            source_adjacency(torch.tensor([[[-1.0, 1.0], [0.0, 0.0]]]))

    def test_teacher_must_be_detached_and_q_diagonal_is_fixed_without_credit(self):
        a = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "teacher must be detached"):
            degree_corrected_modularity_loss(a.requires_grad_(), torch.ones_like(a, requires_grad=True))
        q_native = torch.tensor([[[0.0, .6], [.6, 0.0]]], requires_grad=True)
        q_one = torch.tensor([[[1.0, .6], [.6, 1.0]]], requires_grad=True)
        loss_native = degree_corrected_modularity_loss(a.detach(), q_native)
        loss_one = degree_corrected_modularity_loss(a.detach(), q_one)
        self.assertTrue(torch.equal(loss_native, loss_one))
        grad, = torch.autograd.grad(loss_native, q_native)
        self.assertTrue(torch.equal(torch.diagonal(grad, dim1=-2, dim2=-1), torch.zeros((1, 2))))


if __name__ == "__main__":
    unittest.main()
