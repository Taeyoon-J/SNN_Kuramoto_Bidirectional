from __future__ import annotations

import unittest

import torch

from collaborative_test.SW_0150_spike_spatial_mixture.spatial_diffusion import (
    NODES, spatial_diffusion,
)


def varied_symmetric_q(batch=1):
    torch.manual_seed(1500)
    raw = torch.rand(batch, NODES, NODES, dtype=torch.float32)
    q = (raw + raw.transpose(1, 2)) * 0.5
    q.diagonal(dim1=1, dim2=2).zero_()
    return q


class SpatialDiffusionTests(unittest.TestCase):
    def test_assignments_are_row_stochastic_and_fixed_seed_rows_are_one_hot(self):
        q = varied_symmetric_q()
        seeds = [[0, 127, 900]]
        p = spatial_diffusion(q, seeds)
        self.assertEqual(tuple(p.shape), (1, 1024, 3))
        self.assertTrue(torch.isfinite(p).all())
        self.assertTrue(torch.all(p >= 0))
        self.assertTrue(torch.allclose(p.sum(-1), torch.ones((1, 1024)), atol=1e-6, rtol=0))
        self.assertTrue(torch.equal(p[0, 0], torch.tensor([1.0, 0.0, 0.0])))
        self.assertTrue(torch.equal(p[0, 127], torch.tensor([0.0, 1.0, 0.0])))
        self.assertTrue(torch.equal(p[0, 900], torch.tensor([0.0, 0.0, 1.0])))

    def test_k1_is_exact_background_and_variable_batch_seed_counts_pad_with_zero(self):
        q = varied_symmetric_q(batch=2).requires_grad_()
        p = spatial_diffusion(q, [[81], [4, 55]])
        self.assertEqual(tuple(p.shape), (2, 1024, 2))
        self.assertTrue(torch.equal(p[0], torch.ones((1024, 2)) * torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.equal(p[1, 4], torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.equal(p[1, 55], torch.tensor([0.0, 1.0])))
        self.assertTrue(torch.allclose(p.sum(-1), torch.ones((2, 1024)), atol=1e-6, rtol=0))
        (p[0, 400, 1] + p[1, 400, 1]).backward()
        self.assertTrue(torch.equal(q.grad[0], torch.zeros_like(q.grad[0])))
        self.assertGreater(float(q.grad[1].abs().max().detach()), 0.0)

    def test_diagonal_values_do_not_affect_diffusion_or_receive_gradient(self):
        q0 = varied_symmetric_q().requires_grad_()
        q1 = q0.detach().clone()
        q1.diagonal(dim1=1, dim2=2).fill_(1.0)
        p0 = spatial_diffusion(q0, [[0, 511, 1023]])
        p1 = spatial_diffusion(q1, [[0, 511, 1023]])
        self.assertTrue(torch.equal(p0, p1))
        (p0[:, 200:800, 1] * torch.linspace(0.1, 0.9, 600)[None]).sum().backward()
        self.assertIsNotNone(q0.grad)
        self.assertTrue(torch.equal(q0.grad.diagonal(dim1=1, dim2=2), torch.zeros((1, NODES))))

    def test_offdiagonal_actual_q_receives_finite_nonzero_gradient(self):
        q = varied_symmetric_q().requires_grad_()
        p = spatial_diffusion(q, [[0, 17, 1000]])
        probe = torch.linspace(-1, 1, NODES, dtype=q.dtype).reshape(1, NODES)
        loss = (p[:, :, 1] * probe).sum() + 0.37 * (p[:, :, 2] ** 2).sum()
        loss.backward()
        grad = q.grad
        offdiagonal = grad.clone()
        offdiagonal.diagonal(dim1=1, dim2=2).zero_()
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(offdiagonal.abs().max().detach()), 0.0)
        self.assertTrue(torch.equal(grad.diagonal(dim1=1, dim2=2), torch.zeros((1, NODES))))

    def test_only_four_neighbor_edges_receive_q_gradient_no_wraparound(self):
        q = torch.zeros((1, NODES, NODES), dtype=torch.float32)
        for left, right, value in ((0, 1, 0.70), (1, 2, 0.25), (0, 31, 0.80),
                                   (31, 32, 0.70), (0, 1023, 0.60)):
            q[0, left, right] = value
            q[0, right, left] = value
        q.requires_grad_()
        p = spatial_diffusion(q, [[0, 2]])
        p[0, 1, 1].backward()
        grad = q.grad
        self.assertGreater(float(grad[0, 0, 1].abs().detach()
                                + grad[0, 1, 0].abs().detach()), 0.0)
        for left, right in ((0, 31), (31, 32), (0, 1023)):
            self.assertEqual(float(grad[0, left, right].detach()), 0.0)
            self.assertEqual(float(grad[0, right, left].detach()), 0.0)

    def test_rejects_wrong_grid_q_or_invalid_seed_contracts(self):
        q = varied_symmetric_q()
        with self.assertRaisesRegex(ValueError, "32x32"):
            spatial_diffusion(q[:, :16, :16], [[0]])
        asymmetric = q.clone()
        asymmetric[0, 0, 1] = 0.25
        asymmetric[0, 1, 0] = 0.75
        with self.assertRaisesRegex(ValueError, "symmetric"):
            spatial_diffusion(asymmetric, [[0]])
        with self.assertRaisesRegex(ValueError, "unique"):
            spatial_diffusion(q, [[0, 0]])
        with self.assertRaisesRegex(ValueError, "outside"):
            spatial_diffusion(q, [[0, 1024]])
        with self.assertRaisesRegex(ValueError, "one ordered seed list"):
            spatial_diffusion(q, [])
        with self.assertRaisesRegex(ValueError, "exactly 64"):
            spatial_diffusion(q, [[0]], steps=63)

    def test_rejects_nonfinite_negative_and_out_of_range_actual_q(self):
        q = varied_symmetric_q()
        nonfinite = q.clone()
        nonfinite[0, 0, 1] = float("nan")
        nonfinite[0, 1, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            spatial_diffusion(nonfinite, [[0]])
        for value in (-0.1, 1.1):
            bad = q.clone()
            bad[0, 0, 1] = value
            bad[0, 1, 0] = value
            with self.assertRaisesRegex(ValueError, r"in \[0,1\]"):
                spatial_diffusion(bad, [[0]])


if __name__ == "__main__":
    unittest.main()
