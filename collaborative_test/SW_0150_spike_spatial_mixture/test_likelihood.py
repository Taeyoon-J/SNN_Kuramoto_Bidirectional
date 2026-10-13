from __future__ import annotations

import unittest
from unittest import mock

import torch

from collaborative_test.SW_0150_spike_spatial_mixture.likelihood import (
    SpatialMixtureDecoder, calibrate_sigma2, gaussian_mixture_nll, make_decoder,
)
from collaborative_test.SW_0150_spike_spatial_mixture.spatial_diffusion import (
    NODES, spatial_diffusion,
)


class SpatialLikelihoodTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1501)
        self.gamma = torch.randn(1, NODES, 4, dtype=torch.float32)
        self.rgb = torch.randint(0, 256, (1, 128, 128, 3), dtype=torch.uint8)

    def test_k1_matches_direct_three_channel_gaussian_nll(self):
        p = torch.ones((1, NODES, 1), dtype=torch.float32)
        decoder = make_decoder(4, seed=11)
        sigma2 = 0.037
        means = decoder.component_means(self.gamma, p)[..., 0, :]
        target = self.rgb.float() / 255.0
        direct = 0.5 * ((target - means).square().sum(-1) / sigma2
                        + 3 * torch.log(torch.tensor(2 * torch.pi * sigma2)))
        expected = direct.mean()
        actual = gaussian_mixture_nll(self.gamma, p, self.rgb, decoder, sigma2=sigma2)
        self.assertTrue(torch.allclose(actual, expected, atol=2e-6, rtol=2e-6))

    def test_foreground_component_permutation_preserves_mixture_likelihood(self):
        raw = torch.rand(1, NODES, 3)
        p = torch.softmax(raw, dim=-1)
        permuted = p[:, :, [0, 2, 1]]
        decoder = make_decoder(4, seed=12)
        first = gaussian_mixture_nll(self.gamma, p, self.rgb, decoder, sigma2=0.05)
        second = gaussian_mixture_nll(self.gamma, permuted, self.rgb, decoder, sigma2=0.05)
        self.assertTrue(torch.allclose(first, second, atol=2e-6, rtol=2e-6))

    def test_padded_zero_weight_components_have_finite_backward(self):
        p = torch.zeros((2, NODES, 3), dtype=torch.float32)
        p[0, :, 0] = 1.0  # K=1 image padded to the batch maximum K=3.
        p[1] = torch.softmax(torch.randn(NODES, 3), dim=-1)
        p.requires_grad_()
        gamma = self.gamma.expand(2, -1, -1).clone().requires_grad_()
        rgb = self.rgb.expand(2, -1, -1, -1).clone()
        decoder = make_decoder(4, seed=13)
        loss = gaussian_mixture_nll(gamma, p, rgb, decoder, sigma2=0.04)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNone(gamma.grad)
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertTrue(torch.equal(p.grad[0, :, 1:], torch.zeros_like(p.grad[0, :, 1:])))
        self.assertGreater(float(p.grad[1].abs().sum().detach()), 0.0)

    def test_gamma_path_is_detached_while_assignment_path_is_live(self):
        gamma = self.gamma.clone().requires_grad_()
        p = torch.softmax(torch.randn(1, NODES, 3), dim=-1).requires_grad_()
        decoder = make_decoder(4, seed=14)
        loss = gaussian_mixture_nll(gamma, p, self.rgb, decoder, sigma2=0.06)
        loss.backward()
        self.assertIsNone(gamma.grad)
        self.assertIsNotNone(p.grad)
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(float(p.grad.abs().sum().detach()), 0.0)

    def test_actual_diffusion_q_receives_likelihood_gradient(self):
        torch.manual_seed(15)
        raw = torch.rand(1, NODES, NODES)
        q = ((raw + raw.transpose(1, 2)) * 0.5).requires_grad_()
        p = spatial_diffusion(q, [[0, 508, 1023]])
        decoder = make_decoder(4, seed=15)
        loss = gaussian_mixture_nll(self.gamma, p, self.rgb, decoder, sigma2=0.045)
        loss.backward()
        offdiag = q.grad.clone()
        offdiag.diagonal(dim1=1, dim2=2).zero_()
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertGreater(float(offdiag.abs().max().detach()), 0.0)
        self.assertTrue(torch.equal(q.grad.diagonal(dim1=1, dim2=2),
                                    torch.zeros((1, NODES))))

    def test_exact64_image_population_variance_calibration_and_rejection(self):
        pixels = (torch.arange(64 * 128 * 128 * 3, dtype=torch.int64)
                  .remainder(251).reshape(64, 128, 128, 3).to(torch.uint8))
        actual = calibrate_sigma2(pixels)
        x = pixels.to(torch.float64) / 255.0
        expected = float(x.var(dim=(1, 2), unbiased=False).mean())
        self.assertAlmostEqual(actual, expected, places=14)
        self.assertGreater(actual, 0.0)
        with self.assertRaisesRegex(ValueError, r"exactly \[64,128,128,3\]"):
            calibrate_sigma2(pixels[:63])
        with self.assertRaisesRegex(ValueError, "positive"):
            calibrate_sigma2(torch.zeros_like(pixels))
        image_constant = torch.empty_like(pixels)
        for index in range(64):
            image_constant[index].fill_(index)
        with self.assertRaisesRegex(ValueError, "positive"):
            calibrate_sigma2(image_constant)
        with self.assertRaises(TypeError):
            calibrate_sigma2(pixels.float())

    def test_decoder_initialization_is_repeatable_without_changing_global_rng(self):
        torch.manual_seed(999)
        before = torch.random.get_rng_state().clone()
        first = make_decoder(4, seed=19)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        second = make_decoder(4, seed=19)
        self.assertTrue(all(torch.equal(a, b) for a, b in
                            zip(first.state_dict().values(), second.state_dict().values())))

    def test_decoder_initialization_never_seeds_cuda(self):
        with mock.patch.object(torch.cuda, "manual_seed_all",
                               side_effect=AssertionError("CUDA seeding is forbidden")) as seed_cuda:
            decoder = make_decoder(4, seed=191)
        self.assertIsInstance(decoder, SpatialMixtureDecoder)
        seed_cuda.assert_not_called()

    def test_decoder_and_fixed_variance_contracts_reject_invalid_values(self):
        with self.assertRaises(ValueError):
            SpatialMixtureDecoder(0)
        p = torch.ones((1, NODES, 1))
        decoder = make_decoder(4, seed=20)
        for sigma2 in (0.0, float("nan"), -0.2):
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                gaussian_mixture_nll(self.gamma, p, self.rgb, decoder, sigma2=sigma2)


if __name__ == "__main__":
    unittest.main()
