"""CPU regressions for the SW0132 geometry and chunked differentiable renderer."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "snn_kuramoto_bidirectional"),
                str(ROOT / "collaborative_test")]
import numpy as np  # import numpy before torch in the registered environment
import torch

from collaborative_test.SW_0132_partition_relative_rgb.relative_rgb import (
    CELL_VARIANCE, RelativeRGBDecoder, assignment_to_weights, patch_centers,
    patch_pixel_weights, partition_geometry, pixel_centers, reconstruct_one,
    render_partition,
)


def _fixture(seed=4):
    torch.manual_seed(seed)
    labels = torch.arange(256) % 3
    hard = torch.nn.functional.one_hot(labels, num_classes=3).float()
    q = torch.rand(256, 256)
    q = (q + q.T) / 2
    q.fill_diagonal_(1)
    gamma = torch.randn(256, 8)
    target = torch.rand(128, 128, 3)
    return q, hard, gamma, target


class RelativeRGBTests(unittest.TestCase):
    def test_live_geometry_matches_exact_full_pixel_moments(self):
        _, hard, _, _ = _fixture()
        mass, centroid, sigma = partition_geometry(hard)
        pixel_w = patch_pixel_weights(hard)
        pixel_xy = pixel_centers(hard.device, hard.dtype)
        expected_mass = pixel_w.sum(0)
        expected_mu = pixel_w.T @ pixel_xy / expected_mass[:, None]
        expected_var = (pixel_w[:, :, None] * (pixel_xy[:, None] - expected_mu[None]).square()).sum(0)
        expected_var /= expected_mass[:, None]
        self.assertTrue(torch.equal(mass * (8 * 8), expected_mass))
        self.assertTrue(torch.allclose(centroid, expected_mu, atol=2e-7, rtol=0))
        self.assertTrue(torch.allclose(sigma.square(), expected_var, atol=2e-7, rtol=0))
        self.assertAlmostEqual(CELL_VARIANCE, 0.00128173828125, places=15)

    def test_partition_column_permutation_preserves_full_rgb_forward(self):
        q, hard, gamma, _, = _fixture()
        decoder = RelativeRGBDecoder()
        w, _ = assignment_to_weights(q, hard)
        pred = render_partition(w, gamma, decoder, chunk_size=2048)
        permutation = torch.tensor([2, 0, 1])
        wp, _ = assignment_to_weights(q, hard[:, permutation])
        pred_permuted = render_partition(wp, gamma, decoder, chunk_size=2048)
        self.assertTrue(torch.allclose(pred, pred_permuted, atol=2e-7, rtol=1e-6))

    def test_assignment_geometry_and_decoder_receive_finite_gradients(self):
        q, hard, gamma, target = _fixture()
        q.requires_grad_(True)
        gamma.requires_grad_(True)
        decoder = RelativeRGBDecoder()
        prediction, loss, detail = reconstruct_one(q, hard, gamma, target, decoder,
                                                   chunk_size=4096)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(q.grad)
        self.assertGreater(float(q.grad.abs().sum()), 0.0)
        self.assertIsNone(gamma.grad, "RGB content inputs are detached by protocol")
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in decoder.parameters()))
        self.assertTrue(torch.equal(detail["W"].detach(), hard))

    def test_checkpointed_chunk_matches_full_render_and_gradients(self):
        q, hard, gamma, target = _fixture(seed=11)
        decoder_a = RelativeRGBDecoder()
        decoder_b = RelativeRGBDecoder()
        decoder_b.load_state_dict(decoder_a.state_dict())
        qa = q.clone().requires_grad_(True)
        qb = q.clone().requires_grad_(True)
        wa, _ = assignment_to_weights(qa, hard)
        wb, _ = assignment_to_weights(qb, hard)
        pred_a = render_partition(wa, gamma, decoder_a, chunk_size=1024,
                                  checkpoint_chunks=True)
        pred_b = render_partition(wb, gamma, decoder_b, chunk_size=16384,
                                  checkpoint_chunks=False)
        self.assertTrue(torch.allclose(pred_a, pred_b, atol=2e-7, rtol=1e-6))
        (pred_a - target).square().mean().backward()
        (pred_b - target).square().mean().backward()
        self.assertTrue(torch.allclose(qa.grad, qb.grad, atol=2e-7, rtol=2e-5))
        for pa, pb in zip(decoder_a.parameters(), decoder_b.parameters()):
            self.assertTrue(torch.allclose(pa.grad, pb.grad, atol=2e-7, rtol=2e-5))

    def test_patch_to_pixel_weights_are_row_major_eight_by_eight(self):
        weights = torch.zeros(256, 2)
        weights[17, 0] = 1
        weights[18, 1] = 1
        pixel = patch_pixel_weights(weights)
        self.assertEqual(tuple(pixel.shape), (128 * 128, 2))
        matrix = pixel.reshape(128, 128, 2)
        self.assertTrue(torch.equal(matrix[8:16, 8:16, 0], torch.ones(8, 8)))
        self.assertTrue(torch.equal(matrix[8:16, 16:24, 1], torch.ones(8, 8)))
        self.assertEqual(float(matrix.sum()), 128.0)


if __name__ == "__main__":
    unittest.main()
