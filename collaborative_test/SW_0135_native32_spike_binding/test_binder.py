from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for value in (str(ROOT), str(HERE)):
    if value not in sys.path:
        sys.path.insert(0, value)

from binder import (
    CELL_VARIANCE, DIM, GRID, IMAGE, PATCH, PATCHES, SLOTS,
    NativeSpikeSlotBinder, RelativeSlotRGBDecoder, canonical_hard_labels,
    pixel_assignment_weights, render_slot_rgb, slot_geometry,
)


class Native32BinderTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_actual_spikes_produce_native32_slots_and_finite_rgb_credit(self):
        torch.manual_seed(13501)
        spikes = torch.rand(1, 4, 1024, 512, requires_grad=True)
        binder = NativeSpikeSlotBinder(seed=135)
        decoder = RelativeSlotRGBDecoder(seed=106)
        prediction, assignments, slots, patch_features, labels = __import__(
            "binder").reconstruct_from_spikes(
                spikes, binder, decoder, chunk_size=4096, checkpoint_chunks=True)
        self.assertEqual(tuple(assignments.shape), (1, 1024, 11))
        self.assertEqual(tuple(slots.shape), (1, 11, 64))
        self.assertEqual(tuple(patch_features.shape), (1, 1024, 64))
        self.assertEqual(tuple(prediction.shape), (1, 128, 128, 3))
        self.assertEqual(tuple(labels.shape), (1, 32, 32))
        self.assertTrue(torch.isfinite(prediction).all())
        loss = prediction.square().mean()
        loss.backward()
        self.assertIsNotNone(spikes.grad)
        self.assertTrue(torch.isfinite(spikes.grad).all())
        self.assertGreater(float(spikes.grad.abs().sum()), 0.0)

    def test_slot_attention_is_equivariant_to_permuted_fixed_noise(self):
        torch.manual_seed(13502)
        spikes = torch.rand(1, 4, PATCHES, 512)
        binder = NativeSpikeSlotBinder(seed=135)
        permutation = torch.tensor([4, 0, 7, 1, 8, 2, 10, 5, 3, 9, 6])
        first_p, first_z, features = binder(spikes)
        second_p, second_z, second_features = binder(
            spikes, initial_slots=binder.initial_slots(1)[:, permutation])
        self.assertTrue(torch.allclose(second_p, first_p[:, :, permutation], atol=2e-6, rtol=2e-6))
        self.assertTrue(torch.allclose(second_z, first_z[:, permutation], atol=2e-6, rtol=2e-6))
        self.assertTrue(torch.equal(features, second_features))

    def test_live_geometry_and_each_patch_maps_to_its_own_4x4_pixels(self):
        raw = torch.randn(1, PATCHES, SLOTS)
        assignment = raw.softmax(-1).requires_grad_()
        mass, centroid, scale = slot_geometry(assignment)
        centers_axis = (torch.arange(GRID, dtype=torch.float32) + .5) * (2 / GRID) - 1
        yy, xx = torch.meshgrid(centers_axis, centers_axis, indexing="ij")
        centers = torch.stack((xx.flatten(), yy.flatten()), -1)
        expected_mass = assignment.sum(1).clamp_min(1.)
        expected_mu = torch.einsum("bnk,nd->bkd", assignment, centers) / expected_mass.unsqueeze(-1)
        self.assertTrue(torch.allclose(mass, expected_mass))
        self.assertTrue(torch.allclose(centroid, expected_mu))
        self.assertAlmostEqual(CELL_VARIANCE, ((4**2 - 1) / 12) * (2 / 128) ** 2)

        weights = pixel_assignment_weights(assignment)
        self.assertEqual(tuple(weights.shape), (1, 128 * 128, SLOTS))
        for row, col in ((0, 0), (3, 7), (31, 31)):
            patch_id = row * GRID + col
            start = (row * PATCH) * IMAGE + col * PATCH
            for dy in range(PATCH):
                pixel_start = start + dy * IMAGE
                self.assertTrue(torch.equal(weights[0, pixel_start:pixel_start + PATCH],
                                            assignment[0, patch_id].expand(PATCH, -1)))
        geometry_loss = centroid.square().mean() + scale.square().mean()
        geometry_loss.backward()
        self.assertIsNotNone(assignment.grad)
        self.assertTrue(torch.isfinite(assignment.grad).all())
        self.assertGreater(float(assignment.grad.abs().sum()), 0.)

    def test_checkpointed_chunk_render_matches_plain_render_and_gradients(self):
        torch.manual_seed(13503)
        p0 = torch.randn(1, PATCHES, SLOTS).softmax(-1)
        z0 = torch.randn(1, SLOTS, DIM)
        decoder_a = RelativeSlotRGBDecoder(seed=106)
        decoder_b = RelativeSlotRGBDecoder(seed=106)
        decoder_b.load_state_dict(decoder_a.state_dict(), strict=True)
        p_a, z_a = p0.clone().requires_grad_(), z0.clone().requires_grad_()
        p_b, z_b = p0.clone().requires_grad_(), z0.clone().requires_grad_()
        out_a = render_slot_rgb(p_a, z_a, decoder_a, chunk_size=2048, checkpoint_chunks=True)
        out_b = render_slot_rgb(p_b, z_b, decoder_b, chunk_size=2048, checkpoint_chunks=False)
        probe = torch.linspace(.1, 1., IMAGE * IMAGE * 3).reshape(1, IMAGE, IMAGE, 3)
        loss_a, loss_b = (out_a * probe).mean(), (out_b * probe).mean()
        loss_a.backward()
        loss_b.backward()
        self.assertTrue(torch.allclose(out_a, out_b, atol=2e-7, rtol=2e-6))
        self.assertTrue(torch.allclose(p_a.grad, p_b.grad, atol=1e-8, rtol=2e-5))
        self.assertTrue(torch.allclose(z_a.grad, z_b.grad, atol=1e-8, rtol=2e-5))
        for left, right in zip(decoder_a.parameters(), decoder_b.parameters()):
            self.assertTrue(torch.allclose(left.grad, right.grad, atol=1e-8, rtol=2e-5))

    def test_hard_partition_uses_largest_group_as_background(self):
        assignment = torch.zeros(1, PATCHES, SLOTS)
        assignment[:, :500, 2] = 1.
        assignment[:, 500:, 1] = 1.
        labels = canonical_hard_labels(assignment)
        self.assertEqual(tuple(labels.shape), (1, 32, 32))
        self.assertEqual(int((labels == 0).sum()), 524)
        self.assertEqual(int((labels == 1).sum()), 500)


if __name__ == "__main__":
    unittest.main()
