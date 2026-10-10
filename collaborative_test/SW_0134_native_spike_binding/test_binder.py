"""CPU tests for the isolated SW0134 spike binder and RGB renderer."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

import torch

from collaborative_test.SW_0134_native_spike_binding.binder import (
    CELL_VARIANCE, DIM, GRID, IMAGE, PATCH, PATCHES, SLOTS,
    NativeSpikeSlotBinder, RelativeSlotRGBDecoder, canonical_hard_labels,
    render_slot_rgb,
)


class SW0134BinderTests(unittest.TestCase):
    def test_actual_spike_shape_binding_and_rgb_loss_reach_input_and_binder(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            torch.manual_seed(17)
            binder = NativeSpikeSlotBinder()
            decoder = RelativeSlotRGBDecoder()
            spikes = torch.rand(1, 4, PATCHES, 512, requires_grad=True)
            assignment, slots, features = binder(spikes)
            self.assertEqual(tuple(assignment.shape), (1, PATCHES, SLOTS))
            self.assertEqual(tuple(slots.shape), (1, SLOTS, DIM))
            self.assertEqual(tuple(features.shape), (1, PATCHES, DIM))
            self.assertTrue(torch.allclose(assignment.sum(-1), torch.ones(1, PATCHES),
                                           atol=1e-6, rtol=0.0))
            image = render_slot_rgb(assignment, slots, decoder, chunk_size=8192,
                                    checkpoint_chunks=True)
            self.assertEqual(tuple(image.shape), (1, IMAGE, IMAGE, 3))
            loss = (image - 0.37).square().mean()
            loss.backward()
            self.assertTrue(torch.isfinite(spikes.grad).all())
            self.assertGreater(float(spikes.grad.norm()), 0.0)
            self.assertGreater(float(binder.input_projection.weight.grad.norm()), 0.0)
            self.assertGreater(float(decoder.net[0].weight.grad.norm()), 0.0)
            self.assertAlmostEqual(CELL_VARIANCE,
                                   ((PATCH ** 2 - 1) / 12.0) * (2.0 / IMAGE) ** 2)
            with self.assertRaisesRegex(ValueError, "actual spikes"):
                binder(torch.rand(1, 4, PATCHES, 64))
        finally:
            torch.set_num_threads(old_threads)

    def test_shared_refinement_is_equivariant_to_initial_slot_permutation(self):
        torch.manual_seed(23)
        binder = NativeSpikeSlotBinder()
        spikes = torch.rand(1, 4, PATCHES, 512)
        initial = binder.initial_slots(1)
        order = torch.tensor([3, 0, 9, 2, 10, 1, 7, 5, 4, 8, 6])
        base_p, base_slots, _ = binder(spikes, initial_slots=initial)
        perm_p, perm_slots, _ = binder(spikes, initial_slots=initial[:, order])
        self.assertTrue(torch.allclose(perm_p, base_p[:, :, order], atol=2e-6, rtol=2e-6))
        self.assertTrue(torch.allclose(perm_slots, base_slots[:, order], atol=2e-6, rtol=2e-6))

    def test_shared_mu_scale_and_permuted_noise_remain_exchangeable(self):
        torch.manual_seed(29)
        binder = NativeSpikeSlotBinder()
        self.assertEqual(tuple(binder.slot_mu.shape), (1, 1, DIM))
        self.assertEqual(tuple(binder.slot_log_scale.shape), (1, 1, DIM))
        with torch.no_grad():
            binder.slot_mu.copy_(torch.linspace(-0.2, 0.2, DIM).reshape(1, 1, DIM))
            binder.slot_log_scale.copy_(torch.linspace(-0.1, 0.1, DIM).reshape(1, 1, DIM))
        spikes = torch.rand(1, 4, PATCHES, 512)
        original_noise = binder.initial_noise.detach().clone()
        original_initial = binder.initial_slots(1)
        original_p, _, _ = binder(spikes)
        order = torch.tensor([5, 10, 2, 8, 0, 6, 1, 9, 4, 3, 7])
        with torch.no_grad():
            binder.initial_noise.copy_(original_noise[:, order])
        permuted_initial = binder.initial_slots(1)
        permuted_p, _, _ = binder(spikes)
        self.assertTrue(torch.equal(permuted_initial, original_initial[:, order]))
        self.assertTrue(torch.allclose(permuted_p, original_p[:, :, order],
                                       atol=2e-6, rtol=2e-6))

    def test_hard_readout_is_invariant_to_slot_ids_and_uses_largest_background(self):
        assignments = torch.zeros(1, PATCHES, SLOTS)
        labels = torch.zeros(PATCHES, dtype=torch.long)
        labels[:150] = 3
        labels[150:220] = 8
        labels[220:] = 1
        assignments[0, torch.arange(PATCHES), labels] = 1.0
        expected = canonical_hard_labels(assignments)
        self.assertEqual(int((expected == 0).sum()), 150)
        self.assertEqual(set(torch.unique(expected).tolist()), {0, 1, 2})
        permutation = torch.tensor([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
        actual = canonical_hard_labels(assignments[:, :, permutation])
        self.assertTrue(torch.equal(actual, expected))

    def test_checkpointed_chunk_renderer_matches_plain_forward_and_gradient(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            torch.manual_seed(31)
            decoder = RelativeSlotRGBDecoder()
            logits = torch.randn(1, PATCHES, SLOTS)
            assignments_a = torch.softmax(logits, dim=-1).detach().requires_grad_(True)
            slots_a = torch.randn(1, SLOTS, DIM, requires_grad=True)
            assignments_b = assignments_a.detach().clone().requires_grad_(True)
            slots_b = slots_a.detach().clone().requires_grad_(True)
            output_a = render_slot_rgb(assignments_a, slots_a, decoder, chunk_size=8192,
                                       checkpoint_chunks=False)
            output_b = render_slot_rgb(assignments_b, slots_b, decoder, chunk_size=4096,
                                       checkpoint_chunks=True)
            self.assertTrue(torch.allclose(output_a, output_b, atol=2e-6, rtol=2e-6))
            weight = torch.linspace(0.1, 1.0, output_a.numel()).reshape_as(output_a)
            loss_a = (output_a * weight).mean()
            loss_b = (output_b * weight).mean()
            grad_a = torch.autograd.grad(loss_a, (assignments_a, slots_a))
            grad_b = torch.autograd.grad(loss_b, (assignments_b, slots_b))
            for first, second in zip(grad_a, grad_b):
                self.assertTrue(torch.allclose(first, second, atol=3e-6, rtol=3e-5))
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
