import unittest

import numpy as np  # initialize shared OpenMP runtime before torch on Windows
import torch

from collaborative_test.SW_0126_history_event_binding.binder import TemporalSlotRGBBinder


class BinderTests(unittest.TestCase):
    def test_batched_assignment_decoder_uses_only_spike_traces_and_assignment_mix(self):
        torch.manual_seed(8)
        model = TemporalSlotRGBBinder(patch_grid=4, image_size=32, init_seed=126)
        events = torch.rand(2, 4, 16, 512, requires_grad=True)
        result = model(events)
        self.assertEqual(tuple(result["assignment"].shape), (2, 11, 16))
        self.assertTrue(torch.allclose(result["assignment"].sum(dim=1),
                                       torch.ones(2, 16), atol=1e-6, rtol=0))
        self.assertEqual(tuple(result["slot_rgb_coefficients"].shape), (2, 11, 32, 3))
        self.assertEqual(tuple(result["pixel_assignment"].shape), (2, 11, 32, 32))
        self.assertEqual(tuple(result["reconstruction"].shape), (2, 32, 32, 3))
        slot_rgb = torch.sigmoid(torch.einsum("hwf,bkfc->bkhwc",
                                               result["coordinate_basis"],
                                               result["slot_rgb_coefficients"]))
        direct_mix = torch.einsum("bkhw,bkhwc->bhwc",
                                  result["pixel_assignment"], slot_rgb)
        self.assertTrue(torch.equal(result["reconstruction"], direct_mix))
        result["reconstruction"].square().mean().backward()
        self.assertIsNotNone(events.grad)
        self.assertTrue(torch.isfinite(events.grad).all())
        self.assertGreater(float(events.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.slot_queries.grad.abs().sum()), 0.0)

    def test_paired_initialization_is_deterministic_without_global_rng_side_effect(self):
        torch.manual_seed(90)
        before = torch.random.get_rng_state().clone()
        first = TemporalSlotRGBBinder(patch_grid=2, image_size=8, init_seed=4)
        after = torch.random.get_rng_state()
        second = TemporalSlotRGBBinder(patch_grid=2, image_size=8, init_seed=4)
        self.assertTrue(torch.equal(before, after))
        for left, right in zip(first.parameters(), second.parameters()):
            self.assertTrue(torch.equal(left, right))

    def test_rejects_wrong_trace_grid_or_time_horizon(self):
        model = TemporalSlotRGBBinder(patch_grid=4, image_size=32)
        with self.assertRaisesRegex(ValueError, r"actual \[B,4,N,T\]"):
            model(torch.zeros(1, 3, 16, 512))
        with self.assertRaisesRegex(ValueError, "full settled 512-frame"):
            model(torch.zeros(1, 4, 16, 64))
        with self.assertRaisesRegex(ValueError, "64-D features and divisible image/patch geometry"):
            TemporalSlotRGBBinder(patch_grid=4, image_size=30)


if __name__ == "__main__":
    unittest.main()
