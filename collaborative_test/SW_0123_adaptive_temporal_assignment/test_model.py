import torch
import unittest

from model import AdaptiveTemporalRGBModel, PATCHES, PATCH_SIZE, SLOTS, fit_spike_rms


class AdaptiveTemporalRGBModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_variable_horizons_produce_finite_assignments_and_native_rgb(self):
        model = AdaptiveTemporalRGBModel()
        xy = torch.rand(PATCHES, 2) * 2 - 1
        for steps in (7, 19):
            spikes = torch.rand(2, 4, PATCHES, steps, requires_grad=True)
            out = model(spikes, xy)
            self.assertEqual(tuple(out["assignment"].shape), (2, PATCHES, SLOTS))
            self.assertEqual(tuple(out["reconstructed_patches"].shape), (2, PATCHES, 8, 8, 3))
            self.assertEqual(tuple(out["reconstructed_image"].shape), (2, 3, 128, 128))
            self.assertTrue(torch.isfinite(out["assignment"]).all())
            self.assertTrue(torch.isfinite(out["reconstructed_image"]).all())
            self.assertTrue((out["reconstructed_image"] >= 0).all())
            self.assertTrue((out["reconstructed_image"] <= 1).all())
            self.assertTrue(torch.allclose(out["assignment"].sum(dim=-1), torch.ones(2, PATCHES),
                                           rtol=1e-6, atol=1e-6))
            out["reconstructed_image"].square().mean().backward()
            self.assertIsNotNone(spikes.grad)
            self.assertTrue(torch.isfinite(spikes.grad).all())
            self.assertGreater(float(spikes.grad.norm()), 0.0)

    def test_patch_permutation_with_coordinates_is_equivariant(self):
        model = AdaptiveTemporalRGBModel().eval()
        spikes = torch.rand(1, 4, PATCHES, 11)
        xy = torch.rand(PATCHES, 2) * 2 - 1
        perm = torch.randperm(PATCHES)
        original = model(spikes, xy)
        permuted = model(spikes[:, :, perm], xy[perm])
        torch.testing.assert_close(permuted["assignment"], original["assignment"][:, perm],
                                   rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(permuted["reconstructed_patches"],
                                   original["reconstructed_patches"][:, perm],
                                   rtol=2e-5, atol=2e-6)

    def test_assignment_path_accepts_only_spikes_and_decoder_coordinates(self):
        head = AdaptiveTemporalRGBModel().assignment_head
        self.assertEqual(list(__import__("inspect").signature(head.forward).parameters),
                         ["component_spikes"])
        with self.assertRaises(ValueError):
            head(torch.rand(1, 3, PATCHES, 8))
        with self.assertRaises(ValueError):
            head(torch.rand(1, 4, PATCHES - 1, 8))

    def test_rms_is_fitted_per_component_and_neutral_for_tiny_channels(self):
        spikes = torch.ones(2, 4, PATCHES, 5)
        spikes[:, 1] *= 2
        spikes[:, 2] *= 1e-10
        rms = fit_spike_rms(spikes)
        torch.testing.assert_close(rms, torch.tensor([1.0, 2.0, 1.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
