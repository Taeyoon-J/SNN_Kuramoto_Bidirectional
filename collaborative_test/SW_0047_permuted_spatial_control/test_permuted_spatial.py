"""Checks for the deterministic, label-free permuted spatial control."""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from snn_kuramoto_bidirectional.spike_classifier import (
    permute_spatial_kernel,
    seeded_spatial_permutation,
    spatial_gaussian_kernel,
    spike_synchrony_affinity,
)


class PermutedSpatialTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.spikes = torch.randint(0, 2, (2, 16, 9)).float()
        self.components = torch.randint(0, 2, (2, 3, 16, 9)).float()
        self.kernel = spatial_gaussian_kernel(16, 4, 1.5)
        self.spike_affinity = spike_synchrony_affinity(self.spikes, self.components)

    def test_identity_permutation_matches_aligned_spatial_affinity(self):
        identity_kernel = permute_spatial_kernel(self.kernel, torch.arange(16))
        aligned = spike_synchrony_affinity(
            self.spikes, self.components, spatial_sigma=1.5,
            spatial_grid_size=4, affinity_mode="spike_spatial")
        torch.testing.assert_close(identity_kernel, self.kernel, rtol=0, atol=0)
        torch.testing.assert_close(aligned, self.spike_affinity * identity_kernel)

    def test_seeded_permutation_matches_manual_kernel_permutation(self):
        seed = 2
        permutation = seeded_spatial_permutation(16, seed)
        actual = spike_synchrony_affinity(
            self.spikes, self.components, spatial_sigma=1.5,
            spatial_grid_size=4, affinity_mode="spike_spatial_permuted",
            spatial_permutation_seed=seed)
        manual_kernel = self.kernel[permutation][:, permutation]
        torch.testing.assert_close(actual, self.spike_affinity * manual_kernel)
        self.assertFalse(torch.equal(manual_kernel, self.kernel))
        self.assertTrue(torch.equal(permutation, seeded_spatial_permutation(16, seed)))

    def test_invalid_permutations_and_missing_seed_fail(self):
        for invalid in ([0, 1, 2], [0] * 16, list(range(15)) + [16], [float(i) for i in range(16)]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    permute_spatial_kernel(self.kernel, invalid)
        with self.assertRaises(ValueError):
            spike_synchrony_affinity(
                self.spikes, self.components, spatial_sigma=1.5,
                spatial_grid_size=4, affinity_mode="spike_spatial_permuted")
        with self.assertRaises(ValueError):
            seeded_spatial_permutation(16, -1)


if __name__ == "__main__":
    unittest.main()
