"""Numerical tests for the optional spike-times-spatial affinity."""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from snn_kuramoto_bidirectional.spike_classifier import (
    spatial_gaussian_kernel,
    spike_synchrony_affinity,
    spike_synchrony_components,
)


class SpatialSpikeAffinityTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.spikes = torch.randint(0, 2, (3, 4, 7)).float()
        self.components = torch.randint(0, 2, (3, 2, 4, 7)).float()

    def test_gaussian_kernel_shape_distance_and_symmetry(self):
        kernel = spatial_gaussian_kernel(4, (2, 2), 1.0)
        self.assertEqual(tuple(kernel.shape), (4, 4))
        self.assertTrue(torch.allclose(kernel, kernel.T))
        self.assertAlmostEqual(float(kernel[0, 0]), 1.0)
        self.assertAlmostEqual(float(kernel[0, 1]), torch.exp(torch.tensor(-0.5)).item())
        self.assertAlmostEqual(float(kernel[0, 3]), torch.exp(torch.tensor(-1.0)).item())

    def test_spike_times_gaussian_and_spatial_only_match_formula(self):
        spike = spike_synchrony_affinity(self.spikes, self.components)
        kernel = spatial_gaussian_kernel(4, 2, 1.5)
        combined = spike_synchrony_affinity(
            self.spikes, self.components, spatial_sigma=1.5,
            spatial_grid_size=2, affinity_mode="spike_spatial")
        spatial = spike_synchrony_affinity(
            self.spikes, self.components, spatial_sigma=1.5,
            spatial_grid_size=(2, 2), affinity_mode="spatial_only")
        self.assertEqual(tuple(combined.shape), (3, 4, 4))
        self.assertEqual(tuple(spatial.shape), (3, 4, 4))
        torch.testing.assert_close(combined, spike * kernel)
        torch.testing.assert_close(spatial, kernel.expand(3, -1, -1))

    def test_default_is_unchanged_and_uniform_sigma_is_spike_control(self):
        original = spike_synchrony_affinity(self.spikes, self.components)
        explicit = spike_synchrony_affinity(
            self.spikes, self.components, spatial_sigma=float("inf"),
            spatial_grid_size=2, affinity_mode="spike_spatial")
        torch.testing.assert_close(original, explicit, rtol=0, atol=0)
        default_groups = spike_synchrony_components(
            self.spikes, components=self.components, settle=1, synchrony_threshold=.3)
        explicit_groups = spike_synchrony_components(
            self.spikes, components=self.components, settle=1,
            synchrony_threshold=.3, affinity_mode="spike")
        self.assertEqual(default_groups, explicit_groups)

    def test_permutation_then_spatial_weighting_is_reproducible(self):
        permutation = torch.tensor([2, 0, 3, 1])
        baseline = spike_synchrony_affinity(self.spikes, self.components)
        permuted_spike = spike_synchrony_affinity(
            self.spikes[:, permutation], self.components[:, :, permutation])
        kernel = spatial_gaussian_kernel(4, 2, 1.0)
        permuted_then_weighted = spike_synchrony_affinity(
            self.spikes[:, permutation], self.components[:, :, permutation],
            spatial_sigma=1.0, spatial_grid_size=2, affinity_mode="spike_spatial")
        torch.testing.assert_close(permuted_spike, baseline[:, permutation][:, :, permutation])
        torch.testing.assert_close(permuted_then_weighted, permuted_spike * kernel)

    def test_grouping_accepts_spatial_modes_without_any_target_argument(self):
        groups = spike_synchrony_components(
            self.spikes, components=self.components, settle=1,
            synchrony_threshold=.2, spatial_sigma=1.5, spatial_grid_size=2,
            affinity_mode="spike_spatial")
        self.assertEqual(len(groups), self.spikes.size(0))


if __name__ == "__main__":
    unittest.main()
