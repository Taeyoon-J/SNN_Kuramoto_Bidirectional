import sys
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(ROOT))
from collaborative_test.SW_0153_gamma_ordering_audit import audit_source as audit
from snn_kuramoto_bidirectional.gamma_initializer import GammaToDrive
from snn_kuramoto_bidirectional.graph_generator import ImageConditionedGraph


class GammaOrderingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(153)
        self.drive_fn = GammaToDrive(8, 16, 4, drive_mode="static", phase_mode="none")
        self.channel_projection = nn.Linear(8, 4, bias=False)
        self.graph = ImageConditionedGraph(
            in_channels=8, hidden_dim=8, top_k=4, temperature=.3,
            grid_size=4, spatial_decay=.65, geodesic_steps=0)
        self.gamma = torch.randn(2, 8, 16)

    def _projections(self, gamma):
        drive = self.drive_fn(gamma, self.channel_projection)
        adjacency = self.graph(gamma)
        return drive, adjacency

    def test_consistent_global_channel_reindex_preserves_drive_and_graph(self):
        before_drive, before_graph = self._projections(self.gamma)
        permutation = torch.tensor([6, 1, 4, 0, 7, 3, 5, 2])
        reordered = audit.permute_gamma_channels(self.gamma, permutation)
        with torch.no_grad():
            drive_weight = self.channel_projection.weight.detach().clone()
            graph_weight = self.graph.projection.weight.detach().clone()
            self.channel_projection.weight.copy_(drive_weight.index_select(1, permutation))
            self.graph.projection.weight.copy_(graph_weight.index_select(1, permutation))
        try:
            after_drive, after_graph = self._projections(reordered)
        finally:
            with torch.no_grad():
                self.channel_projection.weight.copy_(drive_weight)
                self.graph.projection.weight.copy_(graph_weight)
        torch.testing.assert_close(after_drive, before_drive, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(after_graph, before_graph, rtol=1e-6, atol=1e-6)

    def test_naive_channel_shuffle_changes_frozen_projection_outputs(self):
        before_drive, before_graph = self._projections(self.gamma)
        reordered = audit.permute_gamma_channels(self.gamma, [6, 1, 4, 0, 7, 3, 5, 2])
        after_drive, after_graph = self._projections(reordered)
        self.assertGreater(float((after_drive - before_drive).abs().max().detach()), 1e-5)
        self.assertGreater(float((after_graph - before_graph).abs().max().detach()), 1e-5)

    def test_patch_shuffle_is_not_channel_shuffle_and_spatial_graph_changes(self):
        before_drive, before_graph = self._projections(self.gamma)
        permutation = torch.randperm(16)
        reordered = self.gamma.index_select(2, permutation)
        after_drive, after_graph = self._projections(reordered)
        restored_drive = torch.empty_like(after_drive)
        restored_drive[:, permutation] = after_drive
        inverse = torch.argsort(permutation)
        restored_graph = after_graph.index_select(1, inverse).index_select(2, inverse)
        torch.testing.assert_close(restored_drive, before_drive, rtol=1e-6, atol=1e-6)
        self.assertGreater(float((restored_graph - before_graph).abs().max().detach()), 1e-5)

    def test_noninvolutive_permutation_restores_pairwise_axes_with_inverse(self):
        # A 3-cycle is not its own inverse, so indexing the new matrix by p twice fails.
        permutation = torch.tensor([1, 2, 0])
        original = torch.arange(9).reshape(3, 3)
        transformed = original.index_select(0, permutation).index_select(1, permutation)
        restored = audit.unpermute_nodes(transformed, permutation)
        torch.testing.assert_close(restored, original)
        wrong = transformed.index_select(0, permutation).index_select(1, permutation)
        self.assertFalse(torch.equal(wrong, original))

    def test_permutation_validation_and_native_layout(self):
        with self.assertRaises(ValueError):
            audit.permute_gamma_channels(torch.randn(1, 1024, 8), range(8))
        with self.assertRaises(ValueError):
            audit.permute_gamma_channels(self.gamma, [0, 1, 2, 3, 4, 5, 6, 6])
        with self.assertRaises(ValueError):
            audit.permute_gamma_patches(torch.randn(2, 8, 16), torch.arange(15))


if __name__ == "__main__":
    unittest.main()
