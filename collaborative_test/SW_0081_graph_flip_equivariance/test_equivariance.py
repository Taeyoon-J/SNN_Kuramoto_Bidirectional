import sys
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from equivariance import (freeze_graph_only, graph_equivariance_mse,
                          horizontal_flip_permutation, target_equivariance_weights,
                          unflip_adjacency)


class ToyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.graph_generator = nn.Linear(3, 3)
        self.other = nn.Linear(2, 2)


class FlipEquivarianceTests(unittest.TestCase):
    def test_permutation_is_horizontal_reflection_and_roundtrips(self):
        perm = horizontal_flip_permutation(4)
        grid = torch.arange(16).reshape(4, 4)
        self.assertTrue(torch.equal(perm.reshape(4, 4), grid.flip(1)))
        self.assertTrue(torch.equal(perm[perm], torch.arange(16)))
        matrix = torch.arange(256).reshape(1, 16, 16)
        self.assertTrue(torch.equal(unflip_adjacency(unflip_adjacency(matrix, 4), 4), matrix))

    def test_equivariant_toy_graph_has_zero_aligned_loss(self):
        # Grid-neighbor graph is invariant under horizontal reflection.
        n = 16
        coords = torch.stack(torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij"), -1).reshape(n, 2)
        graph = (torch.cdist(coords.float(), coords.float()) == 1).float().unsqueeze(0)
        flipped_graph = graph.index_select(-2, horizontal_flip_permutation(4)) \
                             .index_select(-1, horizontal_flip_permutation(4))
        loss, aligned = graph_equivariance_mse(graph, flipped_graph, 4)
        self.assertEqual(float(loss), 0.0)
        self.assertTrue(torch.equal(aligned, graph))

    def test_gradients_and_freeze_contract(self):
        core, encoder = ToyCore(), nn.Linear(2, 2)
        params = freeze_graph_only(core, encoder)
        source = torch.randn(3, 3)
        target = torch.randn(3, 3)
        loss = (core.graph_generator(source) - target).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(core.graph_generator.weight.grad).all())
        self.assertTrue(all(not p.requires_grad for n, p in core.named_parameters()
                            if not n.startswith("graph_generator.")))
        self.assertTrue(all(not p.requires_grad for p in encoder.parameters()))
        self.assertGreater(float(core.graph_generator.weight.grad.norm()), 0)
        self.assertTrue(params)

    def test_measured_gradient_weight_scales(self):
        weights = target_equivariance_weights(torch.tensor(10.), torch.tensor(2.))
        self.assertAlmostEqual(weights["weight_0p1x"], .5)
        self.assertAlmostEqual(weights["weight_1x"], 5.)
        with self.assertRaises(ValueError):
            target_equivariance_weights(1., 0.)


if __name__ == "__main__":
    unittest.main()
