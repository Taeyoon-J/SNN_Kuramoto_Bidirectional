import unittest
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from interventions import permute_gate_regions, temporal_mean_gates
from snn_kuramoto_bidirectional.kuramoto_layer import graphVectorKuramoto


class InterventionTest(unittest.TestCase):
    def test_permutation_preserves_each_node_trajectory_values(self):
        gamma = torch.arange(2 * 4 * 3).reshape(2, 4, 3).float()
        gate = torch.arange(8).reshape(2, 4).float()
        permutation = torch.tensor([2, 0, 3, 1])
        g2, m2 = permute_gate_regions(gamma, gate, permutation)
        self.assertTrue(torch.equal(g2, gamma[:, permutation]))
        self.assertTrue(torch.equal(m2, gate[:, permutation]))
        for b in range(2):
            self.assertEqual(sorted(g2[b].flatten().tolist()), sorted(gamma[b].flatten().tolist()))

    def test_temporal_mean_is_per_sample_and_region(self):
        gh = torch.tensor([[[[0.], [2.]], [[2.], [4.]]]])
        mh = torch.tensor([[[0., 2.], [2., 4.]]])
        gm, mm = temporal_mean_gates(gh, mh)
        self.assertTrue(torch.equal(gm, torch.tensor([[[1.], [3.]]])))
        self.assertTrue(torch.equal(mm, torch.tensor([[1., 3.]])))

    def test_invalid_permutation_rejected(self):
        with self.assertRaises(ValueError):
            permute_gate_regions(torch.zeros(1, 3, 2), torch.zeros(1, 3), [0, 0, 2])

    def test_k_zero_removes_dependence_on_inter_region_graph(self):
        model = graphVectorKuramoto(4, D=2, K=0.0, device="cpu", kuramoto_backend="factorized")
        theta = torch.randn(1, 4, 2)
        gamma = torch.randn(1, 4, 2)
        a = torch.rand(1, 4, 4)
        b = torch.rand(1, 4, 4)
        out_a = model(theta, gamma, A=a)
        out_b = model(theta, gamma, A=b)
        self.assertTrue(torch.equal(out_a, out_b))


if __name__ == "__main__":
    unittest.main()
