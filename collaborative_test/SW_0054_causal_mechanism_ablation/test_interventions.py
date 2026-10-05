import unittest
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from interventions import apply_carrier_mask_intervention, temporal_means
from snn_kuramoto_bidirectional.kuramoto_layer import graphVectorKuramoto


class InterventionTest(unittest.TestCase):
    def test_gate_permutation_changes_mask_but_preserves_local_carrier(self):
        carrier = torch.arange(1 * 4 * 2).reshape(1, 4, 2).float()
        mask = torch.tensor([[.1, .2, .3, .4]])
        permutation = torch.tensor([2, 0, 3, 1])
        drive, used_carrier, used_mask = apply_carrier_mask_intervention(
            carrier, mask, "gate_perm", permutation=permutation)
        self.assertTrue(torch.equal(used_carrier, carrier))
        self.assertTrue(torch.equal(used_mask, mask[:, permutation]))
        self.assertTrue(torch.equal(drive, carrier * mask[:, permutation, None]))

    def test_carrier_permutation_changes_carrier_but_preserves_local_mask(self):
        carrier = torch.arange(1 * 4 * 2).reshape(1, 4, 2).float()
        mask = torch.tensor([[.1, .2, .3, .4]])
        permutation = torch.tensor([2, 0, 3, 1])
        drive, used_carrier, used_mask = apply_carrier_mask_intervention(
            carrier, mask, "carrier_perm", permutation=permutation)
        self.assertTrue(torch.equal(used_carrier, carrier[:, permutation]))
        self.assertTrue(torch.equal(used_mask, mask))
        self.assertTrue(torch.equal(drive, carrier[:, permutation] * mask.unsqueeze(-1)))

    def test_mask_and_carrier_means_intervene_only_on_selected_factor(self):
        carrier = torch.tensor([[[1., 2.], [3., 4.]]])
        mask = torch.tensor([[.2, .8]])
        mean_mask = torch.tensor([[.5, .25]])
        mean_carrier = torch.tensor([[[2., 3.], [4., 5.]]])
        drive_g, carrier_g, mask_g = apply_carrier_mask_intervention(
            carrier, mask, "gate_mean", mean_mask=mean_mask)
        self.assertTrue(torch.equal(carrier_g, carrier))
        self.assertTrue(torch.equal(mask_g, mean_mask))
        self.assertTrue(torch.equal(drive_g, carrier * mean_mask.unsqueeze(-1)))
        drive_c, carrier_c, mask_c = apply_carrier_mask_intervention(
            carrier, mask, "carrier_mean", mean_carrier=mean_carrier)
        self.assertTrue(torch.equal(carrier_c, mean_carrier))
        self.assertTrue(torch.equal(mask_c, mask))
        self.assertTrue(torch.equal(drive_c, mean_carrier * mask.unsqueeze(-1)))

    def test_temporal_means_are_per_sample_and_region(self):
        ch = torch.tensor([[[[0.], [2.]], [[2.], [4.]]]])
        mh = torch.tensor([[[0., 2.], [2., 4.]]])
        cm, mm = temporal_means(ch, mh)
        self.assertTrue(torch.equal(cm, torch.tensor([[[1.], [3.]]])))
        self.assertTrue(torch.equal(mm, torch.tensor([[1., 3.]])))

    def test_invalid_permutation_rejected(self):
        with self.assertRaises(ValueError):
            apply_carrier_mask_intervention(torch.zeros(1, 3, 2), torch.zeros(1, 3),
                                            "gate_perm", permutation=[0, 0, 2])

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
