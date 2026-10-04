"""Focused CPU checks for the opt-in per-region dendritic projection."""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "snn_kuramoto_bidirectional"))

from dendric_layer import DendricLayer
from hyperparameter import S2NetHyperparameters
from s2net_cls import S2NetCore


class DendriticProjectionTest(unittest.TestCase):
    def test_identical_seed_initialization_equivalence_and_gradients(self):
        torch.manual_seed(1234)
        shared = DendricLayer(5, 5, low_n=-4, high_n=0, branch=4,
                              input_vector_dim=1, dendritic_projection="shared")
        torch.manual_seed(1234)
        regional = DendricLayer(5, 5, low_n=-4, high_n=0, branch=4,
                                input_vector_dim=1, dendritic_projection="per_region")
        self.assertTrue(torch.equal(shared.tau_n, regional.tau_n))
        self.assertTrue(torch.equal(shared.oscillator_dense.weight,
                                     regional.oscillator_dense_weight[0]))
        torch.testing.assert_close(shared.oscillator_dense.bias,
                                   regional.oscillator_dense_bias, rtol=0, atol=0)

        wave = torch.randn(2, 5, 1)
        spike = torch.randn(2, 5)
        shared.set_neuron_state(2)
        regional.set_neuron_state(2)
        expected = shared(wave, spike)
        actual = regional(wave, spike)
        # Shared Linear and the batched per-region form use different GEMM
        # shapes, so CPU BLAS implementations may differ by one float32 ULP.
        torch.testing.assert_close(expected, actual, rtol=2e-6, atol=2e-7)
        actual.square().mean().backward()
        self.assertIsNotNone(regional.oscillator_dense_weight.grad)
        self.assertGreater(regional.oscillator_dense_weight.grad.abs().sum().item(), 0)
        self.assertIsNotNone(regional.oscillator_dense_bias.grad)

    def test_expected_component_mode_snn_parameter_counts(self):
        def count(mode):
            hp = S2NetHyperparameters(
                num_feature_maps=8, num_regions=256, sc=None, osc_dim=4,
                gamma_drive_mode="static", num_time_steps=64,
                spike_per_component=True, branch=4,
                dendritic_projection=mode,
            ).validate()
            core = S2NetCore(hp, device="cpu")
            return sum(p.numel() for name, p in core.named_parameters()
                       if name.startswith(("dendric_layer.", "membrane_layer.")))

        self.assertEqual(count("shared"), 1292)
        self.assertEqual(count("per_region"), 3332)

    def test_default_state_dict_compatibility_and_strict_roundtrips(self):
        hp_default = S2NetHyperparameters(num_regions=8, num_feature_maps=2,
                                          sc=None, osc_dim=2)
        hp_shared = S2NetHyperparameters(num_regions=8, num_feature_maps=2,
                                         sc=None, osc_dim=2,
                                         dendritic_projection="shared")
        torch.manual_seed(10)
        default = S2NetCore(hp_default, device="cpu")
        torch.manual_seed(11)
        explicit_shared = S2NetCore(hp_shared, device="cpu")
        self.assertEqual(set(default.state_dict()), set(explicit_shared.state_dict()))
        explicit_shared.load_state_dict(default.state_dict(), strict=True)

        regional_hp = S2NetHyperparameters(num_regions=8, num_feature_maps=2,
                                           sc=None, osc_dim=2,
                                           dendritic_projection="per_region")
        regional = S2NetCore(regional_hp, device="cpu")
        regional.load_state_dict(regional.state_dict(), strict=True)

    def test_invalid_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "dendritic_projection"):
            S2NetHyperparameters(dendritic_projection="regional").validate()


if __name__ == "__main__":
    unittest.main()
