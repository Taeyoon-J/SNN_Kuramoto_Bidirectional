import unittest

import torch

from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity
from collaborative_test.SW_0089_gate_affinity_anchor.arms import ARMS
from collaborative_test.SW_0089_gate_affinity_anchor.train_gate_affinity import (
    component_gate_affinity,
)


class ContractTests(unittest.TestCase):
    def test_anchor_matches_fixed_classifier_affinity(self):
        generator = torch.Generator().manual_seed(89)
        components = torch.randn(2, 4, 7, 13, generator=generator)
        activity = components.sum(dim=1)
        expected = spike_synchrony_affinity(
            activity, components=components, settle=3, affinity_mode="spike")
        actual = component_gate_affinity(components, settle=3)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_registered_weights_are_coarse_and_reconstruction_is_fixed(self):
        self.assertEqual(set(ARMS), {"H", "I", "J"})
        self.assertEqual([ARMS[key]["gate_affinity_anchor_weight"] for key in "HIJ"],
                         [1.0, 10.0, 100.0])
        self.assertEqual({ARMS[key]["reconstruction_weight"] for key in ARMS}, {0.3})


if __name__ == "__main__":
    unittest.main()
