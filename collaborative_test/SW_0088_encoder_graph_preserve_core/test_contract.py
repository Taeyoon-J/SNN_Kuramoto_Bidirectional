import unittest

import torch
from torch import nn

from arms import ARMS
from train_encoder_graph import configure_encoder_graph_only


class TinyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.graph_generator = nn.Linear(3, 3)
        self.downstream = nn.Linear(3, 3)


class ContractTests(unittest.TestCase):
    def test_only_graph_is_trainable(self):
        core = TinyCore()
        selected = configure_encoder_graph_only(core)
        self.assertEqual(selected, list(core.graph_generator.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in core.graph_generator.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in core.downstream.parameters()))

    def test_registered_arms_are_coarse_reconstruction_controls(self):
        self.assertEqual(set(ARMS), {"E", "F", "G"})
        self.assertEqual([ARMS[name]["reconstruction_weight"] for name in "EFG"],
                         [0.0, 0.3, 1.0])


if __name__ == "__main__":
    unittest.main()
