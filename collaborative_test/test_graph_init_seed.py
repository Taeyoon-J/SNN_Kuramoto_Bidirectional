import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "snn_kuramoto_bidirectional")]

from snn_kuramoto_bidirectional.hyperparameter import S2NetHyperparameters
from snn_kuramoto_bidirectional.s2net_cls import S2NetCore
from snn_kuramoto_bidirectional.training.train_s2net_core import reset_graph_initialization


class GraphInitSeedTests(unittest.TestCase):
    def hparams(self):
        return S2NetHyperparameters(
            num_feature_maps=8, num_regions=4, num_time_steps=4, osc_dim=2,
            sc=None, graph_mode="learned", graph_top_k=2,
            spike_spatial_grid_size=(2, 2), gamma_drive_mode="static",
            gamma_phase_mode="standardize_tanh", spike_per_component=True,
            gate_mode="raw",
        ).validate()

    def test_only_graph_is_reset_and_rng_is_preserved(self):
        hp = self.hparams()
        torch.manual_seed(2)
        core = S2NetCore(hp, device="cpu")
        non_graph = {k: v.clone() for k, v in core.state_dict().items()
                     if not k.startswith("graph_generator.")}
        before_rng = torch.random.get_rng_state().clone()
        reset_graph_initialization(core, hp, "cpu", 0)
        self.assertTrue(torch.equal(before_rng, torch.random.get_rng_state()))
        for key, value in non_graph.items():
            self.assertTrue(torch.equal(value, core.state_dict()[key]), key)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            expected = S2NetCore(hp, device="cpu")
        for key, value in expected.graph_generator.state_dict().items():
            self.assertTrue(torch.equal(value, core.graph_generator.state_dict()[key]), key)

    def test_static_graph_rejected(self):
        hp = self.hparams()
        hp.graph_mode = "static"
        hp.sc = torch.eye(4)
        core = S2NetCore(hp, device="cpu")
        with self.assertRaises(ValueError):
            reset_graph_initialization(core, hp, "cpu", 0)


if __name__ == "__main__":
    unittest.main()
