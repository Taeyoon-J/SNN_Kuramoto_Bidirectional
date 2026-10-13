from __future__ import annotations

import unittest

import numpy as np

from collaborative_test.SW_0154_native64_resolution_compare.native64_model import (
    NODES, OLD_NODES, configure_grid_hparams, convert_state_dict16_to64,
    convert_state_dict16_to_grid, grid_distance,
)
from collaborative_test.SW_0154_native64_resolution_compare.resolution64 import parent_index


class Native64ModelMappingTests(unittest.TestCase):
    def test_candidate_node_state_uses_registered_parent_and_shared_state_is_exact(self):
        import torch

        parent = parent_index().reshape(-1)
        node = torch.arange(OLD_NODES * 2, dtype=torch.float32).reshape(OLD_NODES, 2)
        shared = torch.tensor([2.5, -1.0])
        mapped = convert_state_dict16_to64(
            {"kuramoto.omega": node, "shared.bias": shared},
            {"kuramoto.omega": torch.empty(NODES, 2), "shared.bias": torch.empty(2)},
        )
        torch.testing.assert_close(mapped["kuramoto.omega"], node.index_select(0, torch.from_numpy(parent)))
        torch.testing.assert_close(mapped["shared.bias"], shared)
        self.assertEqual(mapped["kuramoto.omega"].shape, (4096, 2))

    def test_unregistered_shape_change_and_missing_state_are_rejected(self):
        import torch

        with self.assertRaisesRegex(ValueError, "unregistered spatial"):
            convert_state_dict16_to64({"shared": torch.zeros(2)}, {"shared": torch.zeros(3)})
        with self.assertRaisesRegex(ValueError, "keys differ"):
            convert_state_dict16_to64({"a": torch.zeros(1)}, {"b": torch.zeros(1)})

    def test_grid_distance_uses_row_major_euclidean_registered_units(self):
        d = grid_distance(3, 0.25)
        self.assertEqual(d.shape, (9, 9))
        self.assertEqual(float(d[0, 0]), 0.0)
        self.assertAlmostEqual(float(d[0, 1]), 0.25)
        self.assertAlmostEqual(float(d[0, 4]), np.sqrt(2) * 0.25)
        with self.assertRaises(ValueError):
            grid_distance(0)
        with self.assertRaises(ValueError):
            grid_distance(3, 0)

    def test_actual_core_constructor_and_generalized32_map_match_frozen_builder(self):
        from collaborative_test.SW_0135_native32_spike_binding import resolution as frozen32
        from SW_0094_aligned_joint_pilot.run import hparams
        from snn_kuramoto_bidirectional.s2net_cls import S2NetCore

        source = S2NetCore(hparams("raw"), device="cpu")
        candidate_hp = configure_grid_hparams(hparams("raw"), grid_size=32, graph_top_k=128)
        candidate = S2NetCore(candidate_hp, device="cpu")
        source_state = source.state_dict()
        target_template = candidate.state_dict()
        mapped = convert_state_dict16_to_grid(source_state, target_template,
                                             grid_size=32, units_per_cell=0.5)
        reference = frozen32.convert_state_dict(source_state, target_template)
        self.assertEqual(set(mapped), set(reference))
        for key in mapped:
            self.assertTrue(__import__("torch").equal(mapped[key], reference[key]), key)
        candidate.load_state_dict(mapped, strict=True)
        self.assertEqual(candidate.in_dim, 1024)
        self.assertEqual(candidate.graph_generator.top_k, 128)

    def test_candidate_hparams_reject_invalid_resolution_or_topk(self):
        from SW_0094_aligned_joint_pilot.run import hparams

        with self.assertRaises(ValueError):
            configure_grid_hparams(hparams("raw"), grid_size=24, graph_top_k=128)
        with self.assertRaises(ValueError):
            configure_grid_hparams(hparams("raw"), grid_size=32, graph_top_k=1025)


if __name__ == "__main__":
    unittest.main()
