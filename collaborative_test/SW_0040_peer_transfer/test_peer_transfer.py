"""Local checks for opt-in geodesic graphs and the peer spike readout."""
import sys
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "snn_kuramoto_bidirectional"))

from graph_generator import ImageConditionedGraph
from spike_classifier import spike_synchrony_components
from evaluate import (_count_metrics, compact_sweep, gamma_row_indices,
                      phase_plv_components)
from count_peer import build_peer_evaluator, count_scores


class PeerTransferTest(unittest.TestCase):
    def test_geodesic_off_preserves_keys_and_output(self):
        torch.manual_seed(21)
        default = ImageConditionedGraph(3, hidden_dim=4, top_k=3,
                                        grid_size=2, spatial_decay=0.5)
        torch.manual_seed(21)
        explicit_off = ImageConditionedGraph(3, hidden_dim=4, top_k=3,
                                             grid_size=2, spatial_decay=0.5,
                                             geodesic_steps=0)
        self.assertEqual(set(default.state_dict()), set(explicit_off.state_dict()))
        explicit_off.load_state_dict(default.state_dict(), strict=True)
        gamma = torch.randn(2, 3, 4)
        torch.testing.assert_close(default(gamma), explicit_off(gamma), rtol=0, atol=0)

    def test_geodesic_forward_and_gradient(self):
        graph = ImageConditionedGraph(3, hidden_dim=4, top_k=3,
                                      grid_size=2, spatial_decay=0.5,
                                      geodesic_steps=2)
        gamma = torch.randn(2, 3, 4, requires_grad=True)
        adjacency = graph(gamma)
        self.assertEqual(tuple(adjacency.shape), (2, 4, 4))
        self.assertTrue(torch.isfinite(adjacency).all())
        adjacency.square().sum().backward()
        self.assertIsNotNone(graph.geodesic_contrast.grad)
        self.assertTrue(torch.isfinite(graph.projection.weight.grad).all())

    def test_peer_component_synchrony_groups_product_matched_traces(self):
        traces = torch.tensor([[[1., 0., 1., 0.], [1., 0., 1., 0.],
                                [0., 1., 0., 1.], [0., 1., 0., 1.]]])
        components = traces[:, None].repeat(1, 2, 1, 1)
        groups = spike_synchrony_components(
            traces, synchrony_threshold=0.9, min_group_size=2,
            components=components, background="activity",
            foreground_threshold=0.0,
        )
        self.assertEqual(set(groups[0]), {(0, 1), (2, 3)})

    def test_object_count_metrics_are_post_prediction_diagnostics(self):
        metrics = _count_metrics([1, 2, 3], [1, 4, 2])
        self.assertAlmostEqual(metrics["exact_accuracy"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["mae"], 1.0)
        self.assertAlmostEqual(metrics["bias"], -1.0 / 3.0)
        self.assertAlmostEqual(metrics["within_one_accuracy"], 2.0 / 3.0)

    def test_final_console_summary_matches_cross_target_json(self):
        rows = [{
            "synchrony_threshold": 0.05,
            "scored_targets": {
                "our_hdf5": {"metrics": {"fg_ari": 0.2}, "object_count": {"mae": 1.0}},
                "peer_targets": {"metrics": {"fg_ari": 0.3}, "object_count": {"mae": 2.0}},
            },
        }]
        summary = compact_sweep(rows)
        self.assertEqual(summary[0]["targets"]["peer_targets"]["metrics"]["fg_ari"], 0.3)

    def test_phase_endpoint_uses_thresholded_plv_components(self):
        plv = torch.tensor([[[1.0, .9, .1, .1], [.9, 1.0, .1, .1],
                             [.1, .1, 1.0, .8], [.1, .1, .8, 1.0]]])
        groups = phase_plv_components(plv, threshold=.7, min_group_size=2)
        self.assertEqual(groups, [[(2, 3)]])

    def test_aligned_validation_gamma_global_id_mapping(self):
        self.assertEqual(gamma_row_indices([1320, 1321, 1639], 1320, 320),
                         [0, 1, 319])
        with self.assertRaisesRegex(ValueError, "does not contain"):
            gamma_row_indices([1319], 1320, 320)

    def test_peer_count_claim_statistics(self):
        result = count_scores([2, 4, 5], [2, 3, 7])
        self.assertAlmostEqual(result["exact_accuracy"], 1.0 / 3.0)
        self.assertAlmostEqual(result["mae"], 1.0)
        self.assertAlmostEqual(result["bias"], -1.0 / 3.0)
        self.assertAlmostEqual(result["within_one_accuracy"], 2.0 / 3.0)

    def test_peer_builder_receives_checkpoint_path(self):
        # The peer build_core reads args.checkpoint itself to load strictly.
        class FakeLoader:
            def create_module(self, spec):
                return None

            def exec_module(self, module):
                def build_core(args, device):
                    assert args.checkpoint == "/tmp/peer/core.pt"
                    assert args.geodesic_steps == 3
                    assert args.num_time_steps == 1024
                    assert args.dendrite_per_region is False
                    assert args.membrane_threshold_mode == "absolute"
                    assert args.membrane_threshold_k == 0.5
                    return SimpleNamespace(
                        membrane_layer=SimpleNamespace(vth=0.0))

                module.build_core = build_core
                module.spike_synchrony_components = lambda *a, **kw: None

        spec = importlib.util.spec_from_loader("fake_peer_evaluator", FakeLoader())
        with patch("count_peer.importlib.util.spec_from_file_location",
                   return_value=spec):
            peer, model = build_peer_evaluator(
                "/peer/evaluate_model.py", "/tmp/peer/core.pt", "cpu",
                steps=1024)
        self.assertEqual(model.membrane_layer.vth, 0.0)
        self.assertTrue(callable(peer.spike_synchrony_components))


if __name__ == "__main__":
    unittest.main()
