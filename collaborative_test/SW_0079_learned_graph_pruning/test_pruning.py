import unittest

import torch

from snn_kuramoto_bidirectional.graph_generator import ImageConditionedGraph
from pruning import adjacency_at_top_k


class LearnedGraphPruningTests(unittest.TestCase):
    def test_topk_is_restored_and_only_adjacency_density_changes(self):
        torch.manual_seed(3)
        graph = ImageConditionedGraph(in_channels=3, hidden_dim=5, top_k=8,
                                      coupling_gain=4.0, temperature=0.2)
        gamma = torch.randn(2, 3, 16)
        before = {key: value.detach().clone() for key, value in graph.state_dict().items()}
        base_a = graph(gamma)
        base_b = adjacency_at_top_k(graph, gamma, 8)
        self.assertTrue(torch.equal(base_a, base_b))
        self.assertEqual(graph.top_k, 8)
        sparse = adjacency_at_top_k(graph, gamma, 4)
        self.assertEqual(graph.top_k, 8)
        self.assertTrue(torch.isfinite(sparse).all())
        self.assertTrue((sparse >= 0).all())
        self.assertTrue(torch.allclose(sparse, sparse.transpose(1, 2), atol=1e-6))
        self.assertTrue(all(torch.equal(before[k], value) for k, value in graph.state_dict().items()))

    def test_topk_restores_when_graph_call_raises(self):
        class BrokenGraph:
            top_k = 8
            def __call__(self, gamma):
                raise RuntimeError("synthetic graph failure")
        graph = BrokenGraph()
        with self.assertRaisesRegex(RuntimeError, "synthetic graph failure"):
            adjacency_at_top_k(graph, torch.ones(1, 2, 4), 2)
        self.assertEqual(graph.top_k, 8)

    def test_invalid_topk_rejected_without_mutation(self):
        graph = ImageConditionedGraph(in_channels=3, hidden_dim=4, top_k=8)
        with self.assertRaisesRegex(ValueError, "top_k"):
            adjacency_at_top_k(graph, torch.randn(1, 3, 16), 17)
        self.assertEqual(graph.top_k, 8)


if __name__ == "__main__":
    unittest.main()
