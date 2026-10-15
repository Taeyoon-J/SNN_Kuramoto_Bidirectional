from __future__ import annotations

import math
import sys
from pathlib import Path
import unittest

import numpy as np  # Load NumPy before Torch on Windows to avoid duplicate OpenMP runtimes.
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from collaborative_test.SW_0171_contextual_instance_graph.instance_objective import (
    capture_native_projection,
    leave_one_out_instance_loss,
    normalized_projection_output,
)
from snn_kuramoto_bidirectional.graph_generator import ImageConditionedGraph


class InstanceObjectiveTests(unittest.TestCase):
    def test_known_two_class_loocv_ce_and_gradients(self):
        z = torch.tensor([[[1.0, 0.0], [1.0, 0.0],
                           [0.0, 1.0], [0.0, 1.0]]], requires_grad=True)
        labels = torch.tensor([[0, 0, 9, 9]], dtype=torch.int64)
        loss, stats = leave_one_out_instance_loss(z, labels, return_stats=True)
        expected = math.log1p(math.exp(-10.0))
        self.assertAlmostEqual(float(loss.detach()), expected, places=6)
        self.assertEqual(stats["images"][0]["valid_query_classes"], 2)
        self.assertEqual(stats["images"][0]["valid_queries"], 4)
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(float(z.grad.norm()), 0.0)

    def test_singleton_samples_are_negative_prototypes_but_not_queries(self):
        z = torch.tensor([[[1.0, 0.0], [1.0, 0.0],
                           [0.0, 1.0], [0.0, -1.0]]])
        labels = torch.tensor([[0, 0, 4, 12]])
        loss, stats = leave_one_out_instance_loss(z, labels, return_stats=True)
        expected_with_singleton_negatives = math.log(1.0 + 2.0 * math.exp(-10.0))
        expected_without_singletons = math.log1p(math.exp(-10.0))
        self.assertAlmostEqual(float(loss), expected_with_singleton_negatives, places=7)
        self.assertGreater(float(loss), expected_without_singletons)
        row = stats["images"][0]
        self.assertEqual(row["valid_query_classes"], 1)
        self.assertEqual(row["valid_queries"], 2)
        self.assertEqual(row["excluded_singleton_queries"], 2)

    def test_exchangeable_label_renaming_including_background(self):
        z = F.normalize(torch.tensor([[[1.0, .1, .2], [.9, .2, .1],
                                       [.1, 1.0, .2], [.2, .9, .1],
                                       [.2, .1, 1.0]]]), dim=-1)
        labels = torch.tensor([[0, 0, 5, 5, 17]])
        renamed = torch.tensor([[31, 31, 0, 0, 8]])
        loss = leave_one_out_instance_loss(z, labels)
        renamed_loss = leave_one_out_instance_loss(z, renamed)
        torch.testing.assert_close(loss, renamed_loss, rtol=1e-6, atol=1e-7)

    def test_zero_loss_image_remains_in_batch_mean(self):
        z = torch.tensor([[[1., 0.], [1., 0.], [0., 1.], [0., 1.]],
                          [[1., 0.], [0., 1.], [0., -1.], [-1., 0.]]])
        labels = torch.tensor([[0, 0, 3, 3], [7, 7, 7, 7]])
        batch = leave_one_out_instance_loss(z, labels)
        first = leave_one_out_instance_loss(z[:1], labels[:1])
        self.assertTrue(torch.equal(batch, first / 2))

    def test_insufficient_class_and_all_singleton_exclusion_statistics_are_uniform(self):
        z = F.normalize(torch.tensor([[[1., 0.], [.9, .1], [.8, .2], [.7, .3]],
                                      [[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]]), dim=-1)
        labels = torch.tensor([[5, 5, 5, 5], [0, 3, 7, 11]])
        loss, stats = leave_one_out_instance_loss(z, labels, return_stats=True)
        one_class, all_singletons = stats["images"]
        self.assertTrue(one_class["zero_loss"])
        self.assertEqual(one_class["valid_queries"], 0)
        self.assertEqual(one_class["excluded_queries"], 4)
        self.assertEqual(one_class["excluded_query_classes"], 1)
        self.assertEqual(one_class["excluded_singleton_queries"], 0)
        self.assertEqual(one_class["zero_loss_reason"], "fewer_than_two_visible_classes")
        self.assertTrue(all_singletons["zero_loss"])
        self.assertEqual(all_singletons["valid_queries"], 0)
        self.assertEqual(all_singletons["excluded_queries"], 4)
        self.assertEqual(all_singletons["excluded_query_classes"], 4)
        self.assertEqual(all_singletons["excluded_singleton_queries"], 4)
        self.assertEqual(all_singletons["zero_loss_reason"], "no_class_with_two_or_more_members")
        self.assertEqual(stats["excluded_queries"], 8)
        self.assertEqual(float(loss), 0.0)

    def test_invalid_norm_labels_and_empty_geometry_reject(self):
        with self.assertRaises(FloatingPointError):
            normalized_projection_output(torch.zeros((1, 2, 3)))
        with self.assertRaises(ValueError):
            leave_one_out_instance_loss(torch.zeros((0, 2, 3)), torch.zeros((0, 2), dtype=torch.long))
        with self.assertRaises(ValueError):
            leave_one_out_instance_loss(torch.ones((1, 3, 2)), torch.tensor([[0, 1]]))
        with self.assertRaises(TypeError):
            leave_one_out_instance_loss(torch.ones((1, 3, 2)), torch.tensor([[0., 1., 1.]]))
        with self.assertRaises(ValueError):
            leave_one_out_instance_loss(torch.ones((1, 3, 2)), torch.tensor([[0, -1, 1]]))
        with self.assertRaises(FloatingPointError):
            leave_one_out_instance_loss(torch.zeros((1, 3, 2)), torch.tensor([[0, 0, 1]]))
        with self.assertRaises(ValueError):
            leave_one_out_instance_loss(torch.full((1, 3, 2), 2.0), torch.tensor([[0, 0, 1]]))

    def test_actual_graph_projection_is_captured_once_and_normalized_natively(self):
        graph = ImageConditionedGraph(in_channels=3, hidden_dim=5, top_k=3,
                                      coupling_gain=1.0, geodesic_steps=0)
        gamma = torch.randn((2, 3, 7), requires_grad=True)
        labels = torch.tensor([[0, 0, 1, 1, 2, 2, 3],
                               [0, 0, 1, 1, 2, 2, 3]])
        with capture_native_projection(graph) as captured:
            adjacency = graph(gamma)
            self.assertEqual(len(captured), 1)
            loss = leave_one_out_instance_loss(captured[0], labels)
            total = adjacency.square().mean() + loss
        self.assertEqual(tuple(captured[0].shape), (2, 7, 5))
        expected = F.normalize(graph.projection(gamma.transpose(1, 2)), dim=-1, eps=1e-12)
        self.assertTrue(torch.equal(captured[0], expected))
        total.backward()
        self.assertTrue(torch.isfinite(gamma.grad).all())
        self.assertTrue(torch.isfinite(graph.projection.weight.grad).all())
        self.assertEqual(len(graph.projection._forward_hooks), 0)

    def test_capture_rejects_projection_rerun_and_removes_hook(self):
        graph = ImageConditionedGraph(in_channels=3, hidden_dim=4, top_k=2,
                                      coupling_gain=1.0, geodesic_steps=0)
        gamma = torch.randn((1, 3, 4))
        with self.assertRaises(RuntimeError):
            with capture_native_projection(graph):
                graph.projection(gamma.transpose(1, 2))
                graph.projection(gamma.transpose(1, 2))
        self.assertEqual(len(graph.projection._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
