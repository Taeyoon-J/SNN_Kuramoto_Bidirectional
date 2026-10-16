import numpy as np  # Keep NumPy/OpenMP initialization before torch on Windows.
import torch
import unittest

from collaborative_test.SW_0178_temporal_class_competition.class_competition import (
    leave_one_out_class_loss, native_component_rho_q, temporal_affinity_losses,
)


class TemporalClassCompetitionTests(unittest.TestCase):
    def test_class_balancing_background_singletons_and_hand_oracle(self):
        # Class 0 has two queries; class 1 has three. Class 2 is a singleton
        # negative prototype for all valid queries. Each class contributes
        # equally after averaging its own queries.
        labels = torch.tensor([[0, 0, 1, 1, 1, 2]], dtype=torch.int64)
        affinity = torch.tensor([[[1., .8, .2, .1, .3, -.1],
                                  [.8, 1., .4, .3, .2, .0],
                                  [.2, .4, 1., .7, .5, .2],
                                  [.1, .3, .7, 1., .9, -.2],
                                  [.3, .2, .5, .9, 1., .1],
                                  [-.1, .0, .2, -.2, .1, 1.]]])
        actual = leave_one_out_class_loss(affinity, labels, temperature=.1)
        # Independent literal implementation of the stated prototype and
        # class/query averaging rule.
        losses = []
        for query_class in (0, 1):
            per_query = []
            for i in torch.where(labels[0] == query_class)[0].tolist():
                prototypes, ids = [], []
                for c in (0, 1, 2):
                    idx = torch.where(labels[0] == c)[0]
                    if c == query_class:
                        idx = idx[idx != i]
                    if idx.numel():
                        prototypes.append(affinity[0, i, idx].mean())
                        ids.append(c)
                pos = ids.index(query_class)
                per_query.append(torch.nn.functional.cross_entropy(
                    torch.stack(prototypes)[None] / .1, torch.tensor([pos])))
            losses.append(torch.stack(per_query).mean())
        expected = torch.stack(losses).mean()
        self.assertAlmostEqual(float(actual), float(expected), places=7)

    def test_singleton_queries_and_single_class_images_are_zero(self):
        affinity = torch.eye(4)[None].requires_grad_(True)
        # Every query class is singleton, including background: no valid own
        # leave-one-out prototype exists, but zero stays connected to autograd.
        labels = torch.tensor([[0, 1, 2, 3]])
        loss = leave_one_out_class_loss(affinity, labels)
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(affinity.grad)
        self.assertEqual(float(affinity.grad.abs().sum()), 0.0)

        labels2 = torch.zeros((1, 4), dtype=torch.int64)
        self.assertEqual(float(leave_one_out_class_loss(affinity.detach(), labels2)), 0.0)

    def test_label_id_permutation_does_not_change_loss(self):
        labels = torch.tensor([[0, 0, 1, 1, 2, 2]])
        affinity = torch.tensor([[[1., .8, .2, .3, -.1, .0],
                                  [.8, 1., .4, .2, .1, -.2],
                                  [.2, .4, 1., .7, .3, .2],
                                  [.3, .2, .7, 1., .6, .4],
                                  [-.1, .1, .3, .6, 1., .9],
                                  [.0, -.2, .2, .4, .9, 1.]]])
        remapped = torch.tensor([[17, 17, 4, 4, 0, 0]])
        self.assertTrue(torch.allclose(leave_one_out_class_loss(affinity, labels),
                                       leave_one_out_class_loss(affinity, remapped),
                                       atol=1e-7, rtol=0))

    def test_native_q_order_and_signed_rho_gradient_survive_q_deadzone(self):
        # Four components have one negative correlation. Production Q is zero,
        # while the signed mean remains differentiable with respect to emitted S.
        a1 = torch.tensor([1., 0., 1., 0., 0., 0.])
        a2 = torch.tensor([1., 0., .8, .2, 0., 0.])
        b1 = torch.tensor([.9, .1, 1., 0., .1, 0.])
        b2 = torch.tensor([.8, 0., .9, .2, 0., .1])
        positive = torch.stack([a1, a2, b1, b2])
        signed = torch.stack([a1, a2, -b1, -b2])
        components = torch.stack([positive, signed, positive, positive])[None].requires_grad_(True)
        spikes = components.mean(dim=1)
        rho, q = native_component_rho_q(spikes, components, settle=0)
        self.assertLess(float(rho.detach()[0, 1, 0, 2]), -0.99)
        self.assertEqual(float(q.detach()[0, 0, 2]), 0.0)
        labels = torch.tensor([[0, 0, 1, 1]])
        mean_loss = leave_one_out_class_loss(rho.mean(dim=1), labels)
        self.assertTrue(torch.isfinite(mean_loss))
        mean_grad, = torch.autograd.grad(rho.mean(dim=1)[0, 0, 2], components, retain_graph=True)
        q_grad, = torch.autograd.grad(q[0, 0, 2], components, retain_graph=True)
        self.assertTrue(bool(torch.isfinite(mean_grad).all()))
        self.assertGreater(float(mean_grad.abs().sum()), 0.0)
        self.assertEqual(float(q_grad.abs().sum()), 0.0)

    def test_shape_and_label_contracts_are_strict(self):
        with self.assertRaisesRegex(ValueError, "flatten modal grids explicitly"):
            leave_one_out_class_loss(torch.eye(4)[None], torch.zeros((1, 2, 2), dtype=torch.long))
        with self.assertRaisesRegex(TypeError, "integer dtype"):
            leave_one_out_class_loss(torch.eye(4)[None], torch.zeros((1, 4), dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, "finite"):
            leave_one_out_class_loss(torch.full((1, 4, 4), float("nan")), torch.zeros((1, 4), dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
