import numpy as np  # Import before torch on this Windows OpenMP runtime.
import unittest

import torch

from collaborative_test.SW_0115_analytic_partition_rgb.loss import (
    ASSIGNMENT_TEMPERATURE,
    assignment_weights,
    batch_reconstruction_loss,
    production_partition,
    reconstruct_image,
)
from snn_kuramoto_bidirectional.evaluation import spatial_components_to_patch_labels
from snn_kuramoto_bidirectional.spike_classifier import (
    spike_synchrony_affinity,
    spike_synchrony_components,
)


def fixture_activity(seed=7, steps=48):
    """Two coherent objects, a large background, and two dropped singletons."""
    generator = torch.Generator().manual_seed(seed)
    sources = torch.randn(4, steps, generator=generator)
    sources = (sources - sources.mean(dim=-1, keepdim=True))
    activity = torch.empty(2, 256, steps)
    activity[:, :248] = sources[0]
    activity[:, 248:251] = sources[1]
    activity[:, 251:254] = sources[2]
    activity[:, 254] = sources[3]
    activity[:, 255] = -sources[3]
    # Each component has the same coherent event pattern, as in a clean
    # component-product fixture; the classifier and affinity remain production.
    components = activity[:, None].expand(-1, 4, -1, -1).clone()
    # The last pair is anticorrelated in one component only, so its signed
    # product is clamped to zero and both nodes stay omitted.
    components[:, 0, 255] = -components[:, 0, 254]
    return activity, components


def content_fixture(labels):
    rgb = torch.empty(labels.shape[0], 256, 3)
    colors = torch.tensor([
        [0.04, 0.05, 0.06], [0.90, 0.08, 0.07], [0.06, 0.85, 0.10]
    ])
    for b in range(labels.shape[0]):
        rgb[b] = colors[labels[b].clamp(max=2)]
    # Add small deterministic variation so reconstruction has a useful scale.
    variation = torch.linspace(-0.01, 0.01, 256).unsqueeze(-1)
    rgb = (rgb + variation).clamp(0.0, 1.0)
    return rgb


class AnalyticPartitionLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.activity, cls.components = fixture_activity()
        cls.labels, cls.hard = production_partition(cls.activity, cls.components, settle=0)

    def test_partition_matches_exact_production_classifier_and_onehot(self):
        groups = spike_synchrony_components(
            self.activity.detach().cpu(), synchrony_threshold=0.5,
            min_group_size=2, settle=0,
            components=self.components.detach().cpu(),
            background="largest_component", affinity_mode="spike",
            spatial_grid_size=16,
        )
        labels = spatial_components_to_patch_labels(groups, 16).reshape(2, 256)
        self.assertTrue(torch.equal(labels, self.labels))
        self.assertEqual(len(self.hard), 2)
        for row_labels, h in zip(labels, self.hard):
            self.assertTrue(torch.equal(h.argmax(dim=-1), row_labels))
            self.assertTrue(torch.equal(h.sum(dim=-1), torch.ones(256)))
            self.assertEqual(h.shape[1], int(row_labels.max()) + 1)
        # Largest component is label zero; both singleton components are also
        # dropped to that same background column by the production conversion.
        self.assertEqual(int((labels[0] == 0).sum()), 250)
        self.assertEqual(int((labels[0, 254:] == 0).sum()), 2)
        self.assertEqual(int(labels[0].max()), 2)

    def test_exact_hard_forward_and_finite_nonzero_credit_to_actual_components(self):
        components = self.components.clone().requires_grad_(True)
        activity = components.mean(dim=1)
        labels, hard = production_partition(activity, components, settle=0)
        q = spike_synchrony_affinity(activity, components=components, settle=0)
        rgb = content_fixture(labels)
        loss, _, predictions, details = batch_reconstruction_loss(q, hard, rgb)
        for index, (h, info, pred) in enumerate(zip(hard, details, predictions)):
            self.assertTrue(torch.equal(info["W"].detach(), h))
            manual_mu = h.T @ rgb[index]
            manual_mu = manual_mu / h.sum(dim=0).unsqueeze(-1)
            self.assertTrue(torch.equal(pred, h @ manual_mu))
        grad = torch.autograd.grad(loss, components)[0]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0.0)
        self.assertAlmostEqual(ASSIGNMENT_TEMPERATURE, 0.1)

    def test_assignment_matches_registered_count_minus_self_formula(self):
        q = torch.linspace(0.0, 1.0, 256 * 256).reshape(256, 256)
        h = self.hard[0]
        w, p = assignment_weights(q, h)
        eye = torch.eye(256)
        q0 = q * (1.0 - eye)
        count = h.sum(dim=0)
        affinity = (q0 @ h) / (count.unsqueeze(0) - h).clamp_min(1.0)
        expected_p = torch.softmax(affinity / 0.1, dim=-1)
        self.assertTrue(torch.equal(w.detach(), h))
        self.assertTrue(torch.equal(p, expected_p))

    def test_one_group_has_zero_q_credit_and_remains_in_batch_mean(self):
        q0 = torch.rand(256, 256, requires_grad=True)
        q0 = (q0 + q0.T) * 0.5
        q0.retain_grad()
        one_group = torch.ones(256, 1)
        rgb0 = content_fixture(torch.zeros(1, 256, dtype=torch.long))[0]
        single_loss, _, info = reconstruct_image(q0, one_group, rgb0)
        grad_q = torch.autograd.grad(single_loss, q0)[0]
        self.assertTrue(torch.equal(grad_q, torch.zeros_like(grad_q)))
        self.assertTrue(torch.isfinite(single_loss))
        self.assertGreater(float(single_loss.detach()), 0.0)

        q_batch = torch.stack((q0.detach(), torch.zeros_like(q0))).requires_grad_(True)
        blank = torch.zeros_like(rgb0)
        combined, per_image, _, _ = batch_reconstruction_loss(
            q_batch, [one_group, one_group], torch.stack((rgb0, blank))
        )
        self.assertTrue(torch.allclose(combined, per_image.mean(), atol=0, rtol=0))
        self.assertTrue(torch.allclose(per_image[0], single_loss.detach()))
        self.assertEqual(float(per_image[1].detach()), 0.0)

    def test_count_preserving_row_scramble_increases_content_loss(self):
        q = spike_synchrony_affinity(self.activity, components=self.components, settle=0)
        labels = self.labels
        rgb = content_fixture(labels)
        permutation = torch.roll(torch.arange(256), shifts=83)
        deltas = []
        for i in range(2):
            real, _, _ = reconstruct_image(q[i], self.hard[i], rgb[i])
            scrambled_h = self.hard[i].index_select(0, permutation)
            self.assertTrue(torch.equal(scrambled_h.sum(dim=0), self.hard[i].sum(dim=0)))
            scrambled, _, _ = reconstruct_image(q[i], scrambled_h, rgb[i])
            deltas.append(scrambled - real)
        self.assertGreater(float(torch.stack(deltas).mean()), 0.0)


if __name__ == "__main__":
    unittest.main()
