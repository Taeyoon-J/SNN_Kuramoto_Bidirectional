from __future__ import annotations

import unittest

import torch

from collaborative_test.SW_0155_supervised_spike_relations import objective


class SupervisedRelationObjectiveTests(unittest.TestCase):
    def test_source_prefix_guard_requires_exact_prefix_and_rejects_reserved_ids(self):
        registered = list(range(2000, 2000 + 4096))
        self.assertEqual(objective.validate_training_prefix(registered[:2048], registered), registered[:2048])
        with self.assertRaisesRegex(ValueError, "exactly equal"):
            objective.validate_training_prefix(registered[1:2049], registered)
        contaminated = registered[:2047] + [1320]
        with self.assertRaisesRegex(ValueError, "reserved"):
            objective.validate_training_prefix(contaminated, contaminated)

    def test_mask_targets_use_production_modal_4x4_and_keep_mixed_background_patch(self):
        masks = torch.zeros((1, 128, 128), dtype=torch.int64)
        masks[0, 0:2, 0:4] = 1  # exact tie with BG; production smallest ID is zero
        masks[0, 4:8, 4:8] = 3
        labels = objective.modal_train_targets32(masks)
        self.assertEqual(labels.shape, (1, 1024))
        self.assertEqual(int(labels[0, 0]), 0)
        self.assertEqual(int(labels[0, 33]), 3)
        with self.assertRaises(ValueError):
            objective.modal_train_targets32(masks.float())
        masks[0, 0, 0] = -1
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            objective.modal_train_targets32(masks)

    def test_actual_q_matches_production_affinity_and_signed_rho_is_live(self):
        torch.manual_seed(155)
        components = torch.rand(2, 4, 6, 10, requires_grad=True)
        trace = {"spikes": components.mean(dim=1), "component_spikes": components}
        q, rho = objective.production_q_and_signed_rho(trace, settle=2)
        reference = objective.spike_synchrony_affinity(
            trace["spikes"], components, settle=2, affinity_mode="spike")
        self.assertTrue(torch.equal(q, reference))
        self.assertEqual(tuple(rho.shape), (2, 4, 6, 6))
        self.assertTrue(torch.isfinite(rho).all())
        rho.square().mean().backward()
        self.assertTrue(torch.isfinite(components.grad).all())

    def test_class_balancing_uses_equal_nonempty_class_means_and_positive_credit(self):
        labels = torch.tensor([[0, 1, 1, 2]], dtype=torch.int64)
        q = torch.zeros((1, 4, 4), dtype=torch.float32)
        pairs = {(0, 1): 0.8, (0, 2): 0.4, (0, 3): 0.2,
                 (1, 2): 0.4, (1, 3): 0.2, (2, 3): 0.6}
        for (i, j), value in pairs.items():
            q[0, i, j] = q[0, j, i] = value
        rho = torch.zeros((1, 4, 4, 4), dtype=torch.float32)
        # Same-FG rho=-.5 gets squared signed error (1.5)^2, not clamped-away credit.
        rho[0, :, 1, 2] = rho[0, :, 2, 1] = -0.5
        loss, detail = objective.balanced_relation_objective(q, rho, labels)
        torch.testing.assert_close(detail["pair_counts"], torch.tensor([[1, 2, 3, 0]]))
        torch.testing.assert_close(detail["loss_q"], torch.tensor(0.28))
        torch.testing.assert_close(detail["loss_positive"], torch.tensor(2.25))
        torch.testing.assert_close(loss, torch.tensor(2.53))

    def test_positive_credit_backpropagates_through_negative_signed_same_fg_rho(self):
        labels = torch.tensor([[0, 1, 1]], dtype=torch.int64)
        q = torch.zeros((1, 3, 3), requires_grad=True)
        rho = torch.zeros((1, 4, 3, 3), requires_grad=True)
        with torch.no_grad():
            rho[0, :, 1, 2] = -0.4
        loss, detail = objective.balanced_relation_objective(q, rho, labels)
        detail["loss_positive"].backward()
        self.assertTrue(torch.isfinite(rho.grad).all())
        self.assertGreater(float(rho.grad[0, :, 1, 2].abs().sum()), 0.0)
        self.assertTrue(torch.equal(rho.grad[0, :, 0, 1], torch.zeros(4)))

    def test_empty_pair_classes_are_excluded_from_mean_and_mixed_batches_backpropagate(self):
        labels = torch.tensor([[0, 1, 1], [0, 0, 0]], dtype=torch.int64)
        q = torch.full((2, 3, 3), 0.5, requires_grad=True)
        rho = torch.full((2, 4, 3, 3), 0.25, requires_grad=True)
        loss, detail = objective.balanced_relation_objective(q, rho, labels)
        self.assertEqual(detail["pair_counts"].shape, (2, 4))
        self.assertTrue(torch.equal(detail["pair_counts"][0], torch.tensor([1, 0, 2, 0])))
        self.assertTrue(torch.equal(detail["pair_counts"][1], torch.tensor([0, 0, 0, 3])))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertTrue(torch.isfinite(rho.grad).all())

    def test_trace_and_label_shape_or_range_errors_fail_closed(self):
        q = torch.zeros((1, 3, 3))
        rho = torch.zeros((1, 4, 3, 3))
        with self.assertRaises(ValueError):
            objective.balanced_relation_objective(q, rho, torch.tensor([[0, 1]]))
        with self.assertRaises(ValueError):
            objective.balanced_relation_objective(q + 2, rho, torch.tensor([[0, 1, 2]]))
        trace = {"spikes": torch.zeros(1, 3, 4),
                 "component_spikes": torch.zeros(1, 3, 3, 4)}
        with self.assertRaises(ValueError):
            objective.production_q_and_signed_rho(trace, settle=1)


if __name__ == "__main__":
    unittest.main()
