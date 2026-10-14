from __future__ import annotations

import unittest

import numpy as np  # Load before torch on the local Windows environment.
import torch

from collaborative_test.SW_0161_supervised_spike_partition import objective


def probabilities_from_labels(labels: torch.Tensor, slots: int = 11, confidence: float = 0.9):
    b, n = labels.shape
    p = torch.full((b, n, slots), (1.0 - confidence) / (slots - 1), dtype=torch.float32)
    p.scatter_(2, labels[..., None], confidence)
    return p


class PartitionObjectiveTests(unittest.TestCase):
    def test_relation_loss_uses_class_means_then_image_mean(self):
        small_labels = torch.tensor([[0, 1, 1, 2]])
        small_p = torch.tensor([[[.5,.5],[.8,.2],[.2,.8],[.1,.9]]], dtype=torch.float32)
        # Production geometry is fixed at 1024, so expand this known block with
        # an additional BG-only region and independently calculate its classes.
        full_labels = torch.zeros((1, 1024), dtype=torch.int64)
        full_labels[:, :4] = small_labels
        p = torch.full((1, 1024, 11), 1.0 / 11)
        p[:, :4] = torch.nn.functional.pad(small_p, (0, 9))
        # Make each tail row a one-hot BG slot to simplify the exact pair oracle.
        p[:, 4:] = 0
        p[:, 4:, 0] = 1
        loss, detail = objective.balanced_relation_loss(p, full_labels)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(detail["pair_counts"].tolist(), [[1, 2, 3063, 520710]])
        # Independent explicit class means over all unordered production pairs.
        ids = full_labels[0].numpy()
        pi = p[0].numpy()
        ii, jj = np.triu_indices(1024, 1)
        li, lj = ids[ii], ids[jj]
        qv = (pi[ii] * pi[jj]).sum(axis=1)
        pair_masks = (
            (li > 0) & (lj > 0) & (li == lj),
            (li > 0) & (lj > 0) & (li != lj),
            (li > 0) ^ (lj > 0),
            (li == 0) & (lj == 0),
        )
        targets = (1.0, 0.0, 0.0, 1.0)
        class_values = [((qv[m] - target) ** 2).mean() for m, target in zip(pair_masks, targets) if m.any()]
        expected = float(np.mean(class_values))
        self.assertAlmostEqual(float(loss), expected, places=6)

    def test_candidate_is_invariant_to_slot_and_foreground_label_permutations(self):
        labels = torch.zeros((1, 1024), dtype=torch.int64)
        labels[:, :300] = 4
        labels[:, 300:610] = 9
        labels[:, 610:700] = 12
        p = probabilities_from_labels(torch.tensor([[0] * 300 + [1] * 310 + [2] * 90 + [3] * 324]))
        # P target structure and GT label structure are intentionally distinct.
        loss, _ = objective.mask_hungarian_loss(p, labels)
        slot_perm = torch.tensor([7, 4, 9, 2, 1, 0, 3, 5, 6, 8, 10])
        p_perm = p[:, :, slot_perm]
        label_perm = labels.clone()
        label_perm[labels == 4] = 21
        label_perm[labels == 9] = 3
        label_perm[labels == 12] = 44
        loss_perm, _ = objective.mask_hungarian_loss(p_perm, label_perm)
        torch.testing.assert_close(loss, loss_perm, rtol=0, atol=1e-6)

        # GT background is an ordinary matched instance label, not a named slot.
        swapped_bg = labels.clone()
        swapped_bg[labels == 0] = 44
        swapped_bg[labels == 4] = 0
        swapped_bg[labels == 9] = 3
        swapped_bg[labels == 12] = 21
        loss_bg_swap, _ = objective.mask_hungarian_loss(p, swapped_bg)
        torch.testing.assert_close(loss, loss_bg_swap, rtol=0, atol=1e-6)

    def test_candidate_matches_independent_ce_and_all_slot_dice_oracles(self):
        labels = torch.empty((1, 1024), dtype=torch.int64)
        labels[0, :400] = 0
        labels[0, 400:700] = 7
        labels[0, 700:] = 9
        slot_for = {0: 4, 7: 2, 9: 8}
        p = torch.full((1, 1024, 11), 0.03)
        for gt, slot in slot_for.items():
            p[0, labels[0] == gt, slot] = 0.70
        loss, detail = objective.mask_hungarian_loss(p, labels)
        got = {gt: slot for gt, slot in detail["matched_labels_to_slots"][0]}
        self.assertEqual(got, slot_for)

        # Independent numerical oracle: known .7 matched probabilities and
        # all 11 channels, including eight empty output slots.
        expected_ce = -np.log(0.70)
        label_np = labels.numpy()[0]
        p_np = p.numpy()[0]
        dice_losses = []
        for slot in range(11):
            target = np.zeros(1024, dtype=np.float64)
            if slot in slot_for.values():
                gt = next(key for key, value in slot_for.items() if value == slot)
                target[label_np == gt] = 1.0
            pred = p_np[:, slot].astype(np.float64)
            dice_losses.append(1.0 - (2 * np.sum(pred * target) + 1.0) /
                               (np.sum(pred) + np.sum(target) + 1.0))
        expected_dice = float(np.mean(dice_losses))
        self.assertAlmostEqual(float(detail["loss_ce"]), float(expected_ce), places=6)
        self.assertAlmostEqual(float(detail["loss_dice"]), expected_dice, places=6)
        self.assertAlmostEqual(float(loss), float(expected_ce + expected_dice), places=6)

        exact = torch.zeros_like(p)
        for gt, slot in slot_for.items():
            exact[0, labels[0] == gt, slot] = 1.0
        exact_loss, _ = objective.mask_hungarian_loss(exact, labels)
        torch.testing.assert_close(exact_loss, torch.tensor(0.0), rtol=0, atol=1e-7)

    def test_slot_exchangeability_and_empty_slots_with_live_gradient(self):
        target = torch.zeros((1, 1024), dtype=torch.int64)
        target[:, :300] = 0
        target[:, 300:600] = 1
        target[:, 600:] = 2
        logits = torch.randn((1, 1024, 11), generator=torch.Generator().manual_seed(161))
        p = logits.softmax(-1).requires_grad_()
        loss, detail = objective.mask_hungarian_loss(p, target)
        perm = torch.randperm(11, generator=torch.Generator().manual_seed(162))
        permuted, _ = objective.mask_hungarian_loss(p[:, :, perm], target)
        torch.testing.assert_close(loss, permuted, rtol=0, atol=2e-6)
        self.assertEqual(detail["gt_label_counts"], [3])
        loss.backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(float(p.grad.abs().sum()), 0)

    def test_capacity_overflow_and_invalid_inputs_fail_closed(self):
        p = torch.full((1, 1024, 11), 1 / 11)
        labels = torch.arange(1024).reshape(1, 1024)
        with self.assertRaisesRegex(ValueError, "label count"):
            objective.mask_hungarian_loss(p, labels)
        with self.assertRaisesRegex(ValueError, "label count"):
            objective.balanced_relation_loss(p, labels)
        empty_p = torch.empty((0, 1024, 11))
        empty_labels = torch.empty((0, 1024), dtype=torch.long)
        for fn in (objective.mask_hungarian_loss, objective.balanced_relation_loss):
            with self.assertRaisesRegex(ValueError, "nonempty"):
                fn(empty_p, empty_labels)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            objective.canonical_hard_labels(empty_p)
        with self.assertRaises(ValueError):
            objective.mask_hungarian_loss(p[:, :, :10], torch.zeros((1, 1024), dtype=torch.long))
        with self.assertRaises(ValueError):
            objective.mask_hungarian_loss(p * 2, torch.zeros((1, 1024), dtype=torch.long))
        bad = p.clone()
        bad[0, 0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            objective.balanced_relation_loss(bad, torch.zeros((1, 1024), dtype=torch.long))

    def test_hard_remap_largest_ties_small_groups_and_no_named_background_slot(self):
        hard = torch.tensor([[0] * 500 + [1] * 500 + [2] * 7 + [3] * 17])
        p = torch.zeros((1, 1024, 11))
        p.scatter_(2, hard[..., None], 1.0)
        mapped = objective.canonical_hard_labels(p)[0].reshape(-1)
        # Largest tie: first row-major group (slot 0) is BG. Small slot2 is
        # also background; positive IDs follow first-patch order (1 then 3).
        self.assertTrue(torch.equal(mapped[:500], torch.zeros(500, dtype=torch.long)))
        self.assertTrue(torch.equal(mapped[500:1000], torch.ones(500, dtype=torch.long)))
        self.assertTrue(torch.equal(mapped[1000:1007], torch.zeros(7, dtype=torch.long)))
        self.assertTrue(torch.equal(mapped[1007:1024], torch.full((17,), 2, dtype=torch.long)))

        hard_unique = torch.tensor([[0] * 600 + [1] * 300 + [2] * 124])
        unique_p = torch.zeros((1, 1024, 11))
        unique_p.scatter_(2, hard_unique[..., None], 1.0)
        base = objective.canonical_hard_labels(unique_p)
        slot_perm = torch.tensor([7, 3, 9, 0, 1, 2, 4, 5, 6, 8, 10])
        self.assertTrue(torch.equal(base, objective.canonical_hard_labels(unique_p[:, :, slot_perm])))

    def test_relation_class_counts_and_gradient(self):
        labels = torch.zeros((1, 1024), dtype=torch.int64)
        labels[:, :2] = 1
        labels[:, 2:4] = 2
        p = torch.softmax(torch.randn((1, 1024, 11), generator=torch.Generator().manual_seed(163)), -1)
        p.requires_grad_()
        loss, details = objective.balanced_relation_loss(p, labels)
        # sameFG=2 pairs; diffFG=4; FG/BG=4*1020; BG/BG=1020 choose 2.
        self.assertEqual(details["pair_counts"].tolist(), [[2, 4, 4080, 519690]])
        self.assertTrue(torch.isfinite(loss))
        perm = torch.randperm(11, generator=torch.Generator().manual_seed(164))
        permuted_loss, _ = objective.balanced_relation_loss(p[:, :, perm], labels)
        torch.testing.assert_close(loss, permuted_loss, rtol=0, atol=2e-6)
        loss.backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(float(p.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
