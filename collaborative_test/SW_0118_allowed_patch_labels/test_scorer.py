import numpy as np  # Keep NumPy before torch on Windows.
import unittest

import torch

from collaborative_test.SW_0118_allowed_patch_labels.scorer import score_allowed_patch_labels
from snn_kuramoto_bidirectional.evaluation import patch_fg_ari


def pixels_from_patch_grid(grid):
    return np.repeat(np.repeat(np.asarray(grid, dtype=np.int64), 8, axis=0), 8, axis=1)


class AllowedPatchLabelTests(unittest.TestCase):
    def test_predicted_id_renaming_preserves_oracle_targets_and_score(self):
        gt = np.zeros((16, 16), dtype=np.int64)
        gt[:, :8] = 3
        gt[:, 8:] = 9
        pred_a = np.zeros_like(gt)
        pred_a[:, :8], pred_a[:, 8:] = 2, 7
        pred_b = np.zeros_like(gt)
        pred_b[:, :8], pred_b[:, 8:] = 40, 11

        a = score_allowed_patch_labels(pred_a, pixels_from_patch_grid(gt))
        b = score_allowed_patch_labels(pred_b, pixels_from_patch_grid(gt))
        self.assertEqual(a["resolved_target_labels"], b["resolved_target_labels"])
        self.assertEqual(a["allowed_label_oracle_fg_ari"], b["allowed_label_oracle_fg_ari"])
        self.assertEqual(a["allowed_label_oracle_fg_ari"], 1.0)

    def test_exact_assignment_tie_is_invariant_to_predicted_id_renaming(self):
        # Every patch permits both objects, so the integer matching objective
        # ties. Stable first-patch group order, not arbitrary predicted IDs,
        # determines the same resolved partition under renaming.
        pixels = np.empty((128, 128), dtype=np.int64)
        for row in range(16):
            for col in range(16):
                pixels[row * 8:row * 8 + 8, col * 8:col * 8 + 4] = 1
                pixels[row * 8:row * 8 + 8, col * 8 + 4:(col + 1) * 8] = 2
        pred_a = np.zeros((16, 16), dtype=np.int64)
        pred_a[:, :8], pred_a[:, 8:] = 4, 9
        pred_b = np.zeros((16, 16), dtype=np.int64)
        pred_b[:, :8], pred_b[:, 8:] = 77, 3
        a = score_allowed_patch_labels(pred_a, pixels)
        b = score_allowed_patch_labels(pred_b, pixels)
        self.assertEqual(a["resolved_target_labels"], b["resolved_target_labels"])
        self.assertEqual(a["allowed_label_oracle_fg_ari"], b["allowed_label_oracle_fg_ari"])

    def test_pure_patches_are_unchanged_and_match_existing_fg_ari(self):
        gt = np.zeros((16, 16), dtype=np.int64)
        gt[:, :8], gt[:, 8:] = 1, 2
        pred = np.zeros_like(gt)
        pred[:, :8], pred[:, 8:] = 5, 6
        result = score_allowed_patch_labels(pred, pixels_from_patch_grid(gt))
        resolved = torch.tensor(result["resolved_target_labels"]).reshape(1, 16, 16)
        standard = patch_fg_ari(torch.tensor(pred).unsqueeze(0), resolved)[0].item()
        self.assertEqual(result["resolved_target_labels"], gt.reshape(-1).tolist())
        self.assertEqual(result["changed_gt_patch_count"], 0)
        self.assertEqual(result["all_patch_acceptance_rate"], 1.0)
        self.assertEqual(result["allowed_label_oracle_fg_ari"], standard)

    def test_background_and_mixed_patch_membership_are_resolved_as_specified(self):
        pixels = np.zeros((128, 128), dtype=np.int64)
        pixels[0:8, 0:4] = 1  # allowed set {0,1}; modal tie selects 0.
        pixels[0:8, 8:16] = 1
        pixels[0:8, 16:24] = 1  # predicted background is still forced to target 0.
        pixels[8:16, 0:4] = 1  # second {0,1} patch remains predicted background.
        pred = np.zeros((16, 16), dtype=np.int64)
        pred[0, 0] = 7
        pred[0, 1] = 7
        before = pixels.copy(), pred.copy()

        result = score_allowed_patch_labels(pred, pixels)
        resolved = np.asarray(result["resolved_target_labels"]).reshape(16, 16)
        self.assertEqual(resolved[0, 0], 1)  # mapped ID 1 is allowed in the mixed patch.
        self.assertEqual(resolved[0, 1], 1)
        self.assertEqual(resolved[0, 2], 1)  # disallowed background falls back to modal object ID.
        self.assertEqual(resolved[1, 0], 0)  # background prediction is allowed by {0,1}.
        self.assertEqual(result["all_patch_acceptance_rate"], 255 / 256)
        self.assertTrue(np.array_equal(pixels, before[0]))
        self.assertTrue(np.array_equal(pred, before[1]))

    def test_merges_and_splits_remain_penalized_by_patch_fg_ari(self):
        gt_merge = np.zeros((16, 16), dtype=np.int64)
        gt_merge[:, :8], gt_merge[:, 8:] = 1, 2
        merged = np.ones((16, 16), dtype=np.int64)
        merged_result = score_allowed_patch_labels(merged, pixels_from_patch_grid(gt_merge))
        self.assertLess(merged_result["allowed_label_oracle_fg_ari"], 1.0)

        gt_split = np.ones((16, 16), dtype=np.int64)
        split = np.zeros((16, 16), dtype=np.int64)
        split[:, :8], split[:, 8:] = 3, 8
        split_result = score_allowed_patch_labels(split, pixels_from_patch_grid(gt_split))
        self.assertLess(split_result["allowed_label_oracle_fg_ari"], 1.0)

    def test_unmatched_groups_get_unique_never_allowed_sentinels(self):
        gt = np.ones((16, 16), dtype=np.int64)
        pred = np.zeros((16, 16), dtype=np.int64)
        pred[:, :5], pred[:, 5:10], pred[:, 10:] = 3, 5, 9
        result = score_allowed_patch_labels(pred, pixels_from_patch_grid(gt))
        mappings = result["mapping_counts"]
        sentinel_ids = [row["mapped_gt_id"] for row in mappings if row["mapped_gt_id"] < 0]
        self.assertEqual(len(sentinel_ids), 2)
        self.assertEqual(len(set(sentinel_ids)), 2)
        self.assertTrue(all(value not in {1} for value in sentinel_ids))
        self.assertLess(result["all_patch_acceptance_rate"], 1.0)
        self.assertTrue(all(label == 1 for label in result["resolved_target_labels"]))

    def test_predicted_background_on_pure_object_patches_uses_modal_fallback(self):
        pixels = np.ones((128, 128), dtype=np.int64)
        pred = np.zeros((16, 16), dtype=np.int64)
        result = score_allowed_patch_labels(pred, pixels)
        self.assertEqual(result["all_patch_acceptance_rate"], 0.0)
        self.assertEqual(result["resolved_foreground_patch_count"], 256)
        self.assertEqual(result["resolved_target_labels"], [1] * 256)

    def test_all_background_and_single_foreground_patch_return_json_safe_undefined(self):
        pred = np.zeros((16, 16), dtype=np.int64)
        all_bg = score_allowed_patch_labels(pred, np.zeros((128, 128), dtype=np.int64))
        self.assertFalse(all_bg["score_valid"])
        self.assertIsNone(all_bg["allowed_label_oracle_fg_ari"])

        one_patch_gt = np.zeros((128, 128), dtype=np.int64)
        one_patch_gt[:8, :8] = 4
        one_patch_pred = np.zeros((16, 16), dtype=np.int64)
        one_patch_pred[0, 0] = 2
        one = score_allowed_patch_labels(one_patch_pred, one_patch_gt)
        self.assertEqual(one["resolved_foreground_patch_count"], 1)
        self.assertFalse(one["score_valid"])
        self.assertIsNone(one["allowed_label_oracle_fg_ari"])

    def test_input_shapes_and_integer_contract_are_checked(self):
        with self.assertRaises(ValueError):
            score_allowed_patch_labels(np.zeros((8, 8), dtype=np.int64), np.zeros((128, 128), dtype=np.int64))
        with self.assertRaises(ValueError):
            score_allowed_patch_labels(np.zeros((16, 16), dtype=np.float32), np.zeros((128, 128), dtype=np.int64))
        with self.assertRaises(ValueError):
            score_allowed_patch_labels(np.zeros((16, 16), dtype=np.int64), np.zeros((64, 64), dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
