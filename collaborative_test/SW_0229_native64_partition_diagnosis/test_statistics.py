import unittest
import numpy as np
from .statistics import pair_errors, visible_geometry


class DiagnosisTests(unittest.TestCase):
    def test_background_merge_and_support_split_are_counted_separately(self):
        gt = np.array([[0, 1, 1], [0, 2, 2]])
        pred = np.array([[1, 0, 0], [0, 0, 3]])
        r = pair_errors(gt, pred)
        self.assertEqual((r['true_positive_pairs'], r['false_merge_pairs'], r['false_split_pairs']), (1, 2, 1))
        self.assertEqual((r['false_merge_background_pairs'], r['false_merge_foreground_pairs']), (2, 0))
        self.assertEqual((r['false_split_support_pairs'], r['false_split_inside_foreground_pairs']), (1, 0))
        self.assertEqual(r['GT_objects_all_predicted_background'], 1)
        self.assertEqual(r['foreground_precision'], .5); self.assertEqual(r['foreground_recall'], .25)

    def test_fragmentation_inside_foreground_is_not_a_support_failure(self):
        r = pair_errors(np.ones((1, 4), dtype=np.int64), np.array([[1, 1, 2, 2]]))
        self.assertEqual(r['false_merge_pairs'], 0); self.assertEqual(r['false_split_pairs'], 4)
        self.assertEqual(r['false_split_support_pairs'], 0); self.assertEqual(r['false_split_inside_foreground_pairs'], 4)

    def test_diagonal_parts_are_four_disconnected_visible_pieces(self):
        gt = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
        r = visible_geometry(gt)
        self.assertEqual((r['objects'], r['multipart_objects'], r['visible_components'], r['minor_component_patches']), (1, 1, 4, 3))

    def test_positive_ID_renaming_does_not_change_accounting(self):
        gt = np.array([[0, 1, 1], [0, 2, 2]]); pred = np.array([[1, 0, 0], [0, 0, 3]])
        changed_gt = np.where(gt == 1, 7, np.where(gt == 2, 3, 0))
        changed_pred = np.where(pred == 1, 9, np.where(pred == 3, 7, 0))
        self.assertEqual(pair_errors(gt, pred), pair_errors(changed_gt, changed_pred))


if __name__ == '__main__':
    unittest.main()
