import unittest
import numpy as np
from .run import errors


class ErrorAttribution(unittest.TestCase):
    def setUp(self):
        self.gt = np.array([[0, 0, 1, 1], [0, 0, 2, 2]])

    def test_perfect_partition_and_arbitrary_large_positive_ids(self):
        pred = np.where(self.gt == 1, 2**40, np.where(self.gt == 2, 2**41, 0))
        r = errors(pred, self.gt)
        for key in ('foreground_miss_rate', 'background_false_positive_rate',
                    'pair_merge_rate', 'pair_split_rate', 'object_balanced_split_rate',
                    'object_pair_balanced_merge_rate'):
            self.assertEqual(r[key], 0.)
        self.assertEqual(r['raw_count_exact'], 1)

    def test_merging_two_objects_has_no_split_error(self):
        pred = (self.gt > 0).astype(int)
        r = errors(pred, self.gt)
        self.assertEqual(r['pair_merge_rate'], 1.)
        self.assertEqual(r['object_pair_balanced_merge_rate'], 1.)
        self.assertEqual(r['pair_split_rate'], 0.)
        self.assertEqual(r['raw_count_exact'], 0)

    def test_splitting_one_object_does_not_create_cross_object_merge(self):
        pred = self.gt.copy();pred[0, 3] = 3
        r = errors(pred, self.gt)
        self.assertEqual(r['pair_split_rate'], .5)
        self.assertEqual(r['object_balanced_split_rate'], .5)
        self.assertEqual(r['pair_merge_rate'], 0.)

    def test_missed_objects_still_merge_in_predicted_background_cluster(self):
        r = errors(np.zeros_like(self.gt), self.gt)
        self.assertEqual(r['foreground_miss_rate'], 1.)
        self.assertEqual(r['pair_merge_rate'], 1.)
        self.assertEqual(r['pair_merge_via_predicted_background_rate'], 1.)
        self.assertEqual(r['pair_split_rate'], 0.)
        self.assertEqual(r['predicted_positive_label_count'], 0)

    def test_background_leak_does_not_change_foreground_partition_errors(self):
        pred = self.gt.copy();pred[self.gt == 0] = 3
        r = errors(pred, self.gt)
        self.assertEqual(r['background_false_positive_rate'], 1.)
        self.assertEqual(r['foreground_miss_rate'], 0.)
        self.assertEqual(r['pair_split_rate'], 0.)
        self.assertEqual(r['pair_merge_rate'], 0.)
        self.assertEqual(r['predicted_positive_label_count'], 3)

    def test_undefined_pair_denominators_are_explicit_and_invalid_labels_rejected(self):
        r = errors(np.zeros((2, 2), int), np.zeros((2, 2), int))
        self.assertIsNone(r['foreground_miss_rate'])
        self.assertIsNone(r['pair_split_rate'])
        self.assertIsNone(r['pair_merge_rate'])
        for pred in (np.zeros((1, 2), int), np.zeros((2, 2), float), -np.ones((2, 2), int)):
            with self.assertRaises(ValueError):
                errors(pred, np.zeros((2, 2), int))


if __name__ == '__main__':
    unittest.main()
