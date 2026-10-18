import unittest
import numpy as np
from .diagnose_partition import analyze_masks


class FrozenPartitionDiagnosis(unittest.TestCase):
    def setUp(self):self.gt=np.array([[[0,0,1,1],[0,0,2,2]]])

    def test_merge_without_support_change_is_identified(self):
        candidate=(self.gt>0).astype(np.int64)
        result=analyze_masks(self.gt,candidate,self.gt)
        for scope in ('full_GT_foreground','common_retained_GT_foreground'):
            self.assertEqual(result[scope]['pair_merge_rate']['delta'],1.)
            self.assertEqual(result[scope]['pair_merge_rate']['ci95'],[1.,1.])
            self.assertEqual(result[scope]['pair_split_rate']['delta'],0.)
        self.assertEqual(result['common_retained_foreground_fraction_mean'],1.)
        self.assertFalse(result['causal_claim'])

    def test_foreground_miss_does_not_fake_a_common_support_split(self):
        candidate=self.gt.copy();candidate[0,1,2:]=0
        result=analyze_masks(self.gt,candidate,self.gt)
        self.assertEqual(result['full_GT_foreground']['foreground_miss_rate']['after'],.5)
        self.assertEqual(result['common_retained_GT_foreground']['pair_split_rate']['delta'],0.)
        self.assertEqual(result['common_retained_GT_foreground']['pair_merge_rate']['defined_images'],0)
        self.assertEqual(result['common_retained_foreground_fraction_mean'],.5)

    def test_label_permutation_is_invariant_and_invalid_batches_fail(self):
        pred=np.where(self.gt>0,self.gt+5,0)
        result=analyze_masks(self.gt,pred,self.gt)
        self.assertEqual(result['full_GT_foreground']['pair_merge_rate']['delta'],0.)
        self.assertEqual(result['full_GT_foreground']['pair_split_rate']['delta'],0.)
        with self.assertRaises(ValueError):analyze_masks(self.gt,pred[:,:1],self.gt)


if __name__=='__main__':unittest.main()
