import unittest
import numpy as np
from .aggregate import paired


class PairedTests(unittest.TestCase):
    def test_constant_paired_difference(self):
        result=paired(np.tile([.02,0.,-.01],(8,1)))
        for key,value in zip(('fg_ari','foreground_iou','matched_object_iou'),(.02,0.,-.01)):
            self.assertAlmostEqual(result[key]['mean'],value)
            np.testing.assert_allclose(result[key]['ci95'],[value,value],rtol=0,atol=1e-15)

    def test_reject_wrong_cohort_or_nonfinite(self):
        for delta in (np.zeros((7,3)),np.zeros((8,2)),np.full((8,3),np.nan)):
            with self.assertRaises(ValueError):paired(delta)
