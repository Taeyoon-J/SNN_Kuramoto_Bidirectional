import unittest
import numpy as np
from .aggregate import paired


class PairedTests(unittest.TestCase):
    def test_constant_delta_and_zero(self):
        result=paired(np.tile([.02,0.,-.01],(16,1)))
        for key,value in zip(('fg_ari','foreground_iou','matched_object_iou'),(.02,0.,-.01)):
            self.assertAlmostEqual(result[key]['mean'],value)
            np.testing.assert_allclose(result[key]['ci95'],[value,value],rtol=0,atol=1e-15)

    def test_reject_wrong_cohort_and_nonfinite(self):
        for value in (np.zeros((8,3)),np.zeros((16,2)),np.full((16,3),np.nan)):
            with self.assertRaises(ValueError):paired(value)
