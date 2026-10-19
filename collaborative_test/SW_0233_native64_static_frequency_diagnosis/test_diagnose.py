import unittest
import numpy as np
from .diagnose import decomposition


class DecompositionTests(unittest.TestCase):
    def test_static_cancellation_is_covariance_not_negative_variance(self):
        omega=np.arange(4096*4,dtype=float).reshape(4096,4)/4096
        x=np.zeros((1,4096,4));gt=np.zeros((1,64,64),np.int64)
        result,error=decomposition(x,omega,2.,gt)
        self.assertLess(error,1e-12);self.assertEqual(result['effective_within'],0.)
        self.assertGreater(result['static_within'],0.)
        self.assertAlmostEqual(result['static_within'],result['sensory_within'])
        self.assertLess(result['twice_cross_covariance'],0.)

    def test_constant_static_offset_has_zero_within_variance(self):
        x=np.random.RandomState(233).normal(size=(2,4096,4));omega=np.ones((4096,4));gt=np.zeros((2,64,64),np.int64)
        result,error=decomposition(x,omega,2.,gt)
        self.assertLess(error,1e-12);self.assertLess(result['static_within'],1e-25)
        self.assertAlmostEqual(result['effective_within'],result['sensory_within'])

    def test_nonfinite_input_rejected(self):
        with self.assertRaises(ValueError):decomposition(np.full((1,4096,4),np.nan),np.zeros((4096,4)),2.,np.zeros((1,64,64),np.int64))


if __name__=='__main__':unittest.main()
