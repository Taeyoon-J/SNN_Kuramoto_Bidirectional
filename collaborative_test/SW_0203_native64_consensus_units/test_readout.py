import unittest
import numpy as np
from .readout import geometric_consensus
class ConsensusTests(unittest.TestCase):
    def test_equal_correlations(self):
        q=np.full((3,3),.7**4);before=q.copy()
        np.testing.assert_allclose(geometric_consensus(q),.7,atol=1e-15)
        np.testing.assert_array_equal(q,before)
    def test_zeros_bounds_order(self):
        q=np.array([[0,.01],[.2,1.]])
        result=geometric_consensus(q)
        self.assertEqual(result[0,0],0);self.assertEqual(result[1,1],1)
        self.assertTrue(np.all(np.diff(result.ravel())>0))
    def test_reject_invalid(self):
        for q in (np.ones((2,3)),np.array([[np.nan]]),np.array([[-.1]]),np.array([[1.1]])):
            with self.assertRaises(ValueError):geometric_consensus(q)
        with self.assertRaises(ValueError):geometric_consensus(np.eye(2),3)
    def test_actual_solver_preserves_support(self):
        from collaborative_test.SW_0164_foreground_preserving_partition.foreground_signed import foreground_induced_merge
        labels=np.array([[1,0],[2,2]],dtype=np.int64)
        result=foreground_induced_merge(geometric_consensus(np.full((4,4),.7**4)),np.zeros((4,4)),1.,labels,grid_shape=(2,2))['labels']
        np.testing.assert_array_equal(result>0,labels>0)
