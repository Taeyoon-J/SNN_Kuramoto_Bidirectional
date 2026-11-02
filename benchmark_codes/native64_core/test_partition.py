"""Small numerical oracles for the resolution-dependent energy coefficient."""
import sys
import unittest
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'unsupervised_readout_benchmark/snapshot/python'))
from partition import foreground_induced_merge


class PartitionOracle(unittest.TestCase):
    def test_native_node_coefficient(self):
        q=np.full((4,4),.75)
        a=np.full((4,4),.2)
        result=foreground_induced_merge(q,a,2.,np.ones((2,2),dtype=np.int64),grid_shape=(2,2))
        expected=(1+4*(a+a.T)/(2*2.))*np.log(q/(1-q))
        np.fill_diagonal(expected,0)
        np.testing.assert_allclose(result['pair_energy'],expected,rtol=1e-14)
        self.assertEqual(len(np.unique(result['labels'])),1)

    def test_negative_evidence_does_not_merge(self):
        result=foreground_induced_merge(np.full((4,4),.25),np.ones((4,4)),2.,
            np.ones((2,2),dtype=np.int64),grid_shape=(2,2))
        self.assertEqual(len(np.unique(result['labels'])),4)


if __name__=='__main__':unittest.main()
