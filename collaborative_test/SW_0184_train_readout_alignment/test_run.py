import unittest
import numpy as np
from collaborative_test.SW_0184_train_readout_alignment import run

class ReadoutAlignment(unittest.TestCase):
    def test_relative_separation_does_not_imply_absolute_QCC_connectivity(self):
        y=np.array([0,0,1,1,2,2]); same=y[:,None]==y[None,:]
        low=np.where(same,.4,.1); high=low+.2
        rho=np.repeat(np.ones_like(low)[None,:,:],4,axis=0)
        a=run.pair_diagnostics(low,rho,y); b=run.pair_diagnostics(high,rho,y)
        self.assertAlmostEqual(a['foreground']['same_q']-a['foreground']['different_q'],
            b['foreground']['same_q']-b['foreground']['different_q'])
        self.assertEqual(a['foreground']['same_threshold_recall'],0.)
        self.assertEqual(b['foreground']['same_threshold_recall'],1.)

    def test_diagonal_does_not_create_singleton_positive_pairs(self):
        q=np.eye(3); rho=np.repeat(q[None,:,:],4,axis=0)
        result=run.pair_diagnostics(q,rho,np.arange(3))
        self.assertEqual(result['all']['valid_classes'],0)
        self.assertEqual(result['foreground']['valid_classes'],0)

    def test_background_scope_and_permutation(self):
        y=np.array([0,0,1,1,2,2]); q=np.where(y[:,None]==y[None,:],.9,.1)
        rho=np.repeat(q[None,:,:],4,axis=0)
        original=run.pair_diagnostics(q,rho,y)
        order=np.array([4,1,3,0,5,2])
        permuted=run.pair_diagnostics(q[order][:,order],rho[:,order][:,:,order],y[order])
        self.assertEqual(original,permuted)
        self.assertEqual(original['all']['valid_classes'],3)
        self.assertEqual(original['foreground']['valid_classes'],2)

    def test_native_geometry_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError):run.pair_diagnostics(np.eye(2),np.zeros((3,2,2)),np.array([1,1]))
        q=np.array([[1,np.nan],[0,1]])
        with self.assertRaises(ValueError):run.pair_diagnostics(q,np.zeros((4,2,2)),np.array([1,1]))

    def test_exact_old_new_cohort_and_distinctness(self):
        self.assertEqual(run.POSITIONS,tuple(range(0,1024,64)))
        self.assertEqual(len(set(run.POSITIONS)),16)
        self.assertEqual(sum(x<512 for x in run.POSITIONS),8)

if __name__=='__main__':unittest.main()
