import unittest
import numpy as np
from .readout import resolution_pair_energy,foreground_induced_merge
from collaborative_test.SW_0164_foreground_preserving_partition import foreground_signed as original


class ResolutionReadoutTests(unittest.TestCase):
    def setUp(self):
        self.q=np.array([[1.,.8,.3,.2],[.8,1.,.6,.2],[.3,.6,1.,.8],[.2,.2,.8,1.]])
        self.a=np.array([[0.,2.,0.,0.],[2.,0.,1.,0.],[0.,1.,0.,2.],[0.,0.,2.,0.]])

    def test_frozen1024_energy_and_entire_solver_are_identical(self):
        expected=original.signed.signed_pair_energy(self.q,self.a,8.)
        np.testing.assert_array_equal(resolution_pair_energy(self.q,self.a,8.,coupling_nodes=1024),expected)
        labels=np.array([[1,1],[2,2]])
        a=original.foreground_induced_merge(self.q,self.a,8.,labels,grid_shape=(2,2))
        b=foreground_induced_merge(self.q,self.a,8.,labels,grid_shape=(2,2),coupling_nodes=1024)
        np.testing.assert_array_equal(a['labels'],b['labels']);self.assertEqual(a['merge_log'],b['merge_log'])

    def test_actualN_matches_explicit_formula_and_retains_support(self):
        q=np.clip(self.q,1e-6,1-1e-6);expected=(1+4*self.a/8.)*np.log(q/(1-q))
        np.fill_diagonal(expected,0.)
        np.testing.assert_allclose(resolution_pair_energy(self.q,self.a,8.),expected,rtol=1e-14,atol=1e-14)
        labels=np.array([[1,1],[0,2]])
        result=foreground_induced_merge(self.q,self.a,8.,labels,grid_shape=(2,2))
        np.testing.assert_array_equal(result['labels']>0,labels>0)

    def test_native32_default_is_exact_and_arbitrary_coefficients_fail(self):
        q=np.full((1024,1024),.6);a=np.zeros_like(q)
        np.testing.assert_array_equal(resolution_pair_energy(q,a,8.),original.signed.signed_pair_energy(q,a,8.))
        for value in (True,512,2048):
            with self.assertRaises(ValueError):resolution_pair_energy(self.q,self.a,8.,coupling_nodes=value)

    def test_relative_QCE_can_stay_equal_while_absolute_merge_boundary_splits_objects(self):
        import torch
        from collaborative_test.SW_0179_qce_seed_replication.train import _vectorized_class_loss
        labels=np.array([[1,1],[2,2]],dtype=np.int64)
        same=labels.reshape(-1)[:,None]==labels.reshape(-1)[None,:]
        before=np.where(same,.75,.35);np.fill_diagonal(before,1.)
        after=before.copy();after[~np.eye(4,dtype=bool)]-=.3
        losses=[_vectorized_class_loss(torch.tensor(q,dtype=torch.float64),torch.from_numpy(labels.reshape(-1))) for q in (before,after)]
        torch.testing.assert_close(losses[0],losses[1],atol=1e-14,rtol=1e-14)
        support=np.ones((2,2),dtype=np.int64)
        old=original.foreground_induced_merge(before,self.a,8.,support,grid_shape=(2,2))
        shifted=original.foreground_induced_merge(after,self.a,8.,support,grid_shape=(2,2))
        self.assertEqual(len(old['clusters']),2);self.assertEqual(len(shifted['clusters']),4)


if __name__=='__main__':unittest.main()
