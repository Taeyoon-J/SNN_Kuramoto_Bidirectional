"""Parity tests against the production spike-QCC pipeline on synthetic spikes."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("MKL_NUM_THREADS","1")
import numpy as np
import torch
torch.set_num_threads(1)
import unittest

from snn_kuramoto_bidirectional.evaluation import spatial_components_to_patch_labels
from snn_kuramoto_bidirectional.spike_classifier import spike_synchrony_affinity, spike_synchrony_components
from collaborative_test.SW_0167_structured_relation_energy.qcc_readout import qcc_labels_from_affinity


class QCCReadoutTests(unittest.TestCase):
    @staticmethod
    def synthetic_components():
        # Four centered ±.25 vectors with unit norm. The seventh patch in the
        # first group has exact rho=.5 in one channel and rho=1 in the other three, so its Q
        # edge to the first seven patches lies exactly on the production cutoff.
        u=torch.tensor([1.]*8+[-1.]*8,dtype=torch.float32)*.25
        w=torch.tensor([1.,-1.]*8,dtype=torch.float32)*.25
        z=torch.tensor([1.,-1.,-1.,1.]*4,dtype=torch.float32)*.25
        v=torch.tensor([1.,-1.,1.,-1.,-1.,1.,-1.,1.]*2,dtype=torch.float32)*.25
        x=u.clone(); x[0],x[1],x[8],x[9]=u[8].clone(),u[9].clone(),u[0].clone(),u[1].clone()
        sig=torch.empty(4,24,16,dtype=torch.float32)
        for d in range(4):
            sig[d,0:7]=u
            sig[d,7]=x if d==0 else u
            sig[d,8:16]=w
            sig[d,16:23]=z
            sig[d,23]=v
        return sig.unsqueeze(0)

    def test_actual_production_affinity_and_qcc_conversion_match_exactly(self):
        components=self.synthetic_components()
        activity=components.sum(dim=1)
        q=spike_synchrony_affinity(activity,components=components,affinity_mode="spike")
        self.assertEqual(tuple(q.shape),(1,24,24))
        self.assertEqual(float(q[0,0,7]),.5)
        prod_groups=spike_synchrony_components(activity,synchrony_threshold=.5,
            min_group_size=8,components=components,background="largest_component")
        prod=spatial_components_to_patch_labels(prod_groups,(4,6),device="cpu")
        got=qcc_labels_from_affinity(q,grid_shape=(4,6),threshold=.5,min_group_size=8)
        torch.testing.assert_close(got,prod,rtol=0,atol=0)
        # Equal largest components resolve by start order: first is BG, second
        # remains; the seven-node group and singleton are dropped by min-size.
        self.assertTrue(torch.all(got[0,0,:]==0))
        self.assertTrue(torch.all(got[0,1,:2]==0))
        self.assertTrue(torch.all(got[0,1,2:]==1))
        self.assertTrue(torch.all(got[0,2,:4]==1))
        self.assertTrue(torch.all(got[0,2,4:]==0))
        self.assertTrue(torch.all(got[0,3,:]==0))

    def test_multiple_batch_and_validation_contract(self):
        q=torch.eye(4).repeat(2,1,1)
        out=qcc_labels_from_affinity(q,grid_shape=(2,2),min_group_size=1)
        self.assertEqual(tuple(out.shape),(2,2,2))
        self.assertEqual(int((out[0]>0).sum()),3) # first singleton BG; other three are retained
        self.assertEqual(int((out[1]>0).sum()),3)
        for bad in (torch.ones(4,4),torch.full((1,4,4),float("nan")),
                    torch.tensor([[[1.,.6],[.4,1.]]])):
            with self.assertRaises(ValueError): qcc_labels_from_affinity(bad,grid_shape=(2,2))
        near=torch.eye(2,dtype=torch.float64).unsqueeze(0)
        near[0,0,1]=near[0,1,0]=.499999999
        with self.assertRaisesRegex(ValueError,"float32"):
            qcc_labels_from_affinity(near,grid_shape=(1,2))


if __name__=="__main__": unittest.main()
