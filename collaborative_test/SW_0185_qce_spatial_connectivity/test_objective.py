import unittest
import numpy as np
import torch
from collaborative_test.SW_0185_qce_spatial_connectivity import objective as o

class SpatialConnectivity(unittest.TestCase):
    def test_edges_are_unique_unoriented_four_neighbors(self):
        i,j=o.spatial_edges(3,'cpu'); edges=set(zip(i.tolist(),j.tolist()))
        self.assertEqual(len(edges),12)
        self.assertTrue(all((b-a==3 or (b-a==1 and a//3==b//3)) for a,b in edges))

    def test_positive_credit_survives_zero_clamped_product(self):
        rho=torch.full((4,4,4),-.1,requires_grad=True)
        q=rho.clamp_min(0).prod(0)
        loss=o.connectivity_loss(q,rho,torch.zeros(4,dtype=torch.long),grid=2)
        loss.backward()
        i,j=o.spatial_edges(2,'cpu')
        self.assertTrue(bool((rho.grad[:,i,j]<0).all()))

    def test_satisfied_connection_and_boundary_margins_have_zero_loss(self):
        labels=torch.tensor([0,0,7,7]);same=labels[:,None]==labels[None,:]
        rho=torch.where(same[None,:,:],torch.tensor(.95),torch.tensor(.1)).expand(4,-1,-1)
        q=rho.clamp_min(0).prod(0)
        self.assertEqual(float(o.connectivity_loss(q,rho,labels,grid=2)),0.)

    def test_cross_class_bridge_receives_downward_Q_gradient(self):
        labels=torch.tensor([0,1,0,1]); q=torch.full((4,4),.9,requires_grad=True)
        rho=torch.ones(4,4,4,requires_grad=True)
        o.connectivity_loss(q,rho,labels,grid=2).backward()
        self.assertGreater(float(q.grad[0,1]),0.)
        self.assertGreater(float(q.grad[2,3]),0.)

    def test_instance_renaming_including_arbitrary_foreground_IDs(self):
        rho=torch.full((4,4,4),.6);q=rho.prod(0)
        a=o.connectivity_loss(q,rho,torch.tensor([0,0,1,2]),grid=2)
        b=o.connectivity_loss(q,rho,torch.tensor([0,0,9,5]),grid=2)
        self.assertEqual(float(a),float(b))

    def test_registered_component_margin_implies_positive_product_margin(self):
        self.assertAlmostEqual(o.POSITIVE_RHO_MARGIN**4,o.POSITIVE_Q_MARGIN)
        self.assertGreater(o.POSITIVE_Q_MARGIN,.5)
        self.assertLess(o.NEGATIVE_Q_MARGIN,.5)

if __name__=='__main__':unittest.main()
