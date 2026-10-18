import unittest
import numpy as np
import torch
from .operator import geodesic_matmul
from collaborative_test.SW_0143_native32_selective_geodesic.geodesic_operator import geodesic_distance_power2
from collaborative_test.SW_0135_native32_spike_binding.resolution import grid_distance


class EquivalentSoftDistanceTests(unittest.TestCase):
    def compare(self,dtype,batch=1,steps=3,shared=True,constant=False):
        torch.manual_seed(196)
        raw=torch.randn(batch,25,4,dtype=dtype)
        if constant:raw.fill_(1.)
        z=torch.nn.functional.normalize(raw,dim=-1).detach().requires_grad_()
        euclidean=grid_distance(5,.25,dtype=dtype).requires_grad_()
        if not shared:euclidean=euclidean.unsqueeze(0).repeat(batch,1,1).detach().requires_grad_()
        contrast=torch.tensor(8.,dtype=dtype,requires_grad=True)
        weight=torch.randn(batch,25,25,dtype=dtype)
        reference=geodesic_distance_power2(z,euclidean,contrast,steps=steps,chunk=3)
        original=torch.autograd.grad((reference*weight).sum(),(z,euclidean,contrast))
        actual=geodesic_matmul(z,euclidean,contrast,steps=steps)
        gradients=torch.autograd.grad((actual*weight).sum(),(z,euclidean,contrast))
        value_tol=1e-10 if dtype==torch.float64 else 2e-5
        grad_tol=1e-9 if dtype==torch.float64 else 2e-4
        torch.testing.assert_close(actual,reference,atol=value_tol,rtol=value_tol)
        for measured,expected in zip(gradients,original):
            self.assertTrue(torch.isfinite(measured).all())
            torch.testing.assert_close(measured,expected,atol=grad_tol,rtol=grad_tol)

    def test_float64_values_and_all_three_input_gradients(self):self.compare(torch.float64)
    def test_float32_values_and_all_three_input_gradients(self):self.compare(torch.float32)
    def test_shared_distance_batch_broadcast(self):self.compare(torch.float64,batch=2)
    def test_per_image_distance_batch(self):self.compare(torch.float64,batch=2,shared=False)
    def test_zero_relaxations(self):self.compare(torch.float64,steps=0)
    def test_constant_features_and_tied_distances(self):self.compare(torch.float64,constant=True)

    def test_source_inside_radius_edges_above_cap_are_not_clipped(self):
        # Opposite unit features at distance1.5 produce49.5>cap16. All paths
        # between these nodes exceed the final cap. Product remains finite in64.
        z=torch.tensor([[[1.,0.],[-1.,0.]]],requires_grad=True)
        distance=torch.tensor([[0.,1.5],[1.5,0.]],requires_grad=True)
        contrast=torch.tensor(8.,requires_grad=True)
        reference=geodesic_distance_power2(z,distance,contrast,steps=3,chunk=1)
        actual=geodesic_matmul(z,distance,contrast,steps=3)
        torch.testing.assert_close(actual,reference,atol=2e-5,rtol=2e-5)
        grads=torch.autograd.grad(actual.sum(),(z,distance,contrast))
        self.assertTrue(all(torch.isfinite(value).all() for value in grads))

    def test_reject_unproved_ranges_or_invalid_inputs(self):
        z=torch.ones(1,4,2);distance=grid_distance(2,.25)
        for settings in ({'steps':4},{'steps':True},{'temperature':0.},{'temperature':.001},{'cap':float('inf')}):
            with self.assertRaises(ValueError):geodesic_matmul(z,distance,8.,**settings)
        with self.assertRaises(ValueError):geodesic_matmul(z.half(),distance,8.)
        with self.assertRaises(ValueError):geodesic_matmul(z,distance+1.,8.)
        bad=distance.clone();bad[0,1]=-1.
        with self.assertRaises(ValueError):geodesic_matmul(z,bad,8.)
        bad=z.clone();bad[0,0,0]=float('nan')
        with self.assertRaises(ValueError):geodesic_matmul(bad,distance,8.)


if __name__=='__main__':unittest.main()
