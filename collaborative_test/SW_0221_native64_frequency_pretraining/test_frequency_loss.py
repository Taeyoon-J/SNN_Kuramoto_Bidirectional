import math,unittest
from types import SimpleNamespace
import numpy as np
import torch
from .frequency_loss import make_targets,effective_frequency,frequency_objective
from snn_kuramoto_bidirectional.gamma_initializer import GammaToDrive


class FrequencyLossTests(unittest.TestCase):
    def test_compact_distinct_classes_zero_and_collapsed_classes_penalized(self):
        target=make_targets(np.array([[[1,1],[2,2]]]))
        correct=torch.tensor([[[-1.,-1.],[-1.,-1.],[1.,1.],[1.,1.]]])
        loss,_=frequency_objective(correct,target);self.assertEqual(float(loss),0)
        collapsed=torch.zeros_like(correct,requires_grad=True);bad,_=frequency_objective(collapsed,target)
        self.assertEqual(float(bad.detach()),1.)

    def test_label_permutation_and_padding_do_not_change_loss(self):
        a=np.array([[[0,0],[1,7]]]);b=np.array([[[9,9],[3,0]]])
        x=torch.tensor([[[-.4,.3],[.2,.1],[.3,-.5],[.5,.1]]],dtype=torch.float64)
        first=frequency_objective(x,make_targets(a))[0];second=frequency_objective(x,make_targets(b))[0]
        self.assertAlmostEqual(float(first),float(second),places=12)
        target=make_targets(np.concatenate((a,np.zeros_like(a))))
        self.assertEqual(target['membership'].shape[1],3)
        total,_=frequency_objective(torch.cat((x,torch.zeros_like(x))),target)
        self.assertAlmostEqual(float(total),float(first)/2,places=12)

    def test_objective_gradient_matches_finite_difference(self):
        target=make_targets(np.array([[[0,0],[1,1]]]))
        x=torch.tensor([[[-.2,.1],[-.1,.15],[.25,-.1],[.2,-.15]]],dtype=torch.float64,requires_grad=True)
        loss,_=frequency_objective(x,target);gradient=torch.autograd.grad(loss,x)[0]
        step=1e-6;plus=x.detach().clone();minus=plus.clone();plus[0,0,0]+=step;minus[0,0,0]-=step
        numeric=(frequency_objective(plus,target)[0]-frequency_objective(minus,target)[0])/(2*step)
        self.assertAlmostEqual(float(gradient[0,0,0]),float(numeric),places=7)

    def test_actual_drive_projection_standardization_and_frequency_modulation(self):
        projection=torch.nn.Linear(3,2,bias=False).double();projection.weight.data.copy_(torch.tensor([[1.,0.,0.],[0.,1.,0.]]))
        core=SimpleNamespace(gamma_to_drive=GammaToDrive(3,4,2,'static','standardize_tanh'),gamma_channel_proj=projection,
            gamma_phase_gain=torch.tensor(1.,dtype=torch.float64),kuramoto=SimpleNamespace(freq_gain=torch.tensor(2.,dtype=torch.float64),omega=torch.full((4,2),.2,dtype=torch.float64)))
        gamma=torch.arange(12,dtype=torch.float64).reshape(1,3,4).requires_grad_()
        actual=effective_frequency(core,gamma);drive=core.gamma_to_drive(gamma,projection,core.gamma_phase_gain)
        torch.testing.assert_close(actual,(.2+2*drive)/(2*math.pi),rtol=0,atol=0)
        target=make_targets(np.array([[[0,0],[1,1]]]))
        frequency_objective(actual,target)[0].backward();self.assertTrue(torch.isfinite(gamma.grad).all());self.assertGreater(float(gamma.grad.abs().sum()),0)

    def test_only_spatially_adjacent_classes_are_repulsed(self):
        target=make_targets(np.array([[[1,0,2]]]))
        # Unique classes are0,1,2; object1/object2 do not touch.
        self.assertTrue(target['adjacent'][0,0,1]);self.assertTrue(target['adjacent'][0,0,2]);self.assertFalse(target['adjacent'][0,1,2])

    def test_invalid_targets_and_single_class(self):
        with self.assertRaises(ValueError):make_targets(np.array([[[-1]]]))
        x=torch.ones((1,1,4),requires_grad=True);loss,_=frequency_objective(x,make_targets(np.zeros((1,1,1),np.int64)))
        self.assertEqual(float(loss.detach()),0);loss.backward();self.assertTrue(torch.isfinite(x.grad).all())


if __name__=='__main__':unittest.main()
