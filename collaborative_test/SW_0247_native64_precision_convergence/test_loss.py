import unittest
from types import SimpleNamespace
import numpy as np
import torch
from collaborative_test.SW_0222_native64_frequency_teacher_denoising.teacher_loss import make_teacher,teacher_objective
from collaborative_test.SW_0221_native64_frequency_pretraining import train as helpers
from .loss import groups_from_labels,within_variance,objective


class WithinPrecisionTests(unittest.TestCase):
    def prepared(self):
        rng=np.random.RandomState(244)
        gamma=rng.randn(1,8,6).astype(np.float32);old=rng.randn(1,6,4).astype(np.float32)
        labels=np.array([[[0,0,1],[1,1,2]]],dtype=np.int64)
        targets=make_teacher(gamma,old,labels)
        return torch.tensor(gamma+.1,requires_grad=True),torch.tensor(old+.2,requires_grad=True),targets,groups_from_labels(labels)

    def test_control_value_and_gradients_equal_original_loss(self):
        gamma,f,t,g=self.prepared();a,_=teacher_objective(gamma,f,t);b,_=objective(gamma,f,t,g,1.)
        ga=torch.autograd.grad(a,(gamma,f));gb=torch.autograd.grad(b,(gamma,f))
        self.assertTrue(torch.equal(a,b))
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(ga,gb)))

    def test_candidate_adds_only_fifteen_times_within_variance(self):
        gamma,f,t,g=self.prepared();base,_=teacher_objective(gamma,f,t);loss,parts=objective(gamma,f,t,g,16.)
        self.assertTrue(torch.equal(loss,base+15*parts['within_variance']))
        grads=torch.autograd.grad(loss,(gamma,f))
        self.assertTrue(all(bool(torch.isfinite(x).all()) for x in grads))

    def test_class_balancing_and_translation_invariance(self):
        labels=np.array([[[0,0,1,1,1]]]);groups=groups_from_labels(labels)
        f=torch.tensor([0.,2.,10.,10.,10.])[None,:,None].expand(1,5,4).clone()
        self.assertEqual(float(within_variance(f,groups)),.5)
        shifted=f.clone();shifted[:,2:]+=7
        self.assertTrue(torch.equal(within_variance(f,groups),within_variance(shifted,groups)))

    def test_encoder_optimizer_freezes_entire_core(self):
        encoder=torch.nn.Module();encoder.base_encoder=torch.nn.Linear(3,3);encoder.context=torch.nn.Linear(3,3)
        found=SimpleNamespace(encoder=encoder,wrapped=torch.nn.Linear(3,4))
        opt=helpers.optimizer_for(found)
        self.assertTrue(all(not p.requires_grad for p in found.wrapped.parameters()))
        self.assertEqual({id(p) for group in opt.param_groups for p in group['params']},{id(p) for p in encoder.parameters()})

    def test_invalid_labels_and_unregistered_coefficient_rejected(self):
        with self.assertRaises(ValueError):groups_from_labels(np.array([[[-1]]]))
        gamma,f,t,g=self.prepared()
        with self.assertRaises(ValueError):objective(gamma,f,t,g,4.)


if __name__=='__main__':unittest.main()
