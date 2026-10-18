import unittest
import numpy as np
import torch
from .loss import absolute_calibration,calibrated_objective,_vectorized_class_loss


def loop_loss(q,y):
    classes=torch.unique(y).tolist();positive=[];negative=[]
    if len(classes)<2:return q.sum()*0
    for c in classes:
        ix=torch.where(y==c)[0].tolist()
        terms=[-q[i,j].clamp(1e-6,1-1e-6).log() for i in ix for j in ix if i!=j]
        if terms:positive.append(torch.stack(terms).mean())
        for d in classes:
            if d!=c:
                js=torch.where(y==d)[0].tolist()
                negative.append(torch.stack([-torch.log1p(-q[i,j].clamp(1e-6,1-1e-6)) for i in ix for j in js]).mean())
    neg=torch.stack(negative).mean()
    return .5*(torch.stack(positive).mean()+neg) if positive else neg


class CalibrationTests(unittest.TestCase):
    def test_value_gradient_independent_unequal_class_loop(self):
        q=torch.tensor(np.random.RandomState(199).uniform(.05,.95,(9,9)),dtype=torch.float64,requires_grad=True)
        y=torch.tensor([0,0,0,0,1,1,1,2,3])
        a=absolute_calibration(q,y);b=loop_loss(q,y)
        torch.testing.assert_close(a,b,atol=1e-13,rtol=0)
        ga=torch.autograd.grad(a,q,retain_graph=True)[0];gb=torch.autograd.grad(b,q)[0]
        torch.testing.assert_close(ga,gb,atol=1e-13,rtol=0)
        torch.testing.assert_close(ga.diagonal(),torch.zeros(9,dtype=torch.float64),atol=1e-14,rtol=0)

    def test_permutation_and_class_renaming(self):
        q=torch.tensor(np.random.RandomState(1).uniform(.1,.9,(6,6)),dtype=torch.float64)
        y=torch.tensor([0,0,0,1,1,2]);p=torch.tensor([5,1,4,0,3,2])
        torch.testing.assert_close(absolute_calibration(q,y),absolute_calibration(q[p][:,p],y[p]*7+4))

    def test_absolute_gradient_pulls_same_above_boundary(self):
        q=torch.full((4,4),.3,dtype=torch.float64,requires_grad=True);y=torch.tensor([0,0,1,1])
        g=torch.autograd.grad(absolute_calibration(q,y),q)[0]
        self.assertLess(float(g[0,1]),0);self.assertGreater(float(g[0,2]),0)
        self.assertEqual(float(g[0,0]),0)

    def test_relative_QCE_invariance_absolute_loss_not_invariant(self):
        y=torch.tensor([0,0,1,1]);same=y[:,None]==y[None,:];eye=torch.eye(4,dtype=torch.bool)
        q=torch.where(same,torch.tensor(.75,dtype=torch.float64),torch.tensor(.35,dtype=torch.float64));q[eye]=1
        shifted=q.clone();shifted[~eye]-=.3
        torch.testing.assert_close(_vectorized_class_loss(q,y,temperature=.1),_vectorized_class_loss(shifted,y,temperature=.1),atol=1e-13,rtol=0)
        self.assertGreater(float(absolute_calibration(shifted,y)),float(absolute_calibration(q,y)))

    def test_singletons_and_single_class_connected_zero(self):
        q=torch.full((3,3),.3,dtype=torch.float64,requires_grad=True)
        torch.testing.assert_close(absolute_calibration(q,torch.arange(3)),loop_loss(q,torch.arange(3)))
        zero=absolute_calibration(q,torch.zeros(3,dtype=torch.int64))
        self.assertEqual(float(zero.detach()),0);self.assertEqual(float(torch.autograd.grad(zero,q)[0].sum()),0)

    def test_control_is_original_loss_and_candidate_sum(self):
        q=torch.full((4,4),.4);y=torch.tensor([0,0,1,1])
        old=_vectorized_class_loss(q,y,temperature=.1)
        self.assertTrue(torch.equal(calibrated_objective(q,y,calibrated=False),old))
        torch.testing.assert_close(calibrated_objective(q,y,calibrated=True),old+absolute_calibration(q,y))

    def test_invalid_probabilities_labels(self):
        for bad in (float('nan'),1.1,-.1):
            with self.assertRaises(ValueError):absolute_calibration(torch.full((2,2),bad),torch.arange(2))
        with self.assertRaises(ValueError):absolute_calibration(torch.ones(2,3),torch.arange(2))
        with self.assertRaises(ValueError):absolute_calibration(torch.ones(2,2),torch.tensor([-1,1]))
