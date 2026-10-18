import unittest
import numpy as np
import torch
from .energy_loss import pair_energy,make_context,ranking_loss
from collaborative_test.SW_0162_connected_signed_partition.connected_signed import signed_pair_energy


class RankingLossTests(unittest.TestCase):
    def test_exact_native_energy_and_finite_difference_credit(self):
        q=torch.tensor([[.5,.8,.7],[.8,.5,.4],[.7,.4,.5]],dtype=torch.float64,requires_grad=True)
        a=torch.tensor([[0,.003,.002],[.003,0,.001],[.002,.001,0]],dtype=torch.float64,requires_grad=True)
        gain=torch.tensor(1.3,dtype=torch.float64,requires_grad=True)
        expected=signed_pair_energy(q.detach().numpy(),a.detach().numpy(),float(gain.detach()))
        np.testing.assert_array_equal(pair_energy(q,a,gain).detach().numpy(),expected)
        pred=np.array([[1,1,1]]);gt=np.array([[1,2,2]])
        context=make_context(pred,gt,q,a,gain)
        loss=ranking_loss(q,a,gain,context);grads=torch.autograd.grad(loss,(q,a,gain))
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        for tensor,gradient,index in ((q,grads[0],(0,1)),(a,grads[1],(0,1)),(gain,grads[2],())):
            plus=tensor.detach().clone();minus=plus.clone();step=1e-7;plus[index]+=step;minus[index]-=step
            args=[q.detach(),a.detach(),gain.detach()];position=0 if tensor is q else 1 if tensor is a else 2
            args[position]=plus;high=ranking_loss(*args,context).item();args[position]=minus;low=ranking_loss(*args,context).item()
            self.assertAlmostEqual(float(gradient[index]),(high-low)/(2*step),places=6)
        self.assertGreater(float(grads[0][0,1]),0) # descent reduces a false merge

    def test_reference_repairs_connected_false_split_without_dropping_BG(self):
        pred=np.array([[1,2,0],[3,4,4]]);gt=np.array([[1,1,0],[0,2,2]])
        q=torch.full((6,6),.7,dtype=torch.float64,requires_grad=True);a=q*0;gain=torch.tensor(1.)
        context=make_context(pred,gt,q,a,gain)
        self.assertEqual(context['source_foreground_patches'],5)
        self.assertEqual(context['reference_groups'],3)
        self.assertEqual(context['disagreement_pairs'],1)
        self.assertEqual(context['difference'][0,1],-1)

    def test_correct_partition_connected_zero_and_label_permutation_invariance(self):
        pred=np.array([[3,3],[4,4]]);gt=np.array([[1,1],[2,2]])
        q=torch.full((4,4),.6,requires_grad=True);a=torch.zeros((4,4),requires_grad=True);gain=torch.tensor(1.,requires_grad=True)
        context=make_context(pred,gt,q,a,gain);loss=ranking_loss(q,a,gain,context)
        self.assertEqual(float(loss.detach()),0);self.assertEqual(context['disagreement_pairs'],0)
        gradients=torch.autograd.grad(loss,(q,a,gain));self.assertTrue(all(torch.count_nonzero(g)==0 for g in gradients))
        swapped=np.where(pred==3,99,10);other=make_context(swapped,gt,q,a,gain)
        np.testing.assert_array_equal(context['difference'],other['difference'])

    def test_frozen_context_not_recomputed_after_parameter_change(self):
        q=torch.full((4,4),.8,dtype=torch.float64);a=torch.zeros_like(q);gain=torch.tensor(1.)
        context=make_context(np.ones((2,2),np.int64),np.array([[1,1],[2,2]]),q,a,gain)
        scale=context['scale'];first=ranking_loss(q,a,gain,context).item()
        second=ranking_loss(torch.full_like(q,.2),a,gain,context).item()
        self.assertGreater(first,second);self.assertEqual(context['scale'],scale)
        self.assertAlmostEqual(second,0,places=12)

    def test_empty_support_and_invalid_geometry(self):
        q=torch.full((1,1),.5,requires_grad=True);a=q*0;gain=torch.tensor(1.)
        c=make_context(np.zeros((1,1),np.int64),np.ones((1,1),np.int64),q,a,gain)
        self.assertEqual(float(ranking_loss(q,a,gain,c).detach()),0)
        with self.assertRaises(ValueError):make_context(np.ones((1,2),np.int64),np.ones((1,2),np.int64),q,a,gain)


if __name__=='__main__':unittest.main()
