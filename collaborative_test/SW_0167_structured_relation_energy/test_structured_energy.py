"""Tiny CPU oracles for SW0167's provisional frozen-source objective pieces."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import torch
torch.set_num_threads(1)
import unittest

from collaborative_test.SW_0162_connected_signed_partition.connected_signed import signed_pair_energy
from collaborative_test.SW_0164_foreground_preserving_partition.foreground_signed import foreground_induced_merge
from collaborative_test.SW_0167_structured_relation_energy import structured_energy as se


class StructuredEnergyTests(unittest.TestCase):
    def test_zero_residual_reproduces_sealed_164_energy_merge_log_and_labels(self):
        rng=np.random.RandomState(167)
        n=12; h,w=3,4; gain=2.75
        raw=rng.normal(0,.9,(n,n)); q=1/(1+np.exp(-raw)); q=(q+q.T)/2
        a=rng.uniform(0,.003,(n,n)); a=(a+a.T)/2
        labels=np.array([[2,2,0,7],[2,0,7,7],[5,5,5,0]],dtype=np.int32)
        e0=se.base_energy(q,a,gain)
        ref=foreground_induced_merge(q,a,gain,labels,grid_shape=(h,w))
        zero_residual=se.solve_energy(e0,labels,grid_shape=(h,w))
        np.testing.assert_array_equal(e0,ref["pair_energy"])
        np.testing.assert_array_equal(zero_residual["labels"],ref["labels"])
        self.assertEqual(zero_residual["merge_count"],ref["merge_count"])
        self.assertEqual(zero_residual["merge_log"],ref["merge_log"])

    def test_pair_features_keep_signed_component_correlations_and_node_stats(self):
        spikes=torch.tensor([
            [[0.,1.,0.,1.],[1.,0.,1.,0.],[0.,0.,1.,1.]],
            [[0.,1.,0.,1.],[0.,1.,0.,1.],[1.,1.,0.,0.]],
            [[1.,0.,0.,1.],[1.,0.,0.,1.],[0.,1.,1.,0.]],
            [[0.,0.,1.,1.],[1.,1.,0.,0.],[1.,0.,1.,0.]],
        ])
        rho=[]
        for component in spikes:
            centered=component-component.mean(-1,keepdim=True)
            centered=centered/centered.norm(dim=-1,keepdim=True).clamp_min(1e-8)
            rho.append((centered@centered.T).clamp(-1,1))
        rho=torch.stack(rho)
        q=rho.clamp_min(0).prod(0)
        a=torch.tensor([[1.,.01,.02],[.01,1.,.03],[.02,.03,1.]])
        f=se.pair_features(spikes,q,a,2.)
        self.assertEqual(tuple(f.shape),(3,3,10))
        torch.testing.assert_close(f,f.transpose(0,1))
        # Anti-correlated first component remains negative in channel 0.
        self.assertLess(float(f[0,1,0]),-.99)
        self.assertTrue(torch.isfinite(f).all())

    def test_zero_initialized_pair_mlp_is_exact_identity_on_energy(self):
        torch.manual_seed(167001)
        model=se.PairRelationMLP()
        x=torch.randn(5,5,10)
        e=torch.randn(5,5); e=(e+e.T)/2; e.fill_diagonal_(0)
        w=torch.ones_like(e); w.fill_diagonal_(0)
        corrected,residual=model(x,e,w)
        torch.testing.assert_close(residual,torch.zeros_like(residual),rtol=0,atol=0)
        torch.testing.assert_close(corrected,e,rtol=0,atol=0)

    def test_feasible_target_keeps_gt_background_and_splits_disconnected_fragments(self):
        source=np.array([[4,4,0,3],[0,4,3,3],[4,0,0,0]],dtype=np.int32)
        gt=np.array([[0,0,1,1],[0,0,0,1],[0,0,0,0]],dtype=np.int32)
        target=se.feasible_target(source,gt)
        np.testing.assert_array_equal(target>0,source>0)
        # GT background at 0,0 remains a positive projected target component.
        self.assertGreater(target[0,0],0)
        # Same GT id 0 splits where source foreground is disconnected by support.
        self.assertNotEqual(int(target[0,0]),int(target[2,0]))
        # Two adjacent source-supported pixels of the same GT object stay together.
        self.assertEqual(int(target[0,3]),int(target[1,3]))

    def test_balanced_pair_weights_and_structured_and_local_losses_have_gradients(self):
        # A three-node chain with target {0}|{1,2}. The loss-augmented greedy
        # partition differs from the target, yielding a nonzero structured gradient.
        target=torch.tensor([[1,2,2]])
        support=torch.ones((1,3),dtype=torch.bool)
        same,weights,idx=se.pair_targets(target,support)
        self.assertEqual(idx.tolist(),[0,1,2])
        upper=torch.triu(torch.ones_like(weights,dtype=torch.bool),1)
        self.assertAlmostEqual(float(weights[upper & (same>0)].sum()),.5)
        self.assertAlmostEqual(float(weights[upper & (same==0)].sum()),.5)
        e0=torch.zeros((3,3)); e0.fill_diagonal_(0)
        e=torch.zeros((3,3),requires_grad=True)
        loss=se.structured_hinge(e,e0,target,support)
        self.assertAlmostEqual(float(loss.detach()),.75,places=7)
        loss.backward()
        self.assertTrue(torch.isfinite(e.grad).all())
        expected=torch.zeros_like(e.grad); expected[0,1]=1.; expected[1,2]=-1.
        torch.testing.assert_close(e.grad,expected,rtol=0,atol=1e-7)
        e2=torch.zeros((3,3),requires_grad=True)
        margin=se.local_margin_loss(e2,e0,target,support)
        self.assertAlmostEqual(float(margin.detach()),1.,places=7)
        margin.backward()
        self.assertTrue(torch.isfinite(e2.grad).all())
        expected2=torch.zeros_like(e2.grad)
        expected2[0,1]=expected2[0,2]=.25; expected2[1,2]=-.5
        torch.testing.assert_close(e2.grad,expected2,rtol=0,atol=1e-7)

    def test_structured_oracle_keeps_gapped_original_grid_coordinates(self):
        # Nodes 0 and 2 are both in F but separated by background node 1.
        # Even overwhelming positive cross-energy cannot join them.
        target=torch.tensor([[1,0,2]])
        support=torch.tensor([[True,False,True]])
        e0=torch.zeros((3,3)); e=torch.zeros((3,3),requires_grad=True)
        with torch.no_grad(): e[0,2]=e[2,0]=.5
        loss=se.structured_hinge(e,e0,target,support)
        self.assertAlmostEqual(float(loss.detach()),0.0,places=7)
        loss.backward()
        self.assertTrue(torch.isfinite(e.grad).all())
        self.assertEqual(float(e.grad[0,2]),0.)

    def test_zero_and_singleton_support_produce_zero_losses(self):
        for labels,support in ((torch.zeros((2,2),dtype=torch.long),torch.zeros((2,2),dtype=torch.bool)),
                               (torch.ones((1,1),dtype=torch.long),torch.ones((1,1),dtype=torch.bool))):
            e=torch.randn(*labels.shape,requires_grad=True); e0=torch.zeros_like(e)
            self.assertEqual(float(se.structured_hinge(e,e0,labels,support).detach()),0.)
            self.assertEqual(float(se.local_margin_loss(e,e0,labels,support).detach()),0.)

    def test_two_disposable_updates_credit_final_then_hidden_layer_in_both_arms(self):
        torch.manual_seed(167)
        n=3; features=torch.zeros(n,n,10); features[:,:,0]=1.
        features[0,1,1]=features[1,0,1]=2.
        features[1,2,2]=features[2,1,2]=.5
        e0=torch.zeros(n,n); weight=torch.ones(n,n)
        target=torch.tensor([[1,2,2]]); support=torch.ones((1,3),dtype=torch.bool)
        for objective in (se.structured_hinge,se.local_margin_loss):
            net=se.PairRelationMLP()
            torch.nn.init.constant_(net.hidden.weight,.1)
            torch.nn.init.constant_(net.hidden.bias,.2)
            initial={k:v.detach().clone() for k,v in net.state_dict().items()}
            opt=torch.optim.SGD(net.parameters(),lr=.01)
            for step in (1,2):
                energy,_=net(features,e0,weight)
                loss=objective(energy,e0,target,support)
                opt.zero_grad(set_to_none=True); loss.backward()
                self.assertTrue(torch.isfinite(net.output.weight.grad).all())
                self.assertGreater(float(net.output.weight.grad.norm()),0.)
                if step==1:
                    self.assertEqual(float(net.hidden.weight.grad.norm()),0.)
                else:
                    self.assertGreater(float(net.hidden.weight.grad.norm()),0.)
                opt.step()
            changed={k:float((net.state_dict()[k]-initial[k]).norm()) for k in initial}
            self.assertGreater(changed["output.weight"],0.)
            self.assertGreater(changed["hidden.weight"],0.)

    def test_invalid_energy_and_targets_fail_closed(self):
        with self.assertRaises(ValueError): se.solve_energy(np.array([[0.,1.],[0.,0.]]),np.ones((1,2),int),grid_shape=(1,2))
        with self.assertRaises(ValueError): se.feasible_target(np.array([[0,1]]),np.array([[0.,1.]]))
        with self.assertRaises(ValueError): se.pair_features(torch.ones(3,2,3),torch.eye(2),torch.eye(2),1.)
        with self.assertRaisesRegex(ValueError,"identical geometry"):
            se.pair_targets(torch.ones((2,2),dtype=torch.long),torch.ones((1,4),dtype=torch.bool))


if __name__ == "__main__": unittest.main()
