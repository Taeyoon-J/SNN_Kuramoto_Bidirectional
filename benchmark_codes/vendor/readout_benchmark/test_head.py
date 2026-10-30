"""Scientific admission: exact baseline preservation, GT-permutation invariance and gradients."""
import inspect
import unittest
import numpy as np
import torch
from .head import ActivityMaskHead, activity_features, mask_objective


class ReadoutAdmission(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)
        self.x=torch.randn(64,144)
        self.base=torch.zeros(8,8,dtype=torch.long)
        self.base[1:3,1:3]=1;self.base[4:7,4:7]=2
        self.proposals=torch.arange(64).reshape(8,8)//8+1

    def test_initial_prediction_preserves_original_partition(self):
        head=ActivityMaskHead()
        np.testing.assert_array_equal(head.predict(self.x,self.base,self.proposals),self.base.numpy())
        for base in (torch.zeros_like(self.base),torch.ones_like(self.base)):
            np.testing.assert_array_equal(head.predict(self.x,base,self.proposals),base.numpy())

    def test_training_loss_ignores_gt_id_order(self):
        head=ActivityMaskHead();logits=head(self.x,self.base,self.proposals)
        truth=self.base.clone();truth[0,0:2]=3
        renamed=truth.clone();renamed[truth==1]=17;renamed[truth==2]=3;renamed[truth==3]=2
        a,_=mask_objective(logits,truth,self.base);b,_=mask_objective(logits,renamed,self.base)
        torch.testing.assert_close(a,b)

    def test_final_mask_loss_reaches_trainable_readout(self):
        head=ActivityMaskHead()
        labels=self.base.clone();labels[0,0]=1
        loss,_=mask_objective(head(self.x,self.base,self.proposals),labels,self.base)
        loss.backward()
        self.assertGreater(float(head.background[-1].weight.grad.norm()),0)
        self.assertGreater(float(head.residual_scale.grad.abs()),0)
        opt=torch.optim.Adam(head.parameters(),lr=.003);opt.step();opt.zero_grad()
        loss,_=mask_objective(head(self.x,self.base,self.proposals),labels,self.base);loss.backward()
        self.assertGreater(float(head.embedding[-1].weight.grad.norm()),0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None))

    def test_no_gt_or_count_at_inference(self):
        self.assertEqual(list(inspect.signature(ActivityMaskHead.forward).parameters),['self','features','baseline','proposals'])
        head=ActivityMaskHead();logits=head(self.x,torch.zeros_like(self.base),torch.ones_like(self.base))
        self.assertGreaterEqual(logits.shape[0],13)

    def test_real_activity_feature_shape_and_gradient(self):
        s=torch.rand(1,4,64,1024,requires_grad=True)
        m=torch.rand(1,4,64,1024,requires_grad=True)
        features=activity_features(dict(component_spikes=s,component_membrane=m))
        self.assertEqual(features.shape,(1,64,144));features.sum().backward()
        self.assertGreater(float(s.grad[...,512:].norm()),0)
        self.assertEqual(float(s.grad[...,:512].norm()),0)


if __name__=='__main__':unittest.main()
