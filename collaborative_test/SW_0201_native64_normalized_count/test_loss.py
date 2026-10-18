import unittest
import numpy as np
import torch
from .loss import count_penalty,count_calibrated_objective,_vectorized_class_loss,normalized_proxy
from collaborative_test.SW_0197_native64_baseline_replication.count_proxy_review import ideal_blocks
from . import train,evaluate


class CountLossTests(unittest.TestCase):
    def test_ideal_unequal_blocks_have_zero_count_penalty(self):
        q=ideal_blocks([90,5,5]);labels=torch.repeat_interleave(torch.arange(3),torch.tensor([90,5,5]))
        self.assertLess(float(count_penalty(q,labels)),1e-25)
        expected=_vectorized_class_loss(q,labels,temperature=.1)
        torch.testing.assert_close(count_calibrated_objective(q,labels),expected)

    def test_target_is_actual_classes_and_class_names_do_not_matter(self):
        q=torch.full((6,6),.4,dtype=torch.float64)
        labels=torch.tensor([0,0,0,1,1,2]);expected=(normalized_proxy(q).log()-np.log(3))**2
        torch.testing.assert_close(count_penalty(q,labels),expected)
        torch.testing.assert_close(count_penalty(q,labels),count_penalty(q,labels*7+10))

    def test_native4096_loss_and_gradient_finite_without_cubic_tensors(self):
        q=torch.full((4096,4096),.3,requires_grad=True)
        labels=torch.arange(4096,dtype=torch.int64)%7
        loss=count_calibrated_objective(q,labels);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(q.grad).all())
        self.assertGreater(float(q.grad.abs().sum()),0)

    def test_invalid_labels(self):
        q=torch.full((4,4),.4)
        for labels in (torch.ones(4),torch.tensor([-1,0,1,1]),torch.arange(3)):
            with self.assertRaises(ValueError):count_penalty(q,labels)
        for bad in (torch.tensor(.4),torch.full((3,4),.4),torch.zeros(0,0)):
            with self.assertRaises(ValueError):count_penalty(bad,torch.arange(4))

    def test_primary_gate_requires_actual_learning_and_warm_floor(self):
        x=np.zeros((16,3));paired={k:evaluate.paired_change(x+.02,x) for k in ('control','warm')}
        report=dict(new_accepted_updates=1,committed_model_changed=True,CNN_changed_from_warm=True)
        self.assertTrue(evaluate.extension_gate(paired,report))
        report['CNN_changed_from_warm']=False;self.assertFalse(evaluate.extension_gate(paired,report))
        self.assertEqual(train.LIVE_TAIL,64);self.assertEqual(evaluate.OFFSETS,tuple(range(0,32,2)))
