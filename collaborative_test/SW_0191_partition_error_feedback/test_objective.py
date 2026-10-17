import unittest
from types import SimpleNamespace
import numpy as np
import torch
from . import objective,feedback_context


class FeedbackObjective(unittest.TestCase):
    def test_split_pairs_restore_signed_component_credit_when_product_is_zero(self):
        rho=torch.full((4,2,2),-.2,dtype=torch.float64,requires_grad=True)
        q=rho.clamp_min(0).prod(0)
        loss=objective.partition_error_loss(q,rho,torch.tensor([1,1]),torch.tensor([4,7]))
        loss.backward()
        self.assertEqual(float(q[0,1].detach()),0.)
        self.assertTrue(bool((rho.grad[:,0,1]<0).all()))
        self.assertGreater(float(loss.detach()),0.)

    def test_merged_object_pairs_push_production_Q_down(self):
        q=torch.full((2,2),.7,dtype=torch.float64,requires_grad=True)
        rho=torch.full((4,2,2),.8,dtype=torch.float64,requires_grad=True)
        loss=objective.partition_error_loss(q,rho,torch.tensor([0,2]),torch.tensor([0,0]))
        loss.backward()
        self.assertGreater(float(q.grad[0,1]),0.)

    def test_correct_partition_has_zero_feedback_and_connected_zero_grad(self):
        q=torch.eye(4,dtype=torch.float64,requires_grad=True)
        rho=torch.zeros((4,4,4),dtype=torch.float64,requires_grad=True)
        loss=objective.partition_error_loss(q,rho,torch.tensor([0,0,2,2]),torch.tensor([8,8,3,3]))
        loss.backward();self.assertEqual(float(loss.detach()),0.)
        self.assertEqual(float(q.grad.abs().sum()+rho.grad.abs().sum()),0.)

    def test_label_permutations_do_not_change_class_balanced_loss(self):
        rho=torch.full((4,4,4),.3,dtype=torch.float64)
        q=rho.prod(0)
        left=objective.partition_error_loss(q,rho,torch.tensor([0,0,2,2]),torch.tensor([5,7,7,7]))
        right=objective.partition_error_loss(q,rho,torch.tensor([92,92,1,1]),torch.tensor([99,3,3,3]))
        torch.testing.assert_close(left,right)

    def test_wrong_geometry_or_noninteger_labels_rejected(self):
        with self.assertRaises(ValueError):
            objective.partition_error_loss(torch.zeros(2,2),torch.zeros(4,2,2),torch.zeros(2),torch.zeros(2,dtype=torch.long))


class FrozenContext(unittest.TestCase):
    def test_context_is_a_copy_and_stale_batches_or_mutations_rejected(self):
        found=SimpleNamespace();rows={2:np.array([0,1,1]),5:np.array([0,2,2])}
        feedback_context.install(found,[2,5],rows);digest=feedback_context.context_sha(found)
        rows[2][0]=9
        self.assertEqual(feedback_context.context_sha(found),digest)
        with self.assertRaises(ValueError):feedback_context.require_batch(found,[5,2])
        predicted=feedback_context.get_prediction(found,2,'cpu');predicted[0]=8
        self.assertEqual(feedback_context.context_sha(found),digest)
        found._sw0191_feedback_context['rows'][2][0]=7
        with self.assertRaises(ValueError):feedback_context.context_sha(found)


if __name__=='__main__':unittest.main()
