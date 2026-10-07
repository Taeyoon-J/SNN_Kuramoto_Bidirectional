import unittest
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(Path(__file__).resolve().parent)]
import torch
from partition_rgb import SharedRGBDecoder,assignment_weights,groups_to_onehot,patch_centers,reconstruct_one,rgb_patch_means
from snn_kuramoto_bidirectional.evaluation import spatial_components_to_patch_labels

class PartitionRGBTests(unittest.TestCase):
 def setUp(self): torch.manual_seed(106)
 def test_hard_forward_and_q_credit(self):
  groups=[list(range(0,16)),list(range(16,28))]
  h=groups_to_onehot(groups)
  q=torch.rand(256,256,requires_grad=True)
  f=torch.randn(256,8,requires_grad=True)
  t=torch.rand(256,3)
  d=SharedRGBDecoder()
  pred,loss,diag=reconstruct_one(q,h,f,t,d,True)
  self.assertTrue(torch.equal(diag['W'].detach(),h))
  self.assertEqual(diag['K'],3)
  self.assertTrue(torch.isfinite(loss))
  loss.backward()
  self.assertIsNotNone(q.grad)
  self.assertGreater(float(q.grad.abs().sum()),0.)
  self.assertIsNone(f.grad)  # content pool detaches F
  self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum())>0 for p in d.parameters()))
 def test_classifier_group_labels_match_production_converter(self):
  groups=[list(range(0,16)),list(range(16,28))]
  labels=spatial_components_to_patch_labels([groups],16).reshape(-1)
  expected=torch.nn.functional.one_hot(labels,num_classes=3).float()
  self.assertTrue(torch.equal(groups_to_onehot(groups),expected))
 def test_control_has_no_q_gradient(self):
  h=groups_to_onehot([list(range(10))])
  q=torch.rand(256,256,requires_grad=True); f=torch.randn(256,8); target=torch.rand(256,3); d=SharedRGBDecoder()
  _,loss,_=reconstruct_one(q,h,f,target,d,False)
  loss.backward()
  self.assertIsNone(q.grad)
 def test_slot_column_permutation(self):
  h=groups_to_onehot([list(range(12)),list(range(12,20))])
  q=torch.rand(256,256); f=torch.randn(256,8); t=torch.rand(256,3); d=SharedRGBDecoder()
  a,la,_=reconstruct_one(q,h,f,t,d,True)
  perm=torch.tensor([2,0,1]); hp=h[:,perm]
  b,lb,_=reconstruct_one(q,hp,f,t,d,True)
  self.assertLessEqual(float((la-lb).abs().detach()),1e-6)
  self.assertLessEqual(float((a-b).abs().max().detach()),1e-6)
 def test_patch_target_and_coordinates(self):
  images=torch.zeros(1,3,128,128); images[:,:,0:8,0:8]=1
  target=rgb_patch_means(images)
  self.assertEqual(tuple(target.shape),(1,256,3))
  self.assertTrue(torch.equal(target[0,0],torch.ones(3)))
  coords=patch_centers('cpu',torch.float32)
  self.assertEqual(tuple(coords.shape),(256,2))
  self.assertEqual(float(coords[0,0]),-15/16)
  self.assertEqual(float(coords[0,1]),-15/16)

if __name__=='__main__': unittest.main()
