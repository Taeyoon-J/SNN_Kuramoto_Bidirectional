import unittest
import numpy as np
import torch
from .component_loss import component_objective,_vectorized_class_loss
class ComponentLossTests(unittest.TestCase):
    def fixture(self):
        rho=torch.full((4,4,4),.2,dtype=torch.float64,requires_grad=True)
        with torch.no_grad():rho[0].fill_(-.2)
        return rho,torch.tensor([0,0,1,1])
    def test_credit_survives_negative_component_product(self):
        rho,labels=self.fixture();q=rho.clamp_min(0).prod(0)
        old=_vectorized_class_loss(q,labels)
        blocked=torch.autograd.grad(old,rho,retain_graph=True)[0]
        self.assertEqual(float(blocked.abs().sum()),0.)
        grad=torch.autograd.grad(component_objective(q,rho,labels),rho)[0]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertTrue(all(float(v.norm())>0 for v in grad))
        for row in grad:
            self.assertLess(float(row[0,1]),0.)
            self.assertGreater(float(row[0,2]),0.)
            self.assertEqual(float(row[0,0]),0.)
    def test_component_order_and_label_names_invariant(self):
        rho,labels=self.fixture();q=rho.clamp_min(0).prod(0)
        a=component_objective(q,rho,labels)
        b=component_objective(q,rho.flip(0),labels*7+4)
        torch.testing.assert_close(a,b,rtol=0,atol=1e-15)
    def test_single_class_zero_objective(self):
        rho,labels=self.fixture();q=rho.clamp_min(0).prod(0)
        loss=component_objective(q,rho,torch.zeros_like(labels))
        self.assertEqual(float(loss.detach()),0.)
        self.assertEqual(float(torch.autograd.grad(loss,rho)[0].abs().sum()),0.)
    def test_invalid_geometry_and_temperature(self):
        rho,labels=self.fixture();q=rho.clamp_min(0).prod(0)
        with self.assertRaises(ValueError):component_objective(q,rho[:3],labels)
        for temp in (0.,float('nan')):
            with self.assertRaises(ValueError):component_objective(q,rho,labels,temperature=temp)
