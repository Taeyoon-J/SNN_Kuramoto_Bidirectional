import unittest
import numpy as np
import torch
from .count_proxy_review import normalized_proxy,ideal_blocks,plv_group_count_loss


class ProxyTests(unittest.TestCase):
    def test_equal_blocks_raw_and_normalized_match_three(self):
        q=ideal_blocks([30,30,30])
        self.assertAlmostEqual(float(q.numel()/q.square().sum()),3)
        self.assertAlmostEqual(float(normalized_proxy(q)),3)
        self.assertEqual(float(plv_group_count_loss(q[None],target_groups=3)),0)

    def test_actual_local_raw_loss_penalizes_perfect_unequal_partition(self):
        q=ideal_blocks([90,5,5]);raw=q.numel()/float(q.square().sum())
        self.assertAlmostEqual(raw,10000/8150)
        self.assertGreater(float(plv_group_count_loss(q[None],target_groups=3)),.3)
        self.assertAlmostEqual(float(normalized_proxy(q)),3)

    def test_proxy_does_not_claim_exact_connected_components(self):
        q=torch.tensor([[1.,1.,0.],[1.,1.,1.],[0.,1.,1.]],dtype=torch.float64)
        self.assertAlmostEqual(float(normalized_proxy(q)),32/23)
        self.assertGreater(float(normalized_proxy(q)),1) # this graph is connected

    def test_gradient_finite_diagonal_zero_and_permutation_invariant(self):
        q=torch.tensor(np.random.RandomState(197).uniform(.1,.9,(7,7)),dtype=torch.float64,requires_grad=True)
        p=torch.tensor([6,1,4,0,3,2,5]);value=normalized_proxy(q)
        torch.testing.assert_close(value,normalized_proxy(q[p][:,p]))
        g=torch.autograd.grad(value,q)[0]
        self.assertTrue(torch.isfinite(g).all());self.assertGreater(float(g.abs().sum()),0)
        self.assertTrue(torch.equal(g.diagonal(),torch.zeros(7,dtype=torch.float64)))

    def test_invalid_input_and_no_mutation(self):
        q=torch.full((3,3),.4);before=q.clone();normalized_proxy(q)
        self.assertTrue(torch.equal(q,before))
        for bad in (float('nan'),-.1,1.1):
            with self.assertRaises(ValueError):normalized_proxy(torch.full((3,3),bad))
