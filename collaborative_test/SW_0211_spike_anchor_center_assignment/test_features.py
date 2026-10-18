import unittest
import numpy as np
import torch
from .features import anchor_indices,descriptors
from collaborative_test.SW_0175_critical_component_binding.critical_component_objective import native_component_rho_q

class FeatureTests(unittest.TestCase):
    def test_canonical_anchor_locations(self):
        expected=[r*64+c for r in range(4,64,8) for c in range(4,64,8)]
        self.assertEqual(anchor_indices().tolist(),expected)
        with self.assertRaises(ValueError):anchor_indices(63,8)

    def test_component_anchor_order_and_statistics(self):
        n=64;s=torch.arange(4*n*8).reshape(1,4,n,8).float()/100
        rho=torch.arange(4*n*n).reshape(1,4,n,n).float()/(4*n*n)
        value=descriptors(dict(component_spikes=s,component_membrane=2*s),rho,grid=8,side=2,settle=4)
        self.assertEqual(value.shape,(28,8,8))
        self.assertTrue(torch.equal(value[:16].reshape(16,n).T,rho[0,:, :,anchor_indices(8,2)].permute(1,0,2).reshape(n,16)))
        self.assertTrue(torch.allclose(value[16:20].reshape(4,n),s[...,4:].mean(-1)[0]))

    def test_common_settled_time_permutation_preserves_descriptors(self):
        rng=torch.Generator().manual_seed(211);s=torch.randn((1,4,64,16),generator=rng);m=torch.randn((1,4,64,16),generator=rng)
        _,q=native_component_rho_q(s.mean(1),s,settle=8)
        rho,_=native_component_rho_q(s.mean(1),s,settle=8)
        before=descriptors(dict(component_spikes=s,component_membrane=m),rho,grid=8,side=2,settle=8)
        order=torch.cat((torch.arange(8),torch.arange(15,7,-1)));s=s[...,order];m=m[...,order]
        rho,_=native_component_rho_q(s.mean(1),s,settle=8)
        after=descriptors(dict(component_spikes=s,component_membrane=m),rho,grid=8,side=2,settle=8)
        self.assertTrue(torch.allclose(before,after,atol=1e-6,rtol=1e-6))

    def test_wrong_native_trace_rejected(self):
        trace=dict(component_spikes=torch.zeros((1,4,64,8)),component_membrane=torch.zeros((1,4,64,8)))
        with self.assertRaises(ValueError):descriptors(trace,torch.zeros((1,4,63,63)),grid=8,side=2,settle=4)
        with self.assertRaises(ValueError):descriptors(trace,torch.zeros((1,4,64,64)),grid=8,side=2,settle=8)
