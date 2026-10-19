import copy,unittest
import numpy as np
import torch
from snn_kuramoto_bidirectional.sinusoidal_gating import sinusoidal_gating as shared
from collaborative_test.SW_0134_native_spike_binding.test_rollout import _core
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout as original_rollout
from collaborative_test.SW_0130_phase_state_integration.model import PhaseStateIntegration
from .component_gate import sinusoidal_gating
from .rollout import late_rollout


class ComponentGateTests(unittest.TestCase):
    def test_components_keep_their_own_delayed_gate(self):
        theta=torch.tensor([[[0.,.5,1.,1.5]]]);old=theta/2
        carrier,gate=sinusoidal_gating([old,theta],1,1)
        self.assertTrue(torch.equal(gate,.5*(1+torch.sin(old))))
        self.assertTrue(torch.equal(carrier,torch.sin(theta)*gate))
        self.assertGreater(float((gate-gate[...,:1]).abs().max()),0)

    def test_aligned_components_match_original_raw_gate(self):
        theta=torch.tensor([[[.25]*4,[1.]*4]])
        a,g=sinusoidal_gating([theta],0,2);b,h=shared([theta],0,2,gate_mode='raw')
        torch.testing.assert_close(a,b,rtol=0,atol=1e-7)
        torch.testing.assert_close(g,h[...,None].expand_as(g),rtol=0,atol=1e-7)

    def test_permutation_equivariance_and_independent_full_cycle_lifts(self):
        x=torch.tensor([[[.1,.7,1.2,2.1]]]);perm=[2,0,3,1]
        a,g=sinusoidal_gating([x],0,2);b,h=sinusoidal_gating([x[...,perm]],0,2)
        self.assertTrue(torch.equal(a[...,perm],b));self.assertTrue(torch.equal(g[...,perm],h))
        lifted=x+torch.tensor([0.,2*np.pi,-2*np.pi,4*np.pi])
        b,h=sinusoidal_gating([lifted],0,2)
        torch.testing.assert_close(a,b,rtol=0,atol=1e-6);torch.testing.assert_close(g,h,rtol=0,atol=1e-6)

    def test_actual_small_core_preserves_theta_and_parameters(self):
        torch.manual_seed(243)
        source=PhaseStateIntegration(_core(steps=16),'phase');variant=copy.deepcopy(source)
        gamma=torch.randn(1,8,64);state={k:v.clone() for k,v in variant.state_dict().items()}
        rng=torch.get_rng_state().clone()
        old=original_rollout(source,gamma,total_steps=16,live_tail_steps=0)
        torch.set_rng_state(rng);new=late_rollout(variant,gamma,total_steps=16,live_tail_steps=0)
        self.assertTrue(torch.equal(old['theta'],new['theta']))
        self.assertEqual(new['component_gates'].shape,(1,4,64,16))
        self.assertGreater(float((new['component_gates']-new['component_gates'][:,:1]).abs().max()),0)
        for name in ('component_spikes','component_gates','component_membrane'):
            self.assertTrue(bool(torch.isfinite(new[name]).all()));self.assertFalse(new[name].requires_grad)
        self.assertTrue(all(torch.equal(v,variant.state_dict()[k]) for k,v in state.items()))

    def test_unsupported_gate_mode_rejected(self):
        with self.assertRaises(ValueError):sinusoidal_gating([torch.zeros(1,2,4)],0,2,'sigmoid')


if __name__=='__main__':unittest.main()
