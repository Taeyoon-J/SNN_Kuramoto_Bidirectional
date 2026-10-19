import unittest
import numpy as np
import torch
from .layout_ops import restore_native_spikes,matched_gate_only_trace


class LayoutControlTests(unittest.TestCase):
    def test_actual_stack_permute_reshape_storage_is_reconstructed(self):
        b,c,n,t=2,4,3,8
        frames=torch.arange(t*b*c*n,dtype=torch.float32).reshape(t,b*c,n)
        original=frames.permute(1,2,0).reshape(b,c,n,t)
        restored=restore_native_spikes(original.numpy().copy(),'cpu')
        self.assertEqual(restored.stride(),original.stride());self.assertTrue(torch.equal(restored,original))

    def test_substitution_preserves_storage_and_leaves_actual_trace_unchanged(self):
        source=np.zeros((1,4,3,8),dtype=np.float32);spikes=restore_native_spikes(source,'cpu')
        gates=torch.full_like(torch.empty((1,4,3,8)),.4)
        trace=dict(component_spikes=spikes,component_gates=gates,spikes=spikes.mean(1))
        result=matched_gate_only_trace(trace)
        self.assertEqual(result['component_spikes'].stride(),spikes.stride())
        self.assertTrue(torch.equal(result['component_spikes'],gates))
        self.assertTrue(torch.equal(result['spikes'],result['component_spikes'].mean(1)))
        self.assertEqual(int(torch.count_nonzero(spikes)),0)

    def test_identical_actual_tail_has_bitwise_identical_real_native_Q(self):
        from collaborative_test.SW_0175_critical_component_binding.critical_component_objective import native_component_rho_q
        rng=torch.Generator().manual_seed(241)
        gates=torch.rand((1,4,3,520),generator=rng)
        raw=gates.numpy().copy();raw[...,:512]=0
        spikes=restore_native_spikes(raw,'cpu')
        actual=dict(component_spikes=spikes,component_gates=gates,spikes=spikes.mean(1))
        matched=matched_gate_only_trace(actual)
        _,qa=native_component_rho_q(actual['spikes'],spikes,settle=512)
        _,qb=native_component_rho_q(matched['spikes'],matched['component_spikes'],settle=512)
        self.assertTrue(torch.equal(qa,qb))

    def test_invalid_component_geometry_rejected(self):
        with self.assertRaises(ValueError):restore_native_spikes(np.zeros((1,3,2,4),dtype=np.float32),'cpu')
        with self.assertRaises(ValueError):matched_gate_only_trace(dict(component_spikes=torch.zeros(1,4,2,4),component_gates=torch.ones(1,4,2,3)))

    def test_queue_routes_only_layout_control_worker(self):
        from . import layout_queue,layout_control
        self.assertIs(layout_queue.api,layout_control)
        self.assertEqual(layout_queue.MODULE,'collaborative_test.SW_0241_native64_event_readout_counterfactual.layout_queue')
        self.assertEqual(layout_queue.worker_command(1,3)[-1],'3')
        with self.assertRaises(ValueError):layout_queue.worker_command(1,4)


if __name__=='__main__':unittest.main()
