import unittest
import numpy as np
import torch
from .readout import gate_only_trace,compare_labels,predict_pair


class CounterfactualTests(unittest.TestCase):
    def test_substitution_preserves_input_and_recomputes_component_mean_and_Q(self):
        gate=torch.tensor([[[.2,.8],[.4,.6]]]);components=gate[:,None].repeat(1,4,1,1)
        components[:,0,0,0]=0
        trace=dict(component_spikes=components,component_gates=gate,spikes=components.mean(1),theta=object())
        old=components.clone();calls=[]
        def correlate(mean,spikes,settle):
            self.assertEqual(settle,512);self.assertTrue(torch.equal(mean,spikes.mean(1)))
            self.assertTrue(torch.equal(spikes,gate[:,None].expand_as(spikes)))
            return None,torch.tensor(23)
        def readout(found,t,adj,q):
            calls.append((t,q));return np.array([[0,1],[2,2]])
        a,b,result=predict_pair(None,trace,None,torch.tensor(7),readout,correlate)
        self.assertEqual([int(q) for _,q in calls],[7,23])
        self.assertIs(calls[0][0],trace);self.assertIs(calls[1][0]['theta'],trace['theta'])
        self.assertTrue(torch.equal(old,components));self.assertTrue(result['labels_bitwise_equal'])
        self.assertNotEqual(trace['spikes'][0,0,0],calls[1][0]['spikes'][0,0,0])

    def test_partition_comparison_ignores_ID_permutation_but_preserves_background(self):
        a=np.array([[0,4],[8,8]]);b=np.array([[0,7],[2,2]])
        self.assertTrue(compare_labels(a,b)['partition_equal_up_to_foreground_ID_permutation'])
        self.assertFalse(compare_labels(a,b)['labels_bitwise_equal'])
        self.assertFalse(compare_labels(a,np.array([[0,7],[7,7]]))['partition_equal_up_to_foreground_ID_permutation'])
        self.assertEqual(compare_labels(a,np.array([[7,0],[2,2]]))['foreground_changed_patch_count'],2)

    def test_invalid_gates_rejected(self):
        with self.assertRaises(ValueError):gate_only_trace(dict(component_spikes=torch.zeros(1,4,2,2),component_gates=torch.ones(1,2,2)*2))

    def test_actual_four_component_gate_geometry(self):
        gates=torch.arange(16,dtype=torch.float32).reshape(1,4,2,2)/16
        trace=dict(component_spikes=torch.zeros_like(gates),component_gates=gates)
        result=gate_only_trace(trace)
        self.assertTrue(torch.equal(result['component_spikes'],gates))
        self.assertTrue(torch.equal(result['spikes'],gates.mean(1)))

    def test_queue_routes_counterfactual_worker_and_rejects_invalid_GPU(self):
        from . import queue,run
        self.assertIs(queue.api,run)
        self.assertEqual(queue.MODULE,'collaborative_test.SW_0241_native64_event_readout_counterfactual.queue')
        self.assertEqual(queue.worker_command(2,3)[-1],'3')
        with self.assertRaises(ValueError):queue.worker_command(2,4)

    def test_real_native_Q_changes_with_event_information(self):
        from collaborative_test.SW_0175_critical_component_binding.critical_component_objective import native_component_rho_q
        rng=torch.Generator().manual_seed(241)
        shared=torch.rand((1,4,1,520),generator=rng)
        gates=.8*shared.expand(1,4,3,520)+.1*torch.rand((1,4,3,520),generator=rng)
        spikes=gates.clone();spikes[:,0,0,-5:]=0
        trace=dict(component_gates=gates,component_spikes=spikes,spikes=spikes.mean(1))
        replacement=gate_only_trace(trace)
        _,before=native_component_rho_q(trace['spikes'],spikes,settle=512)
        _,after=native_component_rho_q(replacement['spikes'],replacement['component_spikes'],settle=512)
        self.assertGreater(float((before-after).abs().max()),0)


if __name__=='__main__':unittest.main()
