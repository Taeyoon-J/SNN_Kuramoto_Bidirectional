from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from . import train


class ContextPipelineTests(unittest.TestCase):
    def setup_fixture(self):
        gain=torch.nn.Parameter(torch.tensor(.1,dtype=torch.float64))
        found=SimpleNamespace(wrapped=SimpleNamespace(core=SimpleNamespace(graph_generator=SimpleNamespace(log_coupling_gain=gain))))
        prepared=dict(found=found,labels=np.array([[[1,1],[2,2]]]),_ranking_contexts={})
        return prepared,gain

    def test_mine_once_before_then_freeze_through_gradient_and_after(self):
        prepared,gain=self.setup_fixture();pred=np.ones((2,2),np.int64)
        with patch.object(train.screen,'prediction_from_trace',return_value=pred) as miner:
            def objective(value,live):
                emitted=torch.tensor(value,dtype=torch.float64,requires_grad=live)
                q=torch.sigmoid(emitted).expand(1,4,4);adj=torch.ones_like(q)*.002
                rho=(2*q-1)[:,None].expand(1,4,4,4)
                total,aux=train.combined_objective(prepared,0,{'component_spikes':emitted},adj,q,rho,live=live)
                return total,aux,emitted
            before,_,_=objective(1.,False)
            context=train.context_record(prepared['_ranking_contexts'][0])
            measured,aux,emitted=objective(1.,True)
            self.assertEqual(float(before.detach()),float(measured.detach()))
            gradients=torch.autograd.grad(aux,(emitted,gain),retain_graph=True)
            self.assertTrue(all(torch.isfinite(g) and g!=0 for g in gradients))
            measured.backward();self.assertTrue(torch.isfinite(emitted.grad))
            objective(-1.,False)
            self.assertEqual(miner.call_count,1)
            self.assertEqual(train.context_record(prepared['_ranking_contexts'][0]),context)

    def test_gradient_pass_cannot_create_new_context(self):
        prepared,_=self.setup_fixture();q=torch.full((1,4,4),.8,requires_grad=True)
        with patch.object(train.screen,'prediction_from_trace') as miner:
            with self.assertRaises(ValueError):train.combined_objective(prepared,0,{},q*0,q,q[:,None].expand(1,4,4,4),live=True)
            miner.assert_not_called()

    def test_GT_barrier_until_candidate_protocol_is_valid(self):
        from . import evaluate
        with patch.object(evaluate,'validated_candidate',side_effect=ValueError('no prediction')),patch.object(evaluate.screen_score,'read_ground_truth') as reader:
            with self.assertRaises(ValueError):evaluate.score_candidate()
            reader.assert_not_called()


if __name__=='__main__':unittest.main()
