import unittest
import contextlib
import copy
import json
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch
from . import train,evaluate
from collaborative_test.SW_0134_native_spike_binding.test_rollout import _core
from collaborative_test.SW_0130_phase_state_integration.model import PhaseStateIntegration
from collaborative_test.SW_0134_native_spike_binding.rollout import late_rollout
from .test_training import fixture


class CreditTests(unittest.TestCase):
    def test_actual_rollout_values_unchanged_by_credit_boundary(self):
        torch.manual_seed(200)
        wrapped=PhaseStateIntegration(_core(), 'phase')
        gamma=torch.randn(1,8,64)
        results={}
        for tail in (0,8,32):
            results[tail]=late_rollout(wrapped,gamma,total_steps=64,live_tail_steps=tail)
        for tail in (8,32):
            for key in ('theta','component_membrane','component_spikes','component_gates'):
                self.assertTrue(torch.equal(results[0][key],results[tail][key]),key)
        self.assertEqual(results[32]['prefix_steps'],32)
        self.assertEqual(results[32]['live_tail_steps'],32)
        self.assertTrue(results[32]['component_spikes'].requires_grad)
        self.assertFalse(results[0]['component_spikes'].requires_grad)

    def test_training_uses512_credit_and_evaluation_zero(self):
        scalar=torch.tensor(.3,requires_grad=True)
        q=scalar.reshape(1,1,1).expand(1,4096,4096)
        gamma=torch.zeros(1,8,4096)
        prepared=dict(found=object(),rgb=np.zeros((1,3,128,128),dtype=np.float32),labels=np.zeros((1,64,64),dtype=np.int64))
        with patch.object(train.probe.native,'_native_forward',return_value=(gamma,{},None,None,q,None)) as forward,patch.object(train,'_vectorized_class_loss',side_effect=lambda values,labels,**kw:values[0,0]):
            train.batch_loss(prepared,[0],'cpu',live=False)
            train.batch_loss(prepared,[0],'cpu',live=True,credit=False)
        self.assertEqual([c.kwargs['live_tail_steps'] for c in forward.call_args_list],[0,512])
        self.assertGreater(float(scalar.grad),0)

    def test_long_credit_still_requires_committed_learning_and_warm_floor(self):
        x=np.zeros((16,3));pair={k:evaluate.paired_change(x+.02,x) for k in ('control','warm')}
        report=dict(new_accepted_updates=1,committed_model_changed=True,CNN_changed_from_warm=True)
        self.assertTrue(evaluate.extension_gate(pair,report))
        report['new_accepted_updates']=0
        self.assertFalse(evaluate.extension_gate(pair,report))
        report['new_accepted_updates']=1;pair['warm']['fg_ari']['mean']=-.001
        self.assertFalse(evaluate.extension_gate(pair,report))

    def test_window_covers_scored_tail_but_preserves_prefix(self):
        self.assertEqual(train.LIVE_TAIL,512)
        self.assertEqual(train.fingerprint()['total_steps']-train.LIVE_TAIL,512)
        self.assertEqual(evaluate.OFFSETS,tuple(range(0,32,2)))

    def admission_fixture(self,mode):
        root,prepared,_runtime=fixture();optimizer=prepared['optimizer'];found=prepared['found']
        before=train.probe.tx.capture_transaction(dict(wrapped=found.wrapped,encoder=found.encoder),optimizer)
        def loss(_p,_indices,_device,*,live,credit=False):
            if mode=='OOM' and live:raise torch.cuda.OutOfMemoryError('bounded fixture')
            value=sum(p.square().sum() for entries in prepared['families'].values() for _,p in entries)
            if live:value.backward()
            return float(value.detach()),float('nan') if mode=='nonfinite' else .5
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(train,'ARCHIVE',root))
            for name in ('set_device','set_per_process_memory_fraction','reset_peak_memory_stats','manual_seed','empty_cache'):
                stack.enter_context(patch.object(torch.cuda,name))
            stack.enter_context(patch.object(torch.cuda,'max_memory_allocated',return_value=1234))
            stack.enter_context(patch.object(train.probe.model64,'move_to_device',side_effect=lambda found,*_:found))
            stack.enter_context(patch.object(train.probe.checkpoint,'runtime_signature',return_value={'device_type':'cuda','cuda_visible_devices':'2'}))
            stack.enter_context(patch.object(train.probe.checkpoint,'load_latest',return_value=dict(completed_updates=2,history=copy.deepcopy(prepared['parent_history']))))
            stack.enter_context(patch.object(train.probe.checkpoint,'capture_rng',return_value={'fixture_RNG':'restored'}))
            stack.enter_context(patch.object(train.probe.checkpoint,'save_boundary'))
            stack.enter_context(patch.object(train,'batch_loss',side_effect=loss))
            step=stack.enter_context(patch.object(optimizer,'step'))
            passed=evaluate.gradient_admission(prepared)
            step.assert_not_called()
        after=train.probe.tx.capture_transaction(dict(wrapped=found.wrapped,encoder=found.encoder),optimizer)
        self.assertEqual(train.probe.tx.transaction_digest(before),train.probe.tx.transaction_digest(after))
        self.assertTrue(torch.equal(before['rng']['torch_cpu'],after['rng']['torch_cpu']))
        path=root/'gradient_admission'/('passed.json' if passed else 'rejected.json')
        return passed,json.loads(path.read_bytes())

    def test_actual_admission_transaction_restores_weights_Adam_rng_without_step(self):
        passed,receipt=self.admission_fixture('pass')
        self.assertTrue(passed);self.assertEqual(receipt['live_tail_steps'],512)
        self.assertEqual(receipt['optimizer_updates'],0)

    def test_OOM_admission_is_terminal_scientific_rejection_and_restores_state(self):
        passed,receipt=self.admission_fixture('OOM')
        self.assertFalse(passed);self.assertIn('bounded allocator',receipt['failure'])
        self.assertTrue(receipt['model_Adam_RNG_restored'])

    def test_nonfinite_credit_rejection_remains_valid_finite_json(self):
        passed,receipt=self.admission_fixture('nonfinite')
        self.assertFalse(passed);self.assertEqual(receipt['failure'],'nonfinite credit')
        self.assertIsNone(receipt['emitted_S_gradient_l2'])
