import contextlib
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid
import numpy as np
import torch
from . import train


def temporary():
    p=train.HERE.parents[1]/'collaborative_testing/temporary'/('sw0218_'+uuid.uuid4().hex)
    p.mkdir(parents=True,exist_ok=False);return p


def fixture():
    root=temporary();parent=root/'source';parent.mkdir();(parent/'boundary.ptx').write_bytes(b'parent')
    wrapped=torch.nn.Module()
    for name in ('context','graph','core','integration'):wrapped.register_parameter(name,torch.nn.Parameter(torch.tensor(.5)))
    wrapped.project_integrations_=lambda:None
    encoder=torch.nn.Linear(1,1,bias=False);encoder.weight.data.fill_(.5)
    found=SimpleNamespace(wrapped=wrapped,encoder=encoder,patcher=torch.nn.Identity())
    names=('context_encoder','graph','native_core','integration','source_base')
    families={name:[(name,p)] for name,p in zip(names,list(wrapped.parameters())+list(encoder.parameters()))}
    optimizer=torch.optim.Adam([dict(name=name,params=[p for _,p in entries],lr=.01) for name,entries in families.items()])
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        sum(p.square().sum() for entries in families.values() for _,p in entries).backward();optimizer.step()
    history=[dict(proposal=i+1,indices=list(range(i*4,(i+1)*4)),accepted=True,parity_error=0.,acceptance_threshold=1e-6) for i in range(2)]
    runtime=train.probe.resume_policy.compatible_runtime({'device_type':'cuda','cuda_visible_devices':'2'})
    prepared=dict(found=found,optimizer=optimizer,families=families,parent_path=parent,
        parent_runtime=runtime,parent_bindings={'source':'exact'},parent_history=history,
        parent_model_sha256=train.probe.native._state_sha(wrapped,encoder),
        warm_CNN_sha256='warmCNN',bindings=dict(parent_boundary_sha256=train.sha(b'parent'),experiment='child'))
    prepared['parent_optimizer_content_sha256']=train.content_sha(optimizer.state_dict())
    prepared['parent_RNG_content_sha256']=train.content_sha({'fixture_RNG':'restored'})
    return root,prepared,runtime


class ContinuationTests(unittest.TestCase):
    def execute(self,reject=False):
        root,p,runtime=fixture();calls=[];saves=[]
        initial=train.probe.tx.capture_transaction({'wrapped':p['found'].wrapped,'encoder':p['found'].encoder},p['optimizer'])
        def load(*args,**kwargs):
            self.assertEqual(Path(args[0]),p['parent_path']);self.assertEqual(kwargs['max_updates'],2)
            self.assertEqual(kwargs['expected_bindings'],p['parent_bindings'])
            return dict(completed_updates=2,history=copy.deepcopy(p['parent_history']))
        def save(path,**kwargs):
            Path(path).mkdir(parents=True,exist_ok=True);saves.append((kwargs['completed_updates'],len(kwargs['history'])))
        def loss(prepared,indices,device,*,live,credit=False):
            calls.append((list(indices),live,credit))
            value=sum(parameter.square().sum() for group in p['families'].values() for _,parameter in group)
            measured=float(value.detach())
            if live:value.backward()
            if len(calls)==3:
                if reject=='nonfinite':measured=float('nan')
                elif reject=='raises':raise FloatingPointError('fixture nonfinite rollout')
                elif reject:measured+=1.
            return measured,.5
        original_step=p['optimizer'].step
        def corrupt_step():
            original_step()
            next(p['found'].wrapped.parameters()).data.fill_(float('inf'))
        with contextlib.ExitStack() as stack:
            if reject=='bad_parameter':stack.enter_context(patch.object(p['optimizer'],'step',side_effect=corrupt_step))
            stack.enter_context(patch.object(train,'TOTAL',3))
            for name in ('set_device','reset_peak_memory_stats','manual_seed','synchronize'):stack.enter_context(patch.object(torch.cuda,name))
            stack.enter_context(patch.object(torch.cuda,'max_memory_allocated',return_value=1234))
            stack.enter_context(patch.object(train.probe.model64,'move_to_device',side_effect=lambda found,*_:found))
            stack.enter_context(patch.object(train.probe.checkpoint,'runtime_signature',return_value={'device_type':'cuda','cuda_visible_devices':'2'}))
            stack.enter_context(patch.object(train.probe.checkpoint,'load_latest',side_effect=load))
            stack.enter_context(patch.object(train.probe.checkpoint,'capture_rng',return_value={'fixture_RNG':'restored'}))
            stack.enter_context(patch.object(train.probe.checkpoint,'save_boundary',side_effect=save))
            stack.enter_context(patch.object(train,'batch_loss',side_effect=loss))
            stack.enter_context(patch.object(train.probe.warm,'_base_encoder_sha',return_value='toyCNN'))
            report=train.run_training(p,work=root/'work',output=root/'report.json')
        return p,report,calls,saves,initial

    def test_continues_actual_existing_Adam_without_repeating_prefix_images(self):
        p,r,calls,saves,_=self.execute()
        self.assertEqual(r['accepted_updates'],3);self.assertEqual(r['actual_optimizer_steps'],[3])
        self.assertEqual(saves,[(2,2),(3,3)])
        self.assertEqual([row[0] for row in calls],[[8,9,10,11]]*3)
        self.assertEqual([row[1] for row in calls],[False,True,False])
        self.assertEqual([row[2] for row in calls],[False,True,False])
        self.assertEqual(r['history'][:2],p['parent_history'])
        self.assertEqual(r['history'][2]['emitted_S_gradient_l2'],.5)

    def test_rejection_restores_inherited_model_AND_nonempty_Adam_moments(self):
        p,r,_calls,_saves,initial=self.execute('nonfinite')
        self.assertEqual(r['accepted_updates'],2);self.assertEqual(r['actual_optimizer_steps'],[2])
        current=train.probe.tx.capture_transaction({'wrapped':p['found'].wrapped,'encoder':p['found'].encoder},p['optimizer'])
        # Proposal RNG is intentionally reseeded; compare actual modules/Adam.
        current['rng']=initial['rng']
        self.assertEqual(train.probe.tx.transaction_digest(current),train.probe.tx.transaction_digest(initial))


    def test_finite_uphill_proposal_keeps_new_weights_and_Adam_moments(self):
        p,r,_calls,_saves,initial=self.execute(True)
        self.assertEqual(r['actual_optimizer_steps'],[3])
        self.assertTrue(r['history'][2]['finite_step_applied'])
        self.assertFalse(r['history'][2]['descent_accepted'])
        current=train.probe.tx.capture_transaction({'wrapped':p['found'].wrapped,'encoder':p['found'].encoder},p['optimizer'])
        self.assertNotEqual(train.probe.tx.transaction_digest(initial),train.probe.tx.transaction_digest(current))

    def test_nonfinite_rollout_exception_restores_transaction_without_bad_JSON(self):
        p,r,_calls,_saves,initial=self.execute('raises')
        self.assertEqual(r['actual_optimizer_steps'],[2])
        self.assertFalse(r['history'][2]['finite_step_applied'])
        self.assertIsNone(r['history'][2]['after_loss'])
        current=train.probe.tx.capture_transaction({'wrapped':p['found'].wrapped,'encoder':p['found'].encoder},p['optimizer'])
        self.assertEqual(train.probe.tx.transaction_digest(initial),train.probe.tx.transaction_digest(current))

    def test_nonfinite_parameter_update_is_restored_before_post_evaluation(self):
        p,r,calls,_saves,initial=self.execute('bad_parameter')
        self.assertEqual(len(calls),2)
        self.assertEqual(r['actual_optimizer_steps'],[2])
        row=r['history'][2]
        self.assertFalse(row['parameters_finite_after_proposal'])
        self.assertIsNone(row['proposed_family_delta_l2']['context_encoder'])
        self.assertFalse(row['finite_step_applied'])
        current=train.probe.tx.capture_transaction({'wrapped':p['found'].wrapped,'encoder':p['found'].encoder},p['optimizer'])
        self.assertEqual(train.probe.tx.transaction_digest(initial),train.probe.tx.transaction_digest(current))

    def test_changed_parent_checkpoint_is_rejected_before_restore(self):
        root,p,runtime=fixture();(p['parent_path']/'boundary.ptx').write_bytes(b'changed')
        with patch.object(train.probe.checkpoint,'load_latest') as restore:
            with self.assertRaises(ValueError):train.restore_for_training(p,root/'work','cpu',runtime)
            restore.assert_not_called()

    def test_own_committed_boundary_is_preferred_and_no_parent_prefix_is_resaved(self):
        root,p,runtime=fixture();work=root/'work';work.mkdir();(work/'boundary.ptx').write_bytes(b'existing')
        with patch.object(train.probe.checkpoint,'load_latest',return_value=dict(completed_updates=2,history=copy.deepcopy(p['parent_history']))) as restore,patch.object(train.probe.checkpoint,'save_boundary') as save:
            history=train.restore_for_training(p,work,'cpu',runtime)
            self.assertEqual(history,p['parent_history']);self.assertEqual(Path(restore.call_args.args[0]),work)
            self.assertEqual(restore.call_args.kwargs['max_updates'],4);save.assert_not_called()

    def test_changed_prefix_or_TRAIN64_order_is_rejected(self):
        _,p,_=fixture();bad=copy.deepcopy(p['parent_history']);bad[0]['accepted']=False
        with self.assertRaises(ValueError):train.validate_history(bad,p['parent_history'])
        bad=copy.deepcopy(p['parent_history'])+[dict(proposal=3,indices=[0,1,2,3],accepted=True,parity_error=0.,acceptance_threshold=1e-6)]
        with self.assertRaises(ValueError):train.validate_history(bad,p['parent_history'])

    def test_content_hash_ignores_tensor_storage_and_dict_insertion_order(self):
        value=torch.arange(4.)
        self.assertEqual(train.content_sha({'Adam':value,'RNG':(1,2)}),train.content_sha({'RNG':(1,2),'Adam':value.clone()}))
        self.assertNotEqual(train.content_sha(value),train.content_sha(value+1))

    def test_wrong_restored_RNG_is_rejected_before_first_new_boundary(self):
        root,p,runtime=fixture()
        loaded=dict(completed_updates=2,history=copy.deepcopy(p['parent_history']))
        with patch.object(train.probe.checkpoint,'load_latest',return_value=loaded),patch.object(train.probe.checkpoint,'capture_rng',return_value={'wrong':'RNG'}),patch.object(train.probe.checkpoint,'save_boundary') as save:
            with self.assertRaises(ValueError):train.restore_for_training(p,root/'work','cpu',runtime)
            save.assert_not_called()

if __name__=='__main__':unittest.main()
