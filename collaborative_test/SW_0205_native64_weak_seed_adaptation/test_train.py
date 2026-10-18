import contextlib
import json
from pathlib import Path
import uuid
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from . import run


class ProbeTests(unittest.TestCase):
    def setUp(self):run.configure(1)

    def execute(self,reject=False):
        wrapped=torch.nn.Module()
        for name in ('ctx','graph','core','integrate'):wrapped.register_parameter(name,torch.nn.Parameter(torch.tensor(.5)))
        wrapped.project_integrations_=lambda:None
        encoder=torch.nn.Linear(1,1,bias=False);encoder.weight.data.fill_(.5)
        found=SimpleNamespace(wrapped=wrapped,encoder=encoder,patcher=torch.nn.Identity())
        families={name:[(name,p)] for name,p in zip(('context_encoder','graph','native_core','integration','source_base'),list(wrapped.parameters())+list(encoder.parameters()))}
        optimizer=torch.optim.Adam([{'name':name,'params':[p for _,p in entries],'lr':.01} for name,entries in families.items()])
        prepared=dict(found=found,optimizer=optimizer,families=families,
            bindings={'factory_proof':{'source_CNN_sha256':'initial'}})
        initial={k:v.clone() for k,v in wrapped.state_dict().items()};initial_encoder=encoder.weight.detach().clone()
        calls=[];saved=[]
        def batch(_prepared,indices,device,*,live,credit=False):
            calls.append((tuple(indices),live,credit))
            loss=sum(p.square().sum() for entries in families.values() for _,p in entries)
            value=float(loss.detach())
            if live:loss.backward()
            if reject and len(calls)%3==0:value+=1.
            return value,1. if credit else 0.
        def save(root,**kwargs):
            Path(root).mkdir(parents=True,exist_ok=True)
            saved.append((kwargs['completed_updates'],len(kwargs['history'])))
        tmp=run.HERE.parents[1]/'collaborative_testing/temporary'/('sw0192_'+uuid.uuid4().hex)
        tmp.mkdir(parents=True,exist_ok=False)
        with contextlib.ExitStack() as stack:
            for name in ('set_device','reset_peak_memory_stats','synchronize','manual_seed'):
                stack.enter_context(patch.object(torch.cuda,name))
            stack.enter_context(patch.object(torch.cuda,'max_memory_allocated',return_value=1234))
            stack.enter_context(patch.object(run.checkpoint,'runtime_signature',return_value={'device_type':'cuda','cuda_visible_devices':'2'}))
            stack.enter_context(patch.object(run.checkpoint,'save_boundary',side_effect=save))
            stack.enter_context(patch.object(run.model64,'move_to_device',side_effect=lambda *args:found))
            stack.enter_context(patch.object(run,'batch_loss',side_effect=batch))
            stack.enter_context(patch.object(run.warm,'_base_encoder_sha',return_value='initial' if reject else 'changed'))
            report=run.run_probe(prepared,Path(tmp)/'report.json',Path(tmp)/'work')
            self.assertEqual(json.loads((Path(tmp)/'report.json').read_text())['status'],report['status'])
        return report,calls,saved,wrapped,encoder,optimizer,initial,initial_encoder

    def test_real_transaction_Adam_accepts_and_records_all_five_families(self):
        report,calls,saved,*_=self.execute()
        self.assertEqual(report['status'],'complete_native64_training_admission')
        self.assertEqual(report['actual_optimizer_steps'],[2]);self.assertEqual(report['accepted_updates'],2)
        self.assertEqual(saved,[(0,0),(1,1),(2,2)])
        self.assertEqual([v[1] for v in calls],[False,True,False]*2)
        self.assertEqual(calls[0][0],tuple(range(4)));self.assertEqual(calls[3][0],tuple(range(4,8)))
        for row in report['history']:
            self.assertEqual(row['credit_failures'],[])
            self.assertGreater(row['emitted_S_gradient_l2'],0)
            self.assertTrue(all(v>0 for v in row['proposed_family_delta_l2'].values()))

    def test_rejected_proposals_restore_both_modules_and_empty_Adam(self):
        report,_,saved,wrapped,encoder,optimizer,initial,initial_encoder=self.execute(True)
        self.assertEqual(report['status'],'complete_rejected_no_committed_learning')
        self.assertEqual(report['accepted_updates'],0);self.assertEqual(report['actual_optimizer_steps'],[])
        self.assertFalse(optimizer.state)
        for key,value in wrapped.state_dict().items():self.assertTrue(torch.equal(value,initial[key]))
        self.assertTrue(torch.equal(encoder.weight,initial_encoder))

    def test_reject_wrong_seed_configuration(self):
        for seed in (0,3,True):
            with self.assertRaises(ValueError):run.configure(seed)

    def test_report_cannot_come_from_another_seed(self):
        report=dict(seed=2,bindings={},validation_data_read=False)
        with self.assertRaises(ValueError):run.validate_report(report,{})

    def test_completed_training_resumes_inference_without_retraining(self):
        report,_,_,wrapped,encoder,optimizer,_,_=self.execute()
        root=run.HERE.parents[1]/'collaborative_testing/temporary'/('sw0205_resume_'+uuid.uuid4().hex)
        (root/'training').mkdir(parents=True)
        run.write_json(root/'training/report.json',report)
        found=SimpleNamespace(wrapped=wrapped,encoder=encoder)
        prepared=dict(found=found,optimizer=optimizer,bindings=report['bindings'])
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(run,'ARCHIVE',root))
            stack.enter_context(patch.object(run,'prepare_inputs_cpu',return_value=prepared))
            for name in ('set_device','set_per_process_memory_fraction'):
                stack.enter_context(patch.object(torch.cuda,name))
            stack.enter_context(patch.object(run.model64,'move_to_device',return_value=found))
            stack.enter_context(patch.object(run.checkpoint,'runtime_signature',return_value={'device_type':'cuda','cuda_visible_devices':'1'}))
            loaded=stack.enter_context(patch.object(run.checkpoint,'load_latest',return_value=dict(completed_updates=2,history=report['history'])))
            training=stack.enter_context(patch.object(run,'run_probe',side_effect=AssertionError('must not retrain')))
            from . import evaluate
            inference=stack.enter_context(patch.object(evaluate,'predict_candidate'))
            run.predict(expected_binding=report['bindings'])
            self.assertTrue(loaded.called);training.assert_not_called();inference.assert_called_once()

    def test_queue_cannot_request_a_nonexistent_GPU(self):
        from .queue import worker_command
        with self.assertRaises(ValueError):worker_command(4)
        argv=worker_command(3)
        self.assertIn('collaborative_test.SW_0205_native64_weak_seed_adaptation.queue',argv)
        self.assertEqual(argv[argv.index('--physical-gpu')+1],'3')
