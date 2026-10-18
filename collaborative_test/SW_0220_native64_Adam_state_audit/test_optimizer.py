import json
import contextlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from .optimizer_audit import compare_steps,decision
from . import queue,run
from collaborative_test.SW_0181_actual_descent_qce import state_transaction as tx


def fixture():
    wrapped=torch.nn.Linear(1,1,bias=False);encoder=torch.nn.Linear(1,1,bias=False)
    for module in (wrapped,encoder):module.weight.data.fill_(.5)
    wrapped.project_integrations_=lambda:None
    found=SimpleNamespace(wrapped=wrapped,encoder=encoder)
    families={'native_core':[('wrapped',wrapped.weight)],'source_base':[('encoder',encoder.weight)]}
    optimizer=torch.optim.Adam([dict(name=name,params=[p for _,p in entries],lr=.05) for name,entries in families.items()])
    for group in optimizer.param_groups:
        for p in group['params']:
            optimizer.state[p]=dict(step=torch.tensor(100.),exp_avg=torch.full_like(p,-1.),exp_avg_sq=torch.full_like(p,.1))
    modules=dict(wrapped=wrapped,encoder=encoder);snapshot=tx.capture_transaction(modules,optimizer)
    gradients={id(p):torch.ones_like(p) for entries in families.values() for _,p in entries}
    return found,families,optimizer,modules,snapshot,gradients


class OptimizerAuditTests(unittest.TestCase):
    def test_same_gradient_old_state_uphill_reset_downhill_and_exact_restoration(self):
        found,families,optimizer,modules,snapshot,gradients=fixture()
        before=.5;measure=lambda:sum(float(p.detach().square().sum()) for entries in families.values() for _,p in entries)
        arms=compare_steps(found,optimizer,families,gradients,snapshot,measure)
        self.assertGreater(arms['inherited_Adam']['after_loss'],before)
        self.assertLess(arms['reset_Adam']['after_loss'],before)
        self.assertGreater(arms['inherited_Adam']['surrogate_gradient_dot_projected_delta'],0)
        self.assertLess(arms['reset_Adam']['surrogate_gradient_dot_projected_delta'],0)
        self.assertEqual(arms['inherited_Adam']['steps_before'],[100]);self.assertEqual(arms['inherited_Adam']['steps_after'],[101])
        self.assertEqual(arms['reset_Adam']['steps_before'],[]);self.assertEqual(arms['reset_Adam']['steps_after'],[1])
        self.assertEqual(tx.transaction_digest(snapshot),tx.transaction_digest(tx.capture_transaction(modules,optimizer)))
        outcome=decision(before,arms);self.assertTrue(outcome['TRAIN_only_followup_eligible']);self.assertFalse(outcome['segmentation_improvement_verified'])
        json.dumps(dict(arms=arms,decision=outcome),allow_nan=False)

    def test_exception_in_actual_measure_restores_model_Adam_and_RNG(self):
        found,families,optimizer,modules,snapshot,gradients=fixture()
        def fail():raise RuntimeError('forward fixture failure')
        with self.assertRaises(RuntimeError):compare_steps(found,optimizer,families,gradients,snapshot,fail)
        self.assertEqual(tx.transaction_digest(snapshot),tx.transaction_digest(tx.capture_transaction(modules,optimizer)))

    def test_projected_delta_is_measured_after_projection(self):
        found,families,optimizer,_,snapshot,gradients=fixture()
        def project():
            with torch.no_grad():found.wrapped.weight.clamp_(-.51,.51)
        found.wrapped.project_integrations_=project
        arms=compare_steps(found,optimizer,families,gradients,snapshot,lambda:1.)
        self.assertAlmostEqual(arms['inherited_Adam']['family_parameter_delta_l2']['native_core'],.01,places=6)

    def test_nonfinite_proposals_skip_forward_serialize_and_restore(self):
        found,families,optimizer,modules,snapshot,gradients=fixture();original=optimizer.step
        def corrupt():
            original();found.wrapped.weight.data.fill_(float('inf'))
        with patch.object(optimizer,'step',side_effect=corrupt),patch('builtins.print'):
            arms=compare_steps(found,optimizer,families,gradients,snapshot,lambda:self.fail('must not forward nonfinite parameters'))
        self.assertTrue(all(v['after_loss'] is None for v in arms.values()));json.dumps(arms,allow_nan=False)
        self.assertEqual(tx.transaction_digest(snapshot),tx.transaction_digest(tx.capture_transaction(modules,optimizer)))

    def test_missing_shared_gradient_or_aliased_family_rejected(self):
        found,families,optimizer,_,snapshot,gradients=fixture();gradients.pop(next(iter(gradients)))
        with self.assertRaises(ValueError):compare_steps(found,optimizer,families,gradients,snapshot,lambda:0.)

    def test_no_followup_from_nonfinite_or_nonimproving_measure(self):
        for old,fresh in ((None,1.),(1.,None),(.9,.95),(1.,1.)):
            result=decision(1.,{a:{'after_loss':x} for a,x in zip(('inherited_Adam','reset_Adam'),(old,fresh))})
            self.assertFalse(result['TRAIN_only_followup_eligible'])

    def test_real_control_import_and_fixed_native_source_GPU_command(self):
        self.assertEqual(run.recover_score.control.HERE.name,'SW_0204_native64_component_credit')
        self.assertEqual(run.INDICES,(8,9,10,11));self.assertEqual(queue.worker_command(3)[-2:],['--physical-gpu','3'])
        for invalid in (True,-1,4):
            with self.assertRaises(ValueError):queue.worker_command(invalid)
        owner=queue.load('collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4')
        for name in ('_foreign_owners','_process_tree','_gpu_info'):
            self.assertTrue(callable(getattr(owner.gpu_guard,name)))

    def test_complete_diagnosis_report_path_restores_source_without_training_outputs(self):
        found,families,optimizer,modules,snapshot,_=fixture();source=run.source
        model_sha=source.probe.native._state_sha(found.wrapped,found.encoder)
        rng={'fixture':'source CUDA RNG content'};runtime=source.probe.resume_policy.compatible_runtime({'fixture_runtime':True})
        binding=dict(parent_model_sha256=model_sha,parent_optimizer_content_sha256=source.content_sha(optimizer.state_dict()),parent_RNG_content_sha256=source.content_sha(rng))
        prepared=dict(found=found,families=families,optimizer=optimizer,parent_runtime=runtime,parent_path='fixture',parent_bindings={},
            parent_model_sha256=model_sha,parent_optimizer_content_sha256=binding['parent_optimizer_content_sha256'],
            parent_RNG_content_sha256=binding['parent_RNG_content_sha256'],audit_binding=binding,
            reference218_first_proposal=dict(ranking_contexts={},before_loss=.5,after_loss=.6))
        def measured(prepared,indices,device,*,live,credit=False):
            self.assertEqual(tuple(indices),(8,9,10,11))
            objective=sum(p.square().sum() for entries in families.values() for _,p in entries)
            value=float(objective.detach())
            if live:objective.backward()
            return value,.5
        with contextlib.ExitStack() as stack:
            for name in ('set_device','manual_seed'):stack.enter_context(patch.object(torch.cuda,name))
            stack.enter_context(patch.object(source.probe.model64,'move_to_device',side_effect=lambda found,*_:found))
            stack.enter_context(patch.object(source.probe.checkpoint,'runtime_signature',return_value=runtime))
            stack.enter_context(patch.object(source.probe.checkpoint,'capture_rng',return_value=rng))
            stack.enter_context(patch.object(source.probe.checkpoint,'load_latest',return_value={'completed_updates':2}))
            stack.enter_context(patch.object(source,'batch_loss',side_effect=measured))
            writer=stack.enter_context(patch.object(run,'write_json'))
            report=run.run_audit(prepared,device='cpu')
        self.assertTrue(report['source_transaction_exactly_restored'])
        self.assertEqual(report['restored_model_sha256'],model_sha)
        self.assertEqual(report['native_forward_images_completed'],16);self.assertEqual(report['committed_optimizer_updates'],0)
        self.assertTrue(report['decision']['TRAIN_only_followup_eligible']);json.dumps(report,allow_nan=False)
        self.assertEqual(writer.call_count,1)


if __name__=='__main__':unittest.main()
