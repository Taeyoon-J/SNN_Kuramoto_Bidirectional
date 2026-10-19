import copy,io,json,unittest,uuid
from pathlib import Path
from unittest import mock
import numpy as np
import torch
from . import train,projection,bootstrap,prefix_guard
from .test_pipeline import prepared


class BootstrapTests(unittest.TestCase):
    def source(self):
        p=prepared();root=train.ROOT/'collaborative_testing/temporary'/('b236_'+uuid.uuid4().hex);root.mkdir()
        old=root/'s';old.mkdir();candidate=root/'c';candidate.mkdir()
        for parameter in list(p['found'].encoder.parameters())+projection.parameters(p['found']):parameter.grad=torch.zeros_like(parameter)
        p['optimizer'].step();p['optimizer'].zero_grad(set_to_none=True)
        for state in p['optimizer'].state.values():state['step'].fill_(128)
        runtime=train.resume_policy.compatible_runtime(train.checkpoint.runtime_signature('cpu'))
        old_bindings={'fixture_source':True};history=[dict(update=i+1,loss=.01) for i in range(128)]
        train.checkpoint.save_boundary(old/'encoder_work',completed_updates=128,wrapped=projection.module(p['found']),encoder=p['found'].encoder,
            optimizer=p['optimizer'],history=history,bindings=old_bindings,runtime=runtime,device='cpu')
        raw=(old/'encoder_work/boundary.ptx').read_bytes()
        report=dict(final_model_sha256=bootstrap.model_sha(p),checkpoint_file_sha256=train.sha(raw),checkpoint_binding=old_bindings)
        train.write_json(old/'training_report.json',report)
        p['binding']['matched_128_control']=dict(report_sha256=train.sha((old/'training_report.json').read_bytes()),model_sha256=report['final_model_sha256'])
        return p,old,candidate,runtime

    def test_actual_checkpoint_model_Adam_history_and_rng_restore_into_new_branch(self):
        p,old,target,runtime=self.source()
        with torch.no_grad():next(p['found'].encoder.parameters()).add_(1.)
        p['optimizer'].state.clear()
        binding=dict(binding=p['binding'],child_branch=True)
        with mock.patch.object(train.accepted,'archive_for',return_value=old),mock.patch.object(train,'archive_for',return_value=target):
            history=bootstrap.start(p,binding,target/'encoder_work',runtime,torch.device('cpu'))
            self.assertEqual(len(history),128);proof=prefix_guard.validate_saved(p)
            self.assertTrue(proof['encoder_projection_Adam_history_and_Torch_RNG_bitwise_equal'])
            self.assertEqual(sorted({int(v['step']) for v in p['optimizer'].state.values()}),[128])
            self.assertEqual(bootstrap.model_sha(p),p['binding']['matched_128_control']['model_sha256'])
            with self.assertRaises(ValueError):bootstrap.start(p,binding,target/'encoder_work',runtime,torch.device('cpu'))

    def test_changed_checkpoint_and_reset_Adam_cannot_be_admitted(self):
        p,old,_,_=self.source()
        with mock.patch.object(train.accepted,'archive_for',return_value=old):
            _,payload=bootstrap.inspect_source(p)
            for state in payload['optimizer_state']['state'].values():state['step'].zero_()
            with mock.patch.object(prefix_guard,'unpack',return_value=payload):
                with self.assertRaises(ValueError):bootstrap.inspect_source(p)
            (old/'encoder_work/boundary.ptx').write_bytes(b'changed')
            with self.assertRaises(ValueError):bootstrap.inspect_source(p)

    def test_new_prefix_snapshot_uses_registered_TRAIN512_archive(self):
        p,old,target,runtime=self.source();p['count']=512
        def archive(seed,count):
            self.assertEqual(count,512);return target
        with mock.patch.object(train.accepted,'archive_for',return_value=old),mock.patch.object(train,'archive_for',side_effect=archive):
            bootstrap.start(p,dict(binding=p['binding']),target/'encoder_work',runtime,torch.device('cpu'))
            self.assertTrue(prefix_guard.validate_saved(p)['encoder_projection_Adam_history_and_Torch_RNG_bitwise_equal'])

    def test_CPU_actual_start_gradient_check_restores_source_without_update(self):
        p,old,_,_=self.source()
        before=bootstrap.model_sha(p);state=copy.deepcopy(p['optimizer'].state_dict())
        with mock.patch.object(train.accepted,'archive_for',return_value=old):result=bootstrap.admit_cpu(p)
        self.assertEqual(bootstrap.model_sha(p),before);self.assertEqual(result['optimizer_updates'],0)
        self.assertGreater(result['projection_gradient_norm'],0.)
        self.assertTrue(prefix_guard.equal(state,p['optimizer'].state_dict()))


if __name__=='__main__':unittest.main()
