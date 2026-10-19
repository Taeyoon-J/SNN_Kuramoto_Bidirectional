import unittest,json,uuid
from unittest import mock
import numpy as np
import torch
from . import train,projection
from collaborative_test.SW_0221_native64_frequency_pretraining.test_pipeline import Found
from collaborative_test.SW_0222_native64_frequency_teacher_denoising.teacher_loss import make_teacher,teacher_objective


class ProjectionTests(unittest.TestCase):
    def test_real_control_reader_freezes_teacher_checkpoint_and_sixteen_masks(self):
        root=train.ROOT/'collaborative_testing/temporary'/('SW0232_control_'+uuid.uuid4().hex);folder=root/'seed0';folder.mkdir(parents=True)
        control=projection.control;mask=np.zeros((64,64),np.int64);mask[:32]=1;labels=np.stack([mask]*16)
        binding={k:'frozen_fixture' for k in ('TRAIN_RGB_sha256','TRAIN_labels_sha256','DEV_RGB_sha256','DEV_baseline_sha256','teacher_arrays_sha256','parent_model_sha256')};binding['TRAIN_ids']=list(range(64))
        report=dict(binding=binding,final_model_sha256='fixture_model');control.write_json(folder/'training_report.json',report)
        expected=dict(training_report_sha256=control.sha((folder/'training_report.json').read_bytes()),binding=binding,model_sha256=report['final_model_sha256'])
        protocol=dict(binding=expected,DEV_GT_read=False,labels_sha256=control.sha(labels.tobytes()));control.write_json(folder/'protocol.json',protocol)
        for i in range(16):
            row=folder/'rows'/f'{i:04d}';control.save_arrays(row/'prediction.npz',labels=mask,frequency=np.zeros((4096,4),np.float32))
            control.write_json(row/'row.json',dict(binding=expected,index=i,data_sha256=control.sha((row/'prediction.npz').read_bytes())))
        work=folder/'encoder_work';work.mkdir();(work/'boundary.ptx').write_bytes(b'accepted-scorer-checkpoint-fixture')
        control.write_json(root/'score_v1/summary.json',dict(fingerprint=control.fingerprint()))
        control.write_json(root/'completion_audit_v1/summary.json',dict(score_sha256=control.sha((root/'score_v1/summary.json').read_bytes()),
            checkpoints={'0':dict(protocol_sha256=control.sha((folder/'protocol.json').read_bytes()),checkpoint_sha256=control.sha((work/'boundary.ptx').read_bytes()))}))
        report['checkpoint_file_sha256']=control.sha((work/'boundary.ptx').read_bytes())
        # Rebind all immutable metadata after including the checkpoint identity.
        (folder/'training_report.json').write_bytes((json.dumps(report)+'\n').encode());expected['training_report_sha256']=control.sha((folder/'training_report.json').read_bytes())
        (folder/'protocol.json').write_bytes((json.dumps(protocol)+'\n').encode())
        for i in range(16):
            path=folder/'rows'/f'{i:04d}'/'row.json';row=json.loads(path.read_bytes());row['binding']=expected;path.write_bytes((json.dumps(row)+'\n').encode())
        path=root/'completion_audit_v1/summary.json';audit=json.loads(path.read_bytes());audit['checkpoints']['0']['protocol_sha256']=control.sha((folder/'protocol.json').read_bytes());path.write_bytes((json.dumps(audit)+'\n').encode())
        with mock.patch.object(control,'ARCHIVE',root),mock.patch.object(control,'archive_for',return_value=folder):
            actual,ref=projection.control_reference(0,binding);np.testing.assert_array_equal(actual,labels)
            self.assertEqual(ref['labels_sha256'],control.sha(labels.tobytes()))
            with self.assertRaises(ValueError):projection.control_reference(0,{**binding,'teacher_arrays_sha256':'different'})
            (work/'boundary.ptx').write_bytes(b'changed checkpoint')
            with self.assertRaises(ValueError):projection.control_reference(0,binding)

    def test_only_declared_core_projection_learns_and_other_state_is_guarded(self):
        torch.manual_seed(223);found=Found();before=projection.frozen_core_sha(found);optimizer=train.optimizer_for(found)
        self.assertEqual({k for k,v in found.wrapped.named_parameters() if v.requires_grad},{projection.KEY})
        self.assertEqual([g['name'] for g in optimizer.param_groups],['source_base','context_encoder','sensory_projection'])
        with torch.no_grad():projection.module(found).weight.add_(.01)
        self.assertEqual(before,projection.frozen_core_sha(found))
        with torch.no_grad():found.wrapped.core.kuramoto.omega.add_(.01)
        self.assertNotEqual(before,projection.frozen_core_sha(found))
        with self.assertRaises(ValueError):projection.enable(found,optimizer)

    def test_actual_projection_gradient_on_same_teacher_and_frozen_phase_band(self):
        torch.manual_seed(223);found=Found();train.optimizer_for(found)
        rgb=torch.from_numpy(np.random.RandomState(223).randint(0,256,(2,2,2,3)).astype(np.uint8));labels=np.array([[[0,0],[1,1]]]*2)
        with torch.no_grad():gamma=found.encode(rgb);frequency=train.effective_frequency(found.wrapped.core,gamma)
        target=make_teacher(gamma.numpy(),frequency.numpy(),labels)
        gamma=found.encode(rgb);loss,_=teacher_objective(gamma,train.effective_frequency(found.wrapped.core,gamma),target);loss.backward()
        gradient=projection.module(found).weight.grad;self.assertTrue(torch.isfinite(gradient).all());self.assertGreater(float(gradient.abs().sum()),0.)
        self.assertIsNone(found.wrapped.core.kuramoto.omega.grad);self.assertIsNone(found.wrapped.core.kuramoto.freq_gain.grad)


if __name__=='__main__':unittest.main()
