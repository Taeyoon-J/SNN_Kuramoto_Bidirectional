import copy,json,unittest,uuid
from contextlib import contextmanager
from pathlib import Path
from unittest import mock
import numpy as np
import torch
from . import train,evaluate,projection
from collaborative_test.SW_0222_native64_frequency_teacher_denoising.teacher_loss import make_teacher
from collaborative_test.SW_0221_native64_frequency_pretraining.test_pipeline import Found


@contextmanager
def fixture():
    path=train.ROOT/'collaborative_testing/temporary'/('SW0226_'+uuid.uuid4().hex);path.mkdir(parents=True);yield path


def prepared():
    torch.manual_seed(222);found=Found();optimizer=train.optimizer_for(found)
    rgb=np.random.RandomState(1).randint(0,256,(4,2,2,3)).astype(np.uint8);labels=np.tile(np.array([[[0,0],[1,1]]]),(4,1,1))
    with torch.no_grad():gamma=found.encode(torch.from_numpy(rgb));frequency=train.effective_frequency(found.wrapped.core,gamma)
    targets=make_teacher(gamma.numpy(),frequency.numpy(),labels)
    binding=dict(source_core_sha256=train.state_sha(found.wrapped.state_dict()),frozen_nonprojection_core_sha256=projection.frozen_core_sha(found),initial_projection_sha256=train.state_sha(projection.module(found).state_dict()),input_statistics_sha256=train.statistics_sha(found),teacher_arrays_sha256=train.state_sha(targets))
    return dict(found=found,optimizer=optimizer,seed=0,count=4,binding=binding,rgb=rgb,labels=labels,targets=targets,dev=rgb[:2])


class PipelineTests(unittest.TestCase):
    def test_registered_protocol_and_same_221_TRAIN_order(self):
        for seed in train.SEEDS:
            for step in range(train.STEPS):self.assertEqual(train.batch_indices(seed,512,step),train.parent.batch_indices(seed,512,step))
        with self.assertRaises(ValueError):train.require_job(0,64)

    def test_fit_teacher_checkpoint_resume_and_frozen_core(self):
        p=prepared();teacher_sha=train.state_sha(p['targets']);runtime=train.checkpoint.runtime_signature('cpu')
        with fixture() as a,mock.patch.object(train,'archive_for',return_value=a),mock.patch.object(train,'STEPS',2),mock.patch.object(train,'BATCH',2):
            report=train.fit(p,'cpu',runtime);first=train.state_sha(p['found'].encoder.state_dict())
            self.assertTrue(report['encoder_changed']);self.assertEqual(report['other_core_updates'],0);self.assertEqual(report['core_projection_updates'],2);self.assertTrue(report['projection_changed']);self.assertEqual(report['actual_Adam_steps'],[2])
            self.assertEqual(teacher_sha,train.state_sha(p['targets']));self.assertTrue(all(v.grad is None for k,v in p['found'].wrapped.named_parameters() if k!=projection.KEY))
            self.assertEqual(report,train.fit(p,'cpu',runtime));self.assertEqual(first,train.state_sha(p['found'].encoder.state_dict()))
            self.assertGreaterEqual(report['final_TRAIN_phase_loss']['gamma_anchor'],0.)
            (a/'teacher.npz').write_bytes(b'corrupt')
            with self.assertRaises(Exception):train.fit(p,'cpu',runtime)

    def test_interrupted_boundary_resumes_same_encoder_and_optimizer(self):
        runtime=train.checkpoint.runtime_signature('cpu')
        with fixture() as control,fixture() as interrupted,mock.patch.object(train,'STEPS',2),mock.patch.object(train,'BATCH',2):
            p=prepared()
            with mock.patch.object(train,'archive_for',return_value=control):expected=train.fit(p,'cpu',runtime)
            expected_projection=copy.deepcopy(projection.module(p['found']).state_dict());expected_encoder=copy.deepcopy(p['found'].encoder.state_dict());expected_optimizer=train.state_sha({str(i)+k:v for i,row in p['optimizer'].state_dict()['state'].items() for k,v in row.items()})
            save=train.checkpoint.save_boundary
            def fail_after_one(*args,**kwargs):
                result=save(*args,**kwargs)
                if kwargs['completed_updates']==1:raise InterruptedError('simulated foreign-owner yield after durable update')
                return result
            p=prepared()
            with mock.patch.object(train,'archive_for',return_value=interrupted),mock.patch.object(train.checkpoint,'save_boundary',side_effect=fail_after_one):
                with self.assertRaises(InterruptedError):train.fit(p,'cpu',runtime)
            p=prepared()
            with mock.patch.object(train,'archive_for',return_value=interrupted):resumed=train.fit(p,'cpu',runtime)
            self.assertEqual(expected['history'],resumed['history'])
            for name,value in p['found'].encoder.state_dict().items():torch.testing.assert_close(value,expected_encoder[name],rtol=0,atol=0)
            actual=train.state_sha({str(i)+k:v for i,row in p['optimizer'].state_dict()['state'].items() for k,v in row.items()})
            self.assertEqual(expected_optimizer,actual)
            torch.testing.assert_close(projection.module(p['found']).weight,expected_projection['weight'],rtol=0,atol=0)

    def test_full_scoring_branch_with_real_metric_implementation(self):
        grid=np.zeros((16,64,64),np.int64);grid[:,:32]=1;grid[:,32:]=2
        report=dict(binding=dict(parent=dict(dataset={})),initial_TRAIN_phase_loss={'total':1.},final_TRAIN_phase_loss={'total':.5},projection_changed=True,history=[dict(projection_gradient_norm=.01)])
        bundle={(s,512):dict(labels=grid.copy(),baseline=grid.copy(),control=grid.copy(),report=report) for s in train.SEEDS};events=[]
        def validated():events.append('three_final_protocols');return bundle
        def gt(_):self.assertEqual(events,['three_final_protocols']);return np.tile(grid[:1],(320,1,1))
        with fixture() as a,mock.patch.object(train,'ARCHIVE',a),mock.patch.object(evaluate,'validated_all',side_effect=validated),mock.patch.object(train.source.source.full,'validated_all',return_value=({},{})),mock.patch.object(train.source.source.full,'read_gt',side_effect=gt):
            summary=evaluate.score();self.assertFalse(summary['full320_candidate_gate']);self.assertAlmostEqual(summary['scores']['train512'][evaluate.METRICS[0]],1.)

    def test_missing_candidate_blocks_DEV_GT(self):
        with mock.patch.object(evaluate,'validated_all',side_effect=FileNotFoundError),mock.patch.object(train.source.source.full,'read_gt') as gt:
            with self.assertRaises(FileNotFoundError):evaluate.score()
            gt.assert_not_called()

    def test_worker_wiring_and_shared_foreign_owner_guard(self):
        from . import queue
        self.assertIs(queue.api,train);self.assertIn('SW_0226_native64_joint_projection_TRAIN512',queue.MODULE)
        command=queue.worker_command(2,512,3);self.assertEqual(command[-1],'3')
        with self.assertRaises(ValueError):queue.worker_command(2,64,3)
        owner=queue.load('collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4')
        self.assertTrue(callable(owner.gpu_guard._foreign_owners));self.assertTrue(callable(train.source.source.source.model64.move_to_device))

    def test_same2048_exposures_cover_each_of512_images_four_times(self):
        for seed in train.SEEDS:
            indices=[i for step in range(train.STEPS) for i in train.batch_indices(seed,512,step)]
            self.assertEqual(len(indices),2048)
            np.testing.assert_array_equal(np.bincount(indices,minlength=512),np.full(512,4))


if __name__=='__main__':unittest.main()
