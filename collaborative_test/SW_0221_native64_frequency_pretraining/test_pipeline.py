import json,unittest,uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import numpy as np
import torch
from . import train,evaluate
from snn_kuramoto_bidirectional.gamma_initializer import GammaToDrive


@contextmanager
def fixture():
    path=train.ROOT/'collaborative_testing/temporary'/('SW0221_'+uuid.uuid4().hex)
    path.mkdir(parents=True)
    yield path


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__();self.base_encoder=torch.nn.Conv2d(3,3,1);self.context=torch.nn.Conv2d(3,3,1)
    def forward(self,x):return self.base_encoder(x)+self.context(x)


class Found:
    def __init__(self):
        self.encoder=Encoder();self.wrapped=torch.nn.Module();self.wrapped.core=torch.nn.Module();c=self.wrapped.core
        c.gamma_channel_proj=torch.nn.Linear(3,2,bias=False);c.gamma_phase_gain=torch.nn.Parameter(torch.tensor(1.))
        c.gamma_to_drive=GammaToDrive(3,4,2,'static','standardize_tanh');c.kuramoto=torch.nn.Module()
        c.kuramoto.freq_gain=torch.nn.Parameter(torch.tensor(2.));c.kuramoto.omega=torch.nn.Parameter(torch.zeros(4,2))
        self.feature_mean=np.zeros((1,3,1,1),np.float32);self.feature_std=np.ones((1,3,1,1),np.float32);self.feature_clip=4.
    def encode(self,rgb):return self.encoder(rgb.permute(0,3,1,2).float()/255.).flatten(2)


class PipelineTests(unittest.TestCase):
    def test_equal_exposure_nested_pool(self):
        for count in train.COUNTS:
            indices=np.array([train.batch_indices(0,count,s) for s in range(train.STEPS)]).reshape(-1)
            self.assertEqual(len(indices),2048);self.assertTrue(np.all(np.bincount(indices,minlength=count)==2048//count))
        self.assertNotEqual(train.batch_indices(0,64,0),train.batch_indices(1,64,0))

    def test_actual_fit_checkpoint_resume_and_frozen_core(self):
        torch.manual_seed(221);found=Found();optimizer=train.optimizer_for(found)
        with fixture() as tmp,mock.patch.object(train,'archive_for',return_value=Path(tmp)),mock.patch.object(train,'STEPS',2),mock.patch.object(train,'BATCH',2):
            rgb=np.random.RandomState(1).randint(0,256,(4,2,2,3)).astype(np.uint8);labels=np.tile(np.array([[[0,0],[1,1]]]),(4,1,1))
            binding=dict(frozen_core_sha256=train.state_sha(found.wrapped.state_dict()),input_statistics_sha256=train.statistics_sha(found))
            p=dict(found=found,optimizer=optimizer,seed=0,count=4,binding=binding,rgb=rgb,labels=labels,dev=rgb[:2])
            runtime=train.checkpoint.runtime_signature('cpu');report=train.fit(p,'cpu',runtime)
            self.assertEqual(report['actual_Adam_steps'],[2]);self.assertTrue(report['encoder_changed']);self.assertEqual(report['core_updates'],0)
            first=train.state_sha(found.encoder.state_dict());again=train.fit(p,'cpu',runtime)
            self.assertEqual(report,again);self.assertEqual(first,train.state_sha(found.encoder.state_dict()))
            self.assertTrue(all(p.grad is None for p in found.wrapped.parameters()))
            # Corruption is rejected rather than silently refitting.
            (Path(tmp)/'training_report.json').write_text(json.dumps(dict(report,final_encoder_sha256='corrupt')))
            with self.assertRaises(ValueError):train.fit(p,'cpu',runtime)

    def test_array_resume_and_corruption_guard(self):
        with fixture() as tmp:
            path=Path(tmp)/'data.npz';train.save_arrays(path,x=np.arange(4));train.save_arrays(path,x=np.arange(4))
            with self.assertRaises(ValueError):train.save_arrays(path,x=np.arange(3))

    def test_score_all_arms_barrier_and_real_metrics(self):
        # Run the entire real scoring branch, not a mocked score return value.
        grid=np.zeros((16,64,64),np.int64);grid[:,:32]=1;grid[:,32:]=2
        report=dict(initial_TRAIN_phase_loss={'total':1.},final_TRAIN_phase_loss={'total':.5})
        bundle={(s,n):dict(labels=grid.copy(),baseline=grid.copy(),report=report) for s in train.SEEDS for n in train.COUNTS}
        events=[]
        def validated():events.append('all_six_validated');return bundle
        def gt(_):self.assertEqual(events,['all_six_validated']);return np.tile(grid[:1],(320,1,1))
        with fixture() as tmp,mock.patch.object(train,'ARCHIVE',Path(tmp)),mock.patch.object(evaluate,'validated_all',side_effect=validated),mock.patch.object(train.source.source.full,'validated_all',return_value=({},{})),mock.patch.object(train.source.source.full,'read_gt',side_effect=gt):
            summary=evaluate.score();self.assertFalse(any(summary['full320_candidate_gates'].values()))
            self.assertAlmostEqual(summary['scores']['TRAIN512'][evaluate.METRICS[0]],1.)
            self.assertTrue((Path(tmp)/'score_v1/summary.json').is_file())

    def test_missing_candidate_prevents_GT(self):
        with mock.patch.object(evaluate,'validated_all',side_effect=FileNotFoundError),mock.patch.object(train.source.source.full,'read_gt') as gt:
            with self.assertRaises(FileNotFoundError):evaluate.score()
            gt.assert_not_called()

    def test_registered_worker_cli_and_foreign_owner_API(self):
        from . import queue
        command=queue.worker_command(2,512,3);self.assertIn('--worker',command);self.assertEqual(command[-1],'3')
        with self.assertRaises(ValueError):queue.worker_command(2,512,4)
        owner=queue.load('collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4')
        self.assertTrue(callable(owner.gpu_guard._foreign_owners));self.assertTrue(callable(owner.gpu_guard._process_tree))
        self.assertTrue(callable(train.source.source.source.model64.move_to_device))


if __name__=='__main__':unittest.main()
