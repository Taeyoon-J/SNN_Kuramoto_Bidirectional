import json,unittest,uuid
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from . import cache,training,run

torch.set_num_threads(1)

def temp():
    p=cache.ROOT/'collaborative_testing/temporary'/('sw0211_'+uuid.uuid4().hex);p.mkdir(parents=True,exist_ok=False);return p

def fixture():
    x=np.random.RandomState(211).normal(size=(4,8,8,8)).astype(np.float32);y=np.zeros((4,8,8),np.int64);y[:,1:3,1:3]=1;y[:,5:7,5:7]=2
    return x,y,dict(fixture='TRAIN_only_no_DEV_labels')

class PipelineTests(unittest.TestCase):
    def test_actual_Adam_model_resume_matches_uninterrupted_training(self):
        root=temp();x,y,protocol=fixture()
        a=training.fit(x,y,protocol,device='cpu',steps=3,snapshots=(1,3),work=root/'a_work',output=root/'a')
        self.assertIsNone(training.fit(x,y,protocol,device='cpu',steps=3,snapshots=(1,3),work=root/'b_work',output=root/'b',stop_after=1))
        b=training.fit(x,y,protocol,device='cpu',steps=3,snapshots=(1,3),work=root/'b_work',output=root/'b')
        self.assertEqual(a['actual_Adam_steps'],[3]);self.assertEqual(a['final_model_sha256'],b['final_model_sha256'])
        self.assertEqual([v['loss'] for v in a['history']],[v['loss'] for v in b['history']])
        self.assertEqual(a['core_optimizer_steps'],0)

    def test_changed_TRAIN_data_cannot_resume_existing_head(self):
        root=temp();x,y,protocol=fixture()
        training.fit(x,y,protocol,device='cpu',steps=2,snapshots=(2,),work=root/'work',output=root/'out',stop_after=1)
        with self.assertRaises(ValueError):training.fit(x+.1,y,protocol,device='cpu',steps=2,snapshots=(2,),work=root/'work',output=root/'out')

    def test_fixed_budgets_and_no_DEV_input_to_fitter(self):
        order=training.batch_order(32,128);self.assertEqual(len(order),128)
        for start in range(0,128,8):self.assertEqual(sorted(sum(order[start:start+8],[])),list(range(32)))
        import inspect
        self.assertNotIn('DEV_labels',inspect.signature(training.fit).parameters)
        self.assertEqual(cache.SNAPSHOTS,(32,64,128))

    def test_frozen_input_ablations_do_not_mutate_features(self):
        head=run.CenterHead();head.mean.fill_(.25);x=torch.rand((1,268,8,8));original=x.clone()
        values=run.head_input(x,head,'moments_only')
        self.assertTrue(torch.equal(values[:,:256],torch.full_like(values[:,:256],.25)))
        self.assertTrue(torch.equal(values[:,256:],x[:,256:]));self.assertTrue(torch.equal(x,original))
        self.assertTrue(torch.equal(run.head_input(x,head,'mean_input'),torch.full_like(x,.25)))

    def test_GT_reader_is_not_called_before_all_predictions_validate(self):
        with patch.object(run,'validated_predictions',side_effect=ValueError('missing predictions')),patch.object(cache.source.full.metrics,'read_ground_truth') as reader:
            with self.assertRaises(ValueError):run.score()
            reader.assert_not_called()

    def test_row_identity_keeps_TRAIN_and_DEV_seeds_and_IDs_distinct(self):
        binding=dict(TRAIN_ids=list(range(32)),DEV_ids=list(range(1320,1352,2)),TRAIN_RGB_rows_sha256=['a']*32,DEV_RGB_rows_sha256=['b']*16)
        self.assertEqual(cache.row_binding(binding,'TRAIN',2)['inference_seed'],211002)
        self.assertEqual(cache.row_binding(binding,'DEV',2)['inference_seed'],193004)
        self.assertEqual(cache.row_binding(binding,'DEV',2)['image_id'],1324)
        with self.assertRaises(ValueError):cache.row_binding(binding,'DEV',16)

    def test_queue_requires_assigned_physical_gpu(self):
        from .queue import worker_command
        self.assertEqual(worker_command(3)[-2:],['--physical-gpu','3'])
        with self.assertRaises(ValueError):worker_command(True)

if __name__=='__main__':unittest.main()
