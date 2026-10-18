import json,unittest,uuid
from pathlib import Path
from unittest import mock
import numpy as np
from . import run,evaluate


class PipelineTests(unittest.TestCase):
    def test_real_score_entrypoint_and_all_three_GT_barrier(self):
        root=run.ROOT/'collaborative_testing/temporary'/('SW0224_score_'+uuid.uuid4().hex);root.mkdir(parents=True)
        mask=np.zeros((16,64,64),np.int64);mask[:,:32]=1;mask[:,32:]=2
        bundle={s:dict(labels=mask.copy(),baseline=mask.copy(),protocol=dict(binding=dict(parent=dict(dataset={})))) for s in run.SEEDS};events=[]
        def validate():events.append('all_three');return bundle
        def gt(_):self.assertEqual(events,['all_three']);return np.tile(mask[:1],(320,1,1))
        with mock.patch.object(run,'ARCHIVE',root),mock.patch.object(evaluate,'validated_all',side_effect=validate),mock.patch.object(run.source.source.full,'read_gt',side_effect=gt):
            result=run.score();self.assertEqual(result['optimizer_updates'],0);self.assertAlmostEqual(result['scores']['aligned']['fg_ari'],1.);self.assertFalse(result['full320_candidate_gate'])

    def test_missing_seed_prevents_GT(self):
        with mock.patch.object(evaluate,'validated_all',side_effect=FileNotFoundError),mock.patch.object(run.source.source.full,'read_gt') as gt:
            with self.assertRaises(FileNotFoundError):run.score()
            gt.assert_not_called()

    def test_raw_mask_adapter_and_source_identity_validation(self):
        root=run.ROOT/'collaborative_testing/temporary'/('SW0224_masks_'+uuid.uuid4().hex);root.mkdir(parents=True)
        baseline=np.ones((16,64,64),np.int64);binding=dict(seed=0,fingerprint=run.fingerprint(),source_checkpoint_sha256='fixture_checkpoint',source_parameter_sha256=run.source.MODELS['0'],baseline_sha256=run.sha(baseline.tobytes()))
        run.helpers.save_arrays(root/'baseline.npz',labels=baseline)
        run.write_json(root/'admission.json',dict(binding=binding,baseline_file_sha256=run.sha((root/'baseline.npz').read_bytes())))
        for i,offset in enumerate(run.OFFSETS):
            folder=root/'rows'/f'{i:04d}';run.helpers.save_arrays(folder/'prediction.npz',labels=baseline[i],frequency=np.zeros((4096,4),np.float32))
            run.write_json(folder/'row.json',dict(binding=binding,index=i,image_id=1320+offset,inference_seed=193000+offset,data_sha256=run.sha((folder/'prediction.npz').read_bytes())))
        protocol=dict(binding=binding,labels_sha256=run.sha(baseline.tobytes()),image_ids=[1320+i for i in run.OFFSETS],DEV_GT_read=False,total_steps=1024,settle_steps=512,
            parameter_sha256_before_after=run.source.MODELS['0'],optimizer_updates=0,adapter_code_sha256=run.sha((run.HERE/'alignment.py').read_bytes()))
        run.write_json(root/'protocol.json',protocol)
        with mock.patch.object(run,'archive_for',return_value=root),mock.patch.object(run,'checkpoint_sha',return_value='fixture_checkpoint'):
            actual=run.validated_seed(0);np.testing.assert_array_equal(actual['labels'],baseline)
            protocol['adapter_code_sha256']='changed adapter';(root/'protocol.json').write_bytes(json.dumps(protocol).encode())
            with self.assertRaises(ValueError):run.validated_seed(0)

    def test_exclusive_queue_wiring(self):
        from . import queue
        self.assertIs(queue.api,run);self.assertIn('SW_0224_native64_aligned_feature_cells',queue.MODULE)
        self.assertEqual(queue.worker_command(2,3)[-1],'3')
        with self.assertRaises(ValueError):queue.worker_command(2,4)
        owner=queue.load('collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4');self.assertTrue(callable(owner.gpu_guard._foreign_owners))


if __name__=='__main__':unittest.main()
