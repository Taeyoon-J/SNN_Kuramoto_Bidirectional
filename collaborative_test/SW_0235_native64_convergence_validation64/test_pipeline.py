import json,unittest,uuid
from unittest import mock
import numpy as np
from . import run,evaluate


class PipelineTests(unittest.TestCase):
    def test_actual64_score_entrypoint_and_all_three_GT_barrier(self):
        root=run.ROOT/'collaborative_testing/temporary'/('SW0235_score_'+uuid.uuid4().hex);root.mkdir(parents=True)
        mask=np.zeros((64,64,64),np.int64);mask[:,:32]=1;mask[:,32:]=2
        bundle={s:dict(labels=mask.copy(),baseline=np.ones_like(mask),control=np.ones_like(mask),protocol=dict(binding=dict(dataset={}))) for s in run.SEEDS};events=[]
        def valid():events.append('three64_frozen');return bundle
        def gt(_):self.assertEqual(events,['three64_frozen']);return np.tile(mask[:1],(320,1,1))
        with mock.patch.object(run,'ARCHIVE',root),mock.patch.object(evaluate,'validated_all',side_effect=valid),mock.patch.object(run.source.source.full,'read_gt',side_effect=gt):
            score=run.score();self.assertTrue(score['full320_candidate_gate']);self.assertTrue(score['balanced_candidate_gate']);self.assertEqual(score['image_count'],64);self.assertEqual(score['optimizer_updates'],0)
            self.assertAlmostEqual(score['scores']['joint_projection']['fg_ari'],1.)

    def test_missing_seed_blocks_all_GT(self):
        with mock.patch.object(evaluate,'validated_all',side_effect=FileNotFoundError),mock.patch.object(run.source.source.full,'read_gt') as gt:
            with self.assertRaises(FileNotFoundError):run.score()
            gt.assert_not_called()

    def test_candidate_tied_with_matched128_cannot_pass_FG_gate(self):
        root=run.ROOT/'collaborative_testing/temporary'/('SW0235_tie_'+uuid.uuid4().hex);root.mkdir(parents=True)
        mask=np.zeros((64,64,64),np.int64);mask[:,:32]=1;mask[:,32:]=2
        bundle={s:dict(labels=mask.copy(),control=mask.copy(),baseline=np.ones_like(mask),protocol=dict(binding=dict(dataset={}))) for s in run.SEEDS}
        with mock.patch.object(run,'ARCHIVE',root),mock.patch.object(evaluate,'validated_all',return_value=bundle),mock.patch.object(run.source.source.full,'read_gt',return_value=np.tile(mask[:1],(320,1,1))):
            score=run.score();self.assertFalse(score['full320_candidate_gate']);self.assertFalse(score['balanced_candidate_gate'])

    def test_raw64_cache_preserves_original16_and_requires_replay(self):
        root=run.ROOT/'collaborative_testing/temporary'/('SW0235_cache_'+uuid.uuid4().hex);root.mkdir(parents=True)
        masks=np.ones((64,64,64),np.int64);mini=masks[list(run.scope.MINI)].copy();rgb_hashes=['RGB'+str(i) for i in run.OFFSETS]
        parent=root/'fixed232';parent.mkdir();(parent/'boundary.ptx').write_bytes(b'fixed232 checkpoint fixture')
        binding=dict(fingerprint=run.fingerprint(),seed=0,candidate=dict(checkpoint_sha256=run.sha((parent/'boundary.ptx').read_bytes())),RGB_row_sha256=rgb_hashes,baseline_sha256=run.sha(masks.tobytes()))
        run.helpers.save_arrays(root/'baseline.npz',labels=masks);run.helpers.save_arrays(root/'mini.npz',labels=mini)
        admission=dict(binding=binding,baseline_file_sha256=run.sha((root/'baseline.npz').read_bytes()),mini_file_sha256=run.sha((root/'mini.npz').read_bytes()));run.write_json(root/'admission.json',admission)
        for offset in run.OFFSETS:
            folder=root/'rows'/f'{offset:04d}';run.helpers.save_arrays(folder/'prediction.npz',labels=masks[offset],frequency=np.zeros((4096,4),np.float32))
            run.write_json(folder/'row.json',dict(binding=binding,index=offset,image_id=1320+offset,inference_seed=193000+offset,RGB_sha256=rgb_hashes[offset],
                origin='frozen232_mini' if offset in run.scope.MINI else 'new_native_forward',data_sha256=run.sha((folder/'prediction.npz').read_bytes())))
        run.write_json(root/'replay.json',dict(binding=binding,mask_bitwise_equal=True,frequency_bitwise_equal=True))
        run.write_json(root/'protocol.json',dict(binding=binding,optimizer_updates=0,DEV_GT_read=False,image_ids=[1320+i for i in run.OFFSETS],total_steps=1024,settle_steps=512,reused_mini_images=16,new_prediction_images=48,labels_sha256=run.sha(masks.tobytes()),replay_file_sha256=run.sha((root/'replay.json').read_bytes())))
        with mock.patch.object(run,'archive_for',return_value=root),mock.patch.object(run.scope.model,'archive_for',return_value=parent.parent):
            # Production expects the fixed checkpoint under encoder_work.
            work=parent.parent/'encoder_work';work.mkdir();(work/'boundary.ptx').write_bytes((parent/'boundary.ptx').read_bytes())
            actual=run.validated_seed(0);np.testing.assert_array_equal(actual['labels'],masks)
            bad=json.loads((root/'replay.json').read_bytes());bad['mask_bitwise_equal']=False;(root/'replay.json').write_bytes(json.dumps(bad).encode())
            with self.assertRaises(ValueError):run.validated_seed(0)

    def test_exclusive_queue_uses_fixed_scope_API(self):
        from . import queue
        self.assertIs(queue.api,run);self.assertIn('SW_0235_native64_convergence_validation64',queue.MODULE);self.assertEqual(queue.worker_command(2,3)[-1],'3')
        with self.assertRaises(ValueError):queue.worker_command(2,4)


if __name__=='__main__':unittest.main()
