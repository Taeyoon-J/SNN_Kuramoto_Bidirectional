import json,unittest,uuid
from unittest import mock
import numpy as np
from . import run,evaluate


class PipelineTests(unittest.TestCase):
    def test_full320_scoring_uses_real_Slot_import_and_all_protocols_before_GT(self):
        root=run.ROOT/'collaborative_testing/temporary'/('SW0237_score_'+uuid.uuid4().hex);root.mkdir(parents=True)
        mask=np.zeros((320,64,64),np.int64);mask[:,32:,:32]=1;mask[:,32:,32:]=2
        baseline=np.ones_like(mask)
        bundle={s:dict(labels=mask.copy(),baseline=baseline.copy(),protocol=dict(binding=dict(dataset={},RGB_sha256='same'))) for s in run.SEEDS}
        slots={s:dict(ids=np.arange(1320,1640),labels64=baseline.copy(),prediction_sha256='fixture',protocol_sha256='fixture') for s in run.SEEDS}
        value=run.source.source.full.metrics.independent.image_metrics(baseline[0],mask[0])
        expected={name:dict(zip(evaluate.METRICS,value)) for name in ('our_three_seed_mean','slot_three_seed_mean')}
        events=[]
        def valid():events.append('three320_frozen');return bundle
        def slot(_):self.assertEqual(events,['three320_frozen']);events.append('three_slots');return slots
        def gt(_):self.assertEqual(events,['three320_frozen','three_slots']);return mask
        with mock.patch.object(run,'ARCHIVE',root),mock.patch.object(evaluate,'validated_all',side_effect=valid),mock.patch.object(run.source.source.full.metrics.slot,'prepare_all_slot_predictions',side_effect=slot),mock.patch.object(evaluate,'reference_scores',return_value=expected),mock.patch.object(evaluate,'current_incumbent_reference',return_value=dict(experiment='fixture',scores=expected['our_three_seed_mean'])),mock.patch.object(run.source.source.full,'read_gt',side_effect=gt):
            score=run.score();self.assertTrue(score['three_seed_strictly_exceeds_Slot_means']);self.assertTrue(score['native64_FG090_three_seed_goal_verified']);self.assertEqual(score['image_count'],320);self.assertEqual(score['optimizer_updates'],0)
            self.assertAlmostEqual(score['scores']['joint_projection']['fg_ari'],1.)

    def test_changed_benchmark_cannot_silently_become_a_comparable_result(self):
        groups={name:np.zeros((3,320,3)) for name in ('baseline','slot')}
        accepted={name:dict.fromkeys(evaluate.METRICS,0.) for name in ('our_three_seed_mean','slot_three_seed_mean')}
        self.assertEqual(evaluate.verify_reference_metrics(groups,accepted),0.)
        accepted['slot_three_seed_mean']['fg_ari']=.1
        with self.assertRaises(ValueError):evaluate.verify_reference_metrics(groups,accepted)

    def test_missing_Slot_blocks_GT_even_when_our_masks_exist(self):
        bundle={s:dict(protocol=dict(binding=dict(dataset={},RGB_sha256='same'))) for s in run.SEEDS}
        with mock.patch.object(evaluate,'validated_all',return_value=bundle),mock.patch.object(run.source.source.full.metrics.slot,'prepare_all_slot_predictions',side_effect=FileNotFoundError),mock.patch.object(run.source.source.full,'read_gt') as gt:
            with self.assertRaises(FileNotFoundError):run.score()
            gt.assert_not_called()

    def test_missing_seed_blocks_all_GT(self):
        with mock.patch.object(evaluate,'validated_all',side_effect=FileNotFoundError),mock.patch.object(run.source.source.full,'read_gt') as gt:
            with self.assertRaises(FileNotFoundError):run.score()
            gt.assert_not_called()

    def test_missing_incumbent_evidence_also_blocks_GT(self):
        bundle={s:dict(protocol=dict(binding=dict(dataset={},RGB_sha256='same'))) for s in run.SEEDS}
        with mock.patch.object(evaluate,'validated_all',return_value=bundle),mock.patch.object(evaluate,'load_slot_masks',return_value={}),mock.patch.object(evaluate,'reference_scores',return_value={}),mock.patch.object(evaluate,'current_incumbent_reference',side_effect=FileNotFoundError),mock.patch.object(run.source.source.full,'read_gt') as gt:
            with self.assertRaises(FileNotFoundError):run.score()
            gt.assert_not_called()

    def test_raw64_cache_preserves_original16_and_requires_replay(self):
        root=run.ROOT/'collaborative_testing/temporary'/('SW0237_cache_'+uuid.uuid4().hex);root.mkdir(parents=True)
        masks=np.ones((320,64,64),np.int64);mini=masks[list(run.scope.MINI)].copy();rgb_hashes=['RGB'+str(i) for i in run.OFFSETS]
        parent=root/'fixed232';parent.mkdir();(parent/'boundary.ptx').write_bytes(b'fixed232 checkpoint fixture')
        binding=dict(fingerprint=run.fingerprint(),seed=0,candidate=dict(checkpoint_sha256=run.sha((parent/'boundary.ptx').read_bytes()),validation64_labels_sha256=run.sha(mini.tobytes())),RGB_row_sha256=rgb_hashes,baseline_sha256=run.sha(masks.tobytes()))
        run.helpers.save_arrays(root/'baseline.npz',labels=masks);run.helpers.save_arrays(root/'mini.npz',labels=mini)
        admission=dict(binding=binding,baseline_file_sha256=run.sha((root/'baseline.npz').read_bytes()),mini_file_sha256=run.sha((root/'mini.npz').read_bytes()));run.write_json(root/'admission.json',admission)
        for offset in run.OFFSETS:
            folder=root/'rows'/f'{offset:04d}';run.helpers.save_arrays(folder/'prediction.npz',labels=masks[offset],frequency=np.zeros((4096,4),np.float32))
            run.write_json(folder/'row.json',dict(binding=binding,index=offset,image_id=1320+offset,inference_seed=193000+offset,RGB_sha256=rgb_hashes[offset],
                origin='frozen235_validation64' if offset in run.scope.MINI else 'new_native_forward',data_sha256=run.sha((folder/'prediction.npz').read_bytes())))
        run.write_json(root/'replay.json',dict(binding=binding,mask_bitwise_equal=True,frequency_bitwise_equal=True))
        run.write_json(root/'protocol.json',dict(binding=binding,optimizer_updates=0,DEV_GT_read=False,image_ids=[1320+i for i in run.OFFSETS],total_steps=1024,settle_steps=512,reused_reference_images=64,new_prediction_images=256,labels_sha256=run.sha(masks.tobytes()),replay_file_sha256=run.sha((root/'replay.json').read_bytes())))
        with mock.patch.object(run,'archive_for',return_value=root),mock.patch.object(run.scope.model,'archive_for',return_value=parent.parent):
            # Production expects the fixed checkpoint under encoder_work.
            work=parent.parent/'encoder_work';work.mkdir();(work/'boundary.ptx').write_bytes((parent/'boundary.ptx').read_bytes())
            actual=run.validated_seed(0);np.testing.assert_array_equal(actual['labels'],masks)
            bad=json.loads((root/'replay.json').read_bytes());bad['mask_bitwise_equal']=False;(root/'replay.json').write_bytes(json.dumps(bad).encode())
            with self.assertRaises(ValueError):run.validated_seed(0)

    def test_exclusive_queue_uses_fixed_scope_API(self):
        from . import queue
        self.assertIs(queue.api,run);self.assertIn('SW_0237_native64_converged_full320',queue.MODULE);self.assertEqual(queue.worker_command(2,3)[-1],'3')
        with self.assertRaises(ValueError):queue.worker_command(2,4)


if __name__=='__main__':unittest.main()
