import unittest
from unittest.mock import patch
import numpy as np
from . import run as api

class ReferenceTests(unittest.TestCase):
    def test_independent_seeds_required(self):
        for bad in (0,3,True,'1',None):
            with self.assertRaises(ValueError):api.require_seed(bad)
        self.assertEqual(api.require_seed(2),2)
        self.assertEqual(len(set(api.MODELS.values())|{api.full.MODELS['warm192']}),3)

    def test_row_identity_binds_model_seed_even_with_shared_inference_noise(self):
        binding=dict(seed=1,RGB_row_sha256=['rgb']*320)
        one=api.row_binding(binding,17);binding['seed']=2;two=api.row_binding(binding,17)
        self.assertEqual(one['inference_seed'],two['inference_seed'])
        self.assertNotEqual(one['source_bindings_sha256'],two['source_bindings_sha256'])
        self.assertNotEqual(one['model_sha256'],two['model_sha256'])

    def test_source_global_configuration_does_not_change_shared_fingerprint(self):
        original=(api.source.SEED,api.source.ARCHIVE)
        try:
            api.source.configure(1);one=api.fingerprint();api.source.configure(2);two=api.fingerprint()
            self.assertEqual(one,two)
        finally:api.source.SEED,api.source.ARCHIVE=original

    def test_mini_reference_uses_original_even_offset_mapping(self):
        mini=np.stack([np.full((64,64),i,dtype=np.int64) for i in range(16)])
        api.check_mini(14,mini[7],mini)
        with self.assertRaises(ValueError):api.check_mini(14,mini[6],mini)
        api.check_mini(33,mini[0],mini)

    def test_high_seed0_does_not_establish_mean_win_or_090(self):
        vectors={f'seed{s}':np.full((320,3),(.95 if s==0 else .70)) for s in range(3)}
        vectors.update({f'slot_seed{s}':np.full((320,3),.80) for s in range(3)})
        r=api.summarize(vectors)
        self.assertFalse(r['three_seed_strictly_exceeds_Slot_means']);self.assertFalse(r['native64_FG090_three_seed_goal_verified'])

    def test_all_three_metrics_and_full_scope_required(self):
        vectors={f'seed{s}':np.full((320,3),.91) for s in range(3)}
        vectors.update({f'slot_seed{s}':np.full((320,3),.80) for s in range(3)})
        self.assertTrue(api.summarize(vectors)['native64_FG090_three_seed_goal_verified'])
        vectors['seed1'][:,2]=0
        self.assertFalse(api.summarize(vectors)['three_seed_strictly_exceeds_Slot_means'])
        vectors['seed1']=np.ones((16,3))
        with self.assertRaises(ValueError):api.summarize(vectors)

    def test_GT_not_opened_when_predictions_are_missing(self):
        with patch.object(api.full,'validated_all',side_effect=ValueError('missing original masks')),patch.object(api.full,'read_gt') as read:
            with self.assertRaises(ValueError):api.score()
            read.assert_not_called()

    def test_worker_and_sequential_supervisor_arguments(self):
        from . import queue
        command=queue.worker_command(2,3,1)
        self.assertEqual(command[-6:],['--seed','2','--shard','3','--physical-gpu','1'])
        with patch('sys.argv',['queue','--seed','1']),patch.object(queue,'supervise') as supervise:
            queue.main();self.assertEqual(supervise.call_args_list,[((1,s),) for s in range(4)])

if __name__=='__main__':unittest.main()
