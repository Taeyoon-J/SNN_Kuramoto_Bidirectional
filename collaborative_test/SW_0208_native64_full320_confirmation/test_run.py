import io
import json
import unittest
from unittest.mock import patch,Mock
import numpy as np
import torch
from . import run as api

class ConfirmationTests(unittest.TestCase):
    def test_shards_cover_each_fixed_image_once(self):
        parts=[api.offsets(s) for s in range(4)]
        self.assertEqual([len(p) for p in parts],[80]*4)
        self.assertEqual(sorted(sum((list(p) for p in parts),[])),list(range(320)))
        self.assertEqual(api.IDS.tolist(),list(range(1320,1640)))

    def test_invalid_shards_rejected(self):
        for bad in (-1,4,True,'0',None):
            with self.assertRaises(ValueError):api.offsets(bad)

    def test_checkpoint_payload_hash_checked_before_unpickling_payload(self):
        stream=io.BytesIO();torch.save({'payload_bytes':b'corrupted','payload_sha256':'wrong'},stream)
        with self.assertRaisesRegex(ValueError,'hash'):api.unpack_checkpoint(stream.getvalue())

    def test_checkpoint_roundtrip_preserves_actual_weights(self):
        inner=io.BytesIO();torch.save({'wrapped_state':{'x':torch.tensor([1.,3.])}},inner)
        raw=inner.getvalue();outer=io.BytesIO();torch.save(dict(payload_bytes=raw,payload_sha256=api.sha(raw)),outer)
        self.assertTrue(torch.equal(api.unpack_checkpoint(outer.getvalue())['wrapped_state']['x'],torch.tensor([1.,3.])))

    def test_row_identity_binds_full_offset_rng_RGB_model(self):
        binding={'RGB_row_sha256':['rgb'+str(i) for i in range(320)]}
        row=api.row_binding(binding,'component204',319)
        self.assertEqual((row['image_id'],row['inference_seed'],row['RGB_sha256']),(1639,193319,'rgb319'))
        self.assertNotEqual(row['model_sha256'],api.row_binding(binding,'warm192',319)['model_sha256'])
        for arm,offset in [('other',0),('warm192',320),('warm192',True)]:
            with self.assertRaises(ValueError):api.row_binding(binding,arm,offset)

    def test_exact_old_mini_replay_required(self):
        record={'binding':{'model_sha256':api.MODELS['component204'],'inference_seed':193002}}
        with patch.object(api.Path,'read_bytes',return_value=json.dumps(record).encode()),patch.object(api.screen,'validate_saved_row',return_value=np.zeros((64,64),np.int64)):
            api.validate_overlap('component204',2,np.zeros((64,64),np.int64))
            with self.assertRaisesRegex(ValueError,'reproduce'):api.validate_overlap('component204',2,np.ones((64,64),np.int64))

    def test_unsupported_overlap_does_not_load_fake_reference(self):
        with patch.object(api.Path,'read_bytes',side_effect=AssertionError('no reference exists')):
            api.validate_overlap('component204',1,np.zeros((64,64),np.int64))
            api.validate_overlap('warm192',319,np.zeros((64,64),np.int64))

    def test_mini_and_nan_cannot_be_promoted_as_full320(self):
        for values in (np.ones((16,3)),np.full((320,3),np.nan)):
            with self.assertRaises(ValueError):api.summarize(dict(warm192=values,component204=values))

    def test_single_seed_high_score_never_claims_three_seed_success(self):
        values=np.ones((320,3))*.91
        result=api.summarize(dict(warm192=values,component204=values+.01))
        self.assertEqual(result['preferred_for_next_training'],'component204')
        self.assertFalse(result['full320_three_seed_goal_verified'])
        self.assertAlmostEqual(result['paired_component204_minus_warm192']['fg_ari']['mean'],.01)

    def test_no_GT_open_until_every_prediction_validates(self):
        with patch.object(api,'validated_all',side_effect=ValueError('missing shard')),patch.object(api,'read_gt') as reader:
            with self.assertRaisesRegex(ValueError,'missing shard'):api.score()
            reader.assert_not_called()

    def test_worker_uses_physical_lease_and_shard(self):
        from .queue import worker_command
        command=worker_command(3,2)
        self.assertEqual(command[-4:],['--shard','3','--physical-gpu','2'])
        with self.assertRaises(ValueError):worker_command(0,True)

if __name__=='__main__':unittest.main()
