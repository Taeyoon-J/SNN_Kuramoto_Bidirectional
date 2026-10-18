import unittest
from unittest.mock import patch
import numpy as np
from . import run
from collaborative_test.SW_0194_native64_train64.test_train import temporary


class ReadoutProtocolTests(unittest.TestCase):
    def fixture(self):
        archive=temporary();mask=np.zeros((64,64),dtype=np.int64);mask[5:50,8:40]=1
        binding=dict(fingerprint=None,image_ids=list(range(1320,1352,4)),models=dict(TRAIN8='a',TRAIN64='b'),
            RGB_rows_sha256=['rgb']*8,validation_GT_used_for_prediction=False,
            frozen_old_mask_sha256={arm:[run.train.sha(mask.tobytes())]*8 for arm in run.ARMS})
        for arm in run.ARMS:
            hashes=[]
            for i in range(8):
                folder=archive/arm/'rows'/f'{i:04d}';folder.mkdir(parents=True)
                with (folder/'masks.npz').open('xb') as stream:np.savez_compressed(stream,old=mask,new=mask)
                run.train.write_json(folder/'row.json',dict(binding=run.row_binding(binding,arm,i),prediction_sha256=run.train.sha((folder/'masks.npz').read_bytes())))
                hashes.append(run.train.sha((folder/'row.json').read_bytes()))
            run.train.write_json(archive/arm/'protocol.json',dict(binding=binding,arm=arm,rows_sha256=hashes))
        return archive,mask

    def test_both_frozen_models_and_eight_rows_validate(self):
        archive,mask=self.fixture()
        with patch.object(run,'ARCHIVE',archive),patch.object(run,'fingerprint',return_value=None):
            labels,_=run.validated_pair()
            for arm in run.ARMS:np.testing.assert_array_equal(labels[arm]['old'],np.stack([mask]*8))

    def test_changed_foreground_rejected_even_with_updated_payload_hash(self):
        archive,mask=self.fixture();folder=archive/'TRAIN64/rows/0000'
        new=mask.copy();new[0,0]=1
        with (folder/'masks.npz').open('wb') as stream:np.savez_compressed(stream,old=mask,new=new)
        import json
        meta=json.loads((folder/'row.json').read_bytes());meta['prediction_sha256']=run.train.sha((folder/'masks.npz').read_bytes())
        expected=meta['binding'];(folder/'row.json').write_text(json.dumps(meta))
        with self.assertRaises(ValueError):run.read_row(folder,expected,run.train.sha(mask.tobytes()))

    def test_corruption_blocks_scoring_before_GT(self):
        archive,_=self.fixture()
        with (archive/'TRAIN8/rows/0000/masks.npz').open('ab') as stream:stream.write(b'bad')
        with patch.object(run,'ARCHIVE',archive),patch.object(run,'fingerprint',return_value=None),patch.object(run.evaluate.screen_score,'read_ground_truth') as gt:
            with self.assertRaises(ValueError):run.score()
            gt.assert_not_called()

    def test_reference_RGB_and_RNG_use_original_development_offsets(self):
        binding=dict(image_ids=list(range(1320,1352,4)),models=dict(TRAIN8='model'),RGB_rows_sha256=['rgb']*8)
        row=run.row_binding(binding,'TRAIN8',3)
        self.assertEqual(row['inference_seed'],193012);self.assertEqual(row['image_id'],1332)


if __name__=='__main__':unittest.main()
