import unittest
from unittest.mock import patch
import numpy as np
from . import run
from collaborative_test.SW_0194_native64_train64.test_train import temporary


class BaselineReplicationTests(unittest.TestCase):
    def fixture(self):
        root=temporary();archive=root/'seed1'
        binding=dict(fingerprint=None,seed=1,model_sha256='source1',validation_GT_used_for_prediction=False,
            optimizer_updates=0,reference_binding=dict(development_RGB_row_sha256=['rgb']*32))
        labels=np.zeros((32,64,64),dtype=np.int64);labels[:,5:50,8:40]=1
        for i in range(32):
            folder=archive/'prediction/rows'/f'{i:04d}';folder.mkdir(parents=True)
            with (folder/'labels.npz').open('xb') as stream:np.savez_compressed(stream,labels=labels[i])
            run.train.write_json(folder/'row.json',dict(binding=run.row_binding(binding,i),prediction_sha256=run.train.sha((folder/'labels.npz').read_bytes())))
        run.train.write_json(archive/'prediction/protocol.json',dict(binding=binding,labels_sha256=run.train.sha(labels.tobytes())))
        return root,labels

    def test_seed_outputs_separate_and_invalid_seeds_rejected(self):
        self.assertNotEqual(run.archive_for(1),run.archive_for(2))
        for seed in (0,3,True):
            with self.assertRaises(ValueError):run.archive_for(seed)

    def test_binding_uses_same_inference_RNG_as_frozen_seed0(self):
        binding=dict(seed=2,model_sha256='model2',reference_binding=dict(development_RGB_row_sha256=['rgb']*32))
        row=run.row_binding(binding,0)
        self.assertEqual(row['image_id'],1320);self.assertEqual(row['inference_seed'],193000)
        self.assertEqual(row['seed'],2);self.assertEqual(row['model_sha256'],'model2')

    def test_actual_frozen_rows_validate_without_GT(self):
        root,labels=self.fixture()
        with patch.object(run,'ROOT_ARCHIVE',root),patch.object(run,'fingerprint',return_value=None):
            actual,binding=run.validated_pair(1)
            np.testing.assert_array_equal(actual,labels);self.assertEqual(binding['seed'],1)

    def test_corrupted_prediction_blocks_scoring_before_GT(self):
        root,_=self.fixture()
        with (root/'seed1/prediction/rows/0031/labels.npz').open('ab') as stream:stream.write(b'bad')
        with patch.object(run,'ROOT_ARCHIVE',root),patch.object(run,'RUN_SEED',1),patch.object(run,'fingerprint',return_value=None),patch.object(run.metrics,'read_ground_truth') as gt:
            with self.assertRaises(ValueError):run.score()
            gt.assert_not_called()

    def test_missing_second_seed_does_not_open_aggregate_GT(self):
        root,_=self.fixture()
        with patch.object(run,'ROOT_ARCHIVE',root),patch.object(run.metrics,'read_ground_truth') as gt:
            run.maybe_aggregate();gt.assert_not_called()


if __name__=='__main__':unittest.main()
