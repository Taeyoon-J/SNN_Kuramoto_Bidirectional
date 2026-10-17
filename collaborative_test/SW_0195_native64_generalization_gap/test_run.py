import json
import unittest
from unittest.mock import patch
import numpy as np
from . import run
from collaborative_test.SW_0194_native64_train64.test_train import temporary


class FrozenPredictionTests(unittest.TestCase):
    def fixture(self):
        archive = temporary()
        binding = dict(fingerprint=None, image_ids=list(range(16)), models=dict(TRAIN8='old', TRAIN64='new'),
                       RGB_rows_sha256=['rgb']*16, GT_used_for_prediction=False)
        labels = np.zeros((16,64,64), dtype=np.int64)
        labels[:,5:50,8:40] = 1
        for arm in run.ARMS:
            for i in range(16):
                folder = archive/arm/'rows'/f'{i:04d}'; folder.mkdir(parents=True)
                with (folder/'labels.npz').open('xb') as stream: np.savez_compressed(stream, labels=labels[i])
                run.train.write_json(folder/'row.json', dict(binding=run.row_binding(binding, arm, i),
                    prediction_sha256=run.train.sha((folder/'labels.npz').read_bytes())))
            run.train.write_json(archive/arm/'protocol.json', dict(binding=binding, arm=arm,
                labels_sha256=run.train.sha(labels.tobytes()), optimizer_updates=0))
        return archive, labels

    def test_same_rng_and_cohort_but_different_models(self):
        binding = dict(image_ids=[42], models=dict(TRAIN8='a', TRAIN64='b'), RGB_rows_sha256=['rgb'])
        a,b = [run.row_binding(binding, arm, 0) for arm in run.ARMS]
        self.assertEqual(a['inference_seed'], b['inference_seed'])
        self.assertEqual(a['image_id'], 42)
        self.assertNotEqual(a['model_sha256'], b['model_sha256'])

    def test_real_saved_rows_and_pair_digest_validate(self):
        archive, labels = self.fixture()
        with patch.object(run,'ARCHIVE',archive), patch.object(run,'fingerprint',return_value=None):
            actual,_ = run.validated_pair()
            for arm in run.ARMS: np.testing.assert_array_equal(actual[arm], labels)

    def test_corruption_blocks_scoring_before_any_GT(self):
        archive,_ = self.fixture()
        with (archive/'TRAIN64/rows/0015/labels.npz').open('ab') as stream: stream.write(b'bad')
        with patch.object(run,'ARCHIVE',archive), patch.object(run,'fingerprint',return_value=None), \
             patch.object(run.evaluate.screen_score,'read_ground_truth') as gt:
            with self.assertRaises(ValueError): run.score()
            gt.assert_not_called()

    def test_mutated_training_flag_or_missing_pair_rejected(self):
        archive,_ = self.fixture()
        path = archive/'TRAIN64/protocol.json'; value=json.loads(path.read_bytes())
        value['optimizer_updates'] = 1; path.write_text(json.dumps(value))
        with patch.object(run,'ARCHIVE',archive), patch.object(run,'fingerprint',return_value=None):
            with self.assertRaises(ValueError):run.validated_pair()


if __name__ == '__main__': unittest.main()
