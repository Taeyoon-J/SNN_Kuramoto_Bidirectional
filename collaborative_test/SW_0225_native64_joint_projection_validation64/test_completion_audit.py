import json
import unittest
import uuid
from unittest import mock
import numpy as np
from . import run, evaluate, completion_audit


class AuditTests(unittest.TestCase):
    def test_independent64_audit_and_corrupted_metric_rejection(self):
        root = run.ROOT / 'collaborative_testing/temporary' / ('SW0225_audit_' + uuid.uuid4().hex)
        root.mkdir(parents=True)
        mask = np.ones((64, 64, 64), np.int64)
        mask[:, 32:] = 2
        bundle = {}
        for seed in run.SEEDS:
            folder = root / f'seed{seed}'
            run.write_json(folder / 'queue/state.json', dict(status='complete'))
            protocol = dict(binding=dict(dataset={}, candidate=dict(checkpoint_sha256='fixture')))
            run.write_json(folder / 'protocol.json', protocol)
            run.write_json(folder / 'replay.json', dict(mask_bitwise_equal=True))
            bundle[seed] = dict(labels=mask, baseline=np.ones_like(mask), protocol=protocol)
        original = root / 'original'
        run.write_json(original / 'score_v1/summary.json', dict(full320_candidate_gate=False))
        def gt(_):
            self.assertEqual(validate.call_count, expected_calls[0])
            return np.tile(mask[:1], (320, 1, 1))
        expected_calls = [1]
        with mock.patch.object(run, 'ARCHIVE', root), mock.patch.object(run.scope.model, 'ARCHIVE', original), \
             mock.patch.object(evaluate, 'validated_all', return_value=bundle) as validate, \
             mock.patch.object(run.source.source.full, 'read_gt', side_effect=gt):
            run.score()
            expected_calls[0] = 2
            result = completion_audit.audit()
            self.assertEqual(result['raw_masks_independently_rescored'], 384)
            self.assertTrue(result['FG_track_candidate_gate'])
            self.assertLess(result['independent_metrics_means_and_bootstrap_maximum_abs_error'], 1e-12)
            private = root / 'score_v1/evaluation_private.json'
            content = json.loads(private.read_bytes())
            content['vectors']['joint_projection_seed0'][0][0] -= .1
            private.write_text(json.dumps(content))
            expected_calls[0] = 3
            with self.assertRaisesRegex(ValueError, 'independent metrics'):
                completion_audit.audit()


if __name__ == '__main__':
    unittest.main()
