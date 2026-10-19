import unittest
import json
from types import SimpleNamespace
from unittest import mock
import numpy as np
from . import scope


class ScopeTests(unittest.TestCase):
    def test_only_independently_accepted235_can_expand(self):
        fingerprint={'fixture':'immutable235'}
        score=json.dumps(dict(fingerprint=fingerprint,balanced_candidate_gate=True)).encode()
        audit=dict(score_sha256=scope.reference.sha(score),
            diagnostic_source_sha256=scope.reference.sha((scope.reference.HERE/'completion_audit.py').read_bytes()),
            balanced_candidate_gate=True,original16_gate_failure_preserved=True,
            raw_masks_independently_rescored=576,independent_metrics_means_and_bootstrap_maximum_abs_error=0.)
        def file(name):
            return SimpleNamespace(read_bytes=lambda:score if name=='score' else json.dumps(audit).encode())
        with mock.patch.object(scope,'reference_file',side_effect=file),mock.patch.object(scope.reference,'fingerprint',return_value=fingerprint):
            scope.selection_fingerprint()
            for key,bad in [('balanced_candidate_gate',False),('raw_masks_independently_rescored',384),('score_sha256','changed')]:
                old=audit[key];audit[key]=bad
                with self.assertRaises(ValueError):scope.selection_fingerprint()
                audit[key]=old

    def test_full320_contains_exact_old64_and_new256(self):
        self.assertEqual(len(scope.MINI),64);self.assertEqual(len(scope.NEW),256)
        self.assertFalse(set(scope.MINI)&set(scope.NEW));self.assertEqual(sorted(scope.MINI+scope.NEW),list(range(320)))
        self.assertIn(0,scope.MINI);self.assertEqual((1320+scope.OFFSETS[0],1320+scope.OFFSETS[-1]),(1320,1639))

    def test_existing_canonical_float_conversion_and_byte_RGB_preservation(self):
        data=np.zeros((2,128,128,3),np.float32);data[...,1]=.5;data[...,2]=1.
        rgb=scope.canonical_rgb(data);self.assertEqual(rgb.dtype,np.uint8);np.testing.assert_array_equal(rgb[0,0,0],[0,128,255])
        before=rgb.copy();np.testing.assert_array_equal(scope.canonical_rgb(rgb),before);self.assertEqual(tuple(rgb.shape),(2,128,128,3))

    def test_geometry_not_silently_resized(self):
        with self.assertRaises(ValueError):scope.canonical_rgb(np.zeros((2,126,126,3),np.float32))


if __name__=='__main__':unittest.main()
