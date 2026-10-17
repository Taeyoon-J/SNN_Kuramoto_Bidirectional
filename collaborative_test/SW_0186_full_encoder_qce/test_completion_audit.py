import unittest
import hashlib
import json
from pathlib import Path
import numpy as np
from collaborative_test.SW_0186_full_encoder_qce import completion_audit as a,audit_queue as q

class IndependentMetrics(unittest.TestCase):
    def test_all_means_CI_and_registered_gate_on_pinned_actual_previous_vectors(self):
        from collaborative_test.SW_0186_full_encoder_qce import score
        folder=Path(__file__).resolve().parents[1]/'SW_0183_shared_prefix_unique_scaling/results_archive'
        candidates=(folder/'score_v1/evaluation.json',folder/'evaluation_seedmean_20261016.json')
        expected='e49ebeeadfaf88e5a2b33002adb0d787f9c795aa91767b1f5728ad03381d9a11'
        matches=[p for p in candidates if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==expected]
        self.assertEqual(len(matches),1)
        previous=json.loads(matches[0].read_text())
        scores={key:value for key,value in previous['scores'].items() if key.startswith(('SW','Slot'))}
        for endpoint in ('conditional','standalone'):
            scores[f'candidate_{endpoint}']=previous['scores'][f'candidate1024_{endpoint}']
        indices=np.random.RandomState(178).randint(0,320,size=(10000,320))
        paired,gate=score._aggregate_paired_and_gate(scores,indices)
        payload=dict(scores=scores,paired_bootstrap=paired,gate=gate)
        checked,_=a.verify_arithmetic(payload)
        self.assertEqual(checked,gate)
        self.assertFalse(checked['primary_gate_passed'])
        payload['gate']={**gate,'primary_gate_passed':True}
        with self.assertRaises(ValueError):a.verify_arithmetic(payload)

    def test_perfect_partition_with_renamed_object_ids(self):
        gt=np.array([[0,0,1],[1,2,2]])
        pred=np.array([[0,0,7],[7,3,3]])
        np.testing.assert_array_equal(a.image_metrics(pred,gt),np.ones(3))

    def test_foreground_ARI_keeps_predicted_background_as_a_cluster(self):
        gt=np.array([[1,1],[2,2]]);pred=np.zeros((2,2),dtype=int)
        np.testing.assert_array_equal(a.image_metrics(pred,gt),[0.,0.,0.])

    def test_unmatched_truth_objects_are_not_dropped_from_IoU_denominator(self):
        gt=np.array([[1,1],[2,2]]);pred=np.array([[4,4],[0,0]])
        values=a.image_metrics(pred,gt)
        self.assertEqual(values[2],.5)
        self.assertEqual(values[1],.5)

    def test_modal_ties_choose_smallest_ID_including_background(self):
        pixels=np.array([[[2,1,4,0],[1,2,0,4]]])
        np.testing.assert_array_equal(a.modal_labels(pixels,patch=2),np.array([[[1,0]]]))

    def test_modal_geometry_and_dtype_rejected(self):
        with self.assertRaises(ValueError):a.modal_labels(np.zeros((1,3,3),dtype=int),patch=2)
        with self.assertRaises(ValueError):a.modal_labels(np.zeros((1,4,4),dtype=float),patch=2)

class AuditProducer(unittest.TestCase):
    def test_pending_score_requires_exact_live_module(self):
        state=dict(experiment='SW0186_score_queue',status='waiting_for_predictions')
        proc=dict(state='S',argv=['python','-m',q.SCORE_MODULE,'--worker'])
        self.assertFalse(q.producer_ready(state,proc))
        with self.assertRaises(RuntimeError):q.producer_ready(state,None)
        proc['argv']=['python','-m','other.score_queue','--worker']
        with self.assertRaises(RuntimeError):q.producer_ready(state,proc)

    def test_terminal_complete_is_allowed_with_exited_producer(self):
        self.assertTrue(q.producer_ready(dict(experiment='SW0186_score_queue',status='complete'),None))

if __name__=='__main__':unittest.main()
