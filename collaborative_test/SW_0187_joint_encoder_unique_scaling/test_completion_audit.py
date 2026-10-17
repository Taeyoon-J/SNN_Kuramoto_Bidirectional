import copy
import hashlib
import json
from pathlib import Path
import unittest
import numpy as np
from collaborative_test.SW_0187_joint_encoder_unique_scaling import score,completion_audit as audit,audit_queue

class AuditArithmetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder=Path(__file__).resolve().parents[1]/'SW_0183_shared_prefix_unique_scaling/results_archive'
        candidates=(folder/'score_v1/evaluation.json',folder/'evaluation_seedmean_20261016.json')
        expected='e49ebeeadfaf88e5a2b33002adb0d787f9c795aa91767b1f5728ad03381d9a11'
        matches=[path for path in candidates if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==expected]
        if len(matches)!=1:raise ValueError('one pinned arithmetic fixture required')
        previous=json.loads(matches[0].read_text());scores=previous['scores']
        current=score._read_sw0186_score()['payload']
        for endpoint in score.ENDPOINTS:scores[f'SW0186_{endpoint}']=current['scores'][f'candidate_{endpoint}']
        paired,gates=score.aggregate_and_gate(scores,np.random.RandomState(178).randint(0,320,size=(10000,320)))
        cls.fixture=dict(scores=scores,paired_bootstrap=paired,gates=gates)

    def test_independent_arithmetic_accepts_all_real_paired_metric_vectors(self):
        gates,_=audit.verify_arithmetic(self.fixture)
        self.assertFalse(gates['unique_data_gate_passed'])

    def test_forged_scaling_success_is_rejected(self):
        report=copy.deepcopy(self.fixture);report['gates']['unique_data_gate_passed']=True
        with self.assertRaises(ValueError):audit.verify_arithmetic(report)

    def test_wrong_paired_CI_is_rejected(self):
        report=copy.deepcopy(self.fixture)
        report['paired_bootstrap']['candidate1024_minus_control512_standalone_fg_ari']['ci95']=[1.,2.]
        with self.assertRaises(ValueError):audit.verify_arithmetic(report)

    def test_wrong_seed_mean_is_rejected(self):
        report=copy.deepcopy(self.fixture)
        report['scores']['candidate1024_standalone']['fg_ari']['per_seed_mean']['0']+=.01
        with self.assertRaises(ValueError):audit.verify_arithmetic(report)

    def test_unknown_pixel_partition_labels_do_not_change_independent_metrics(self):
        gt=np.array([[0,0,1],[1,2,2]])
        pred=np.array([[0,0,7],[7,3,3]])
        np.testing.assert_array_equal(audit.math_audit.image_metrics(pred,gt),np.ones(3))

class ProducerIdentity(unittest.TestCase):
    def test_only_actual_live_score_module_is_a_verified_wait(self):
        state=dict(experiment='SW0187_score_queue',status='waiting_for_predictions')
        proc=dict(state='S',argv=['python','-m',audit_queue.SCORE_MODULE,'--worker'])
        self.assertFalse(audit_queue.producer_ready(state,proc))
        for invalid in (None,{**proc,'state':'Z'},{**proc,'argv':['python','-m','other.module','--worker']}):
            with self.assertRaises(RuntimeError):audit_queue.producer_ready(state,invalid)

    def test_terminal_completion_and_failure_are_distinguished(self):
        self.assertTrue(audit_queue.producer_ready(dict(experiment='SW0187_score_queue',status='complete'),None))
        with self.assertRaises(RuntimeError):
            audit_queue.producer_ready(dict(experiment='SW0187_score_queue',status='failed'),None)

if __name__=='__main__':unittest.main()
