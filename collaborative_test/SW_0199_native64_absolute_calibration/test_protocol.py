import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from . import train,evaluate


class ProtocolTests(unittest.TestCase):
    def tearDown(self):train.configure('control')

    def test_matched_factory_preserves_real_loaded_objects_and_binds_arm(self):
        prepared=dict(bindings={'kept':'lineage'},parent_optimizer_content_sha256='adam',parent_RNG_content_sha256='rng',found=object(),optimizer=object())
        original_optimizer=prepared['optimizer']
        with patch.object(train.parent,'prepare_inputs_cpu',return_value=prepared),patch.object(train.probe.warm,'_base_encoder_sha',return_value='cnn'):
            train.configure('calibrated');value=train.prepare_inputs_cpu()
        self.assertIs(value['optimizer'],original_optimizer)
        self.assertEqual(value['bindings']['kept'],'lineage')
        self.assertEqual(value['bindings']['parent_optimizer_content_sha256'],'adam')
        self.assertEqual(value['bindings']['parent_RNG_content_sha256'],'rng')
        self.assertEqual(value['bindings']['fingerprint']['arm'],'calibrated')
        self.assertEqual(value['bindings']['total_unique_TRAIN'],16)

    def test_history_rejects_repeated_or_changed_prefix(self):
        rows=[dict(proposal=i+1,indices=list(range(i*4,(i+1)*4)),parity_error=0,acceptance_threshold=1e-6) for i in range(4)]
        train.validate_history(rows,rows[:2])
        broken=copy.deepcopy(rows);broken[2]['indices']=[0,1,2,3]
        with self.assertRaises(ValueError):train.validate_history(broken,rows[:2])
        with self.assertRaises(ValueError):train.validate_history(rows,[])

    def test_candidate_must_beat_warm_and_actually_learn(self):
        x=np.zeros((16,3));y=x+.02
        paired={k:evaluate.paired_change(y,x) for k in ('control','warm')}
        report={'calibrated':dict(new_accepted_updates=1,committed_model_changed=True,CNN_changed_from_warm=True)}
        self.assertTrue(evaluate.extension_gate(paired,report))
        paired['warm']['fg_ari']['mean']=-.001
        self.assertFalse(evaluate.extension_gate(paired,report))
        paired['warm']['fg_ari']['mean']=.02;report['calibrated']['CNN_changed_from_warm']=False
        self.assertFalse(evaluate.extension_gate(paired,report))

    def test_fixed_offset_cohort_not_full_goal(self):
        self.assertEqual(evaluate.OFFSETS,tuple(range(0,32,2)))
        self.assertEqual(train.TOTAL,4)
        with self.assertRaises(ValueError):evaluate.paired_change(np.zeros((32,3)),np.zeros((32,3)))

    def test_no_GT_open_until_both_prediction_protocols_frozen(self):
        # fcntl is Linux-only, so exercise the actual scoring entry on the server.
        try:import fcntl
        except ImportError:self.skipTest('Linux queue scoring lock')
        with tempfile.TemporaryDirectory(dir=train.HERE.parents[1]/'collaborative_testing/temporary') as temp:
            root=Path(temp)
            with patch.object(train,'ROOT_ARCHIVE',root),patch.object(evaluate.screen_score,'read_ground_truth') as gt:
                result=evaluate.score_candidate()
            self.assertEqual(result['status'],'arm_complete_waiting_for_matched_pair')
            self.assertFalse(result['screen_gate_passed']);gt.assert_not_called()

    def test_worker_command_keeps_its_arm_and_physical_gpu(self):
        from . import queue
        train.configure('calibrated');cmd=queue.worker_command(3)
        self.assertEqual(cmd[-2:],['--arm','calibrated'])
        self.assertEqual(cmd[cmd.index('--physical-gpu')+1],'3')
        with self.assertRaises(ValueError):queue.worker_command(-1)
