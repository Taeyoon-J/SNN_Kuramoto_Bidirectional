import copy,json,uuid,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from . import run as api,classifier


def temporary():
    path=api.HERE.parents[1]/'collaborative_testing/temporary'/('sw0206_'+uuid.uuid4().hex)
    path.mkdir(parents=True);return path


class ArtifactTests(unittest.TestCase):
    def setUp(self):api.configure(2)

    def test_reject_other_seed_and_wrong_GPU(self):
        for seed in (0,1,True):
            with self.assertRaises(ValueError):api.configure(seed)
        from .queue import worker_command
        with self.assertRaises(ValueError):worker_command(4)
        command=worker_command(3)
        self.assertEqual(command[command.index('--seed')+1],'2')

    def test_cache_is_bound_to_sample_plan_and_binary_targets(self):
        path=temporary();expected={'plan':'a'}
        def write(target):
            np.savez_compressed(path/'pairs.npz',features=np.zeros((8192,4),np.float32),target=target)
            api.write_json(path/'row.json',dict(binding=expected,sha256=api.sha((path/'pairs.npz').read_bytes())),replace=True)
        y=np.r_[np.ones(4096,np.float32),np.zeros(4096,np.float32)]
        write(y);x,actual=api.read_cache(path,expected)
        self.assertEqual(x.shape,(8192,4));np.testing.assert_array_equal(actual,y)
        with self.assertRaises(ValueError):api.read_cache(path,{'plan':'changed'})
        write(y.astype(np.float64))
        with self.assertRaises(ValueError):api.read_cache(path,expected)

    def test_prediction_cannot_change_baseline_or_foreground_support(self):
        path=temporary();old=np.zeros((64,64),np.int64);old[3:5,3:5]=1
        binding={'frozen_old_sha256':[api.sha(old.tobytes())]}
        def write(new):
            np.savez_compressed(path/'masks.npz',old=old,new=new)
            api.write_json(path/'row.json',dict(binding=binding,index=0,classifier_sha256='head',sha256=api.sha((path/'masks.npz').read_bytes())),replace=True)
        write(old*2);api.read_row(path,binding,0,'head')
        with self.assertRaises(ValueError):api.read_row(path,binding,0,'other_head')
        new=old.copy();new[0,0]=1;write(new)
        with self.assertRaises(ValueError):api.read_row(path,binding,0,'head')

    def test_actual_classifier_Adam_payload_binds_weights_and_data(self):
        x=np.tile(np.array([[-.5,0,0,0],[.5,0,0,0]],np.float32),(8,1));y=np.tile(np.array([0.,1.],np.float32),8)
        head,opt,report=classifier.fit(x,y)
        payload=dict(binding={'source':'frozen'},cache_sha256=['cache'],state_dict=head.state_dict(),optimizer_state=opt,fit_report=report)
        loaded,_=api.validate_head_payload(payload,payload['binding'],['cache'])
        self.assertEqual(classifier.state_sha(loaded.state_dict()),report['final_parameter_sha256'])
        with self.assertRaises(ValueError):api.validate_head_payload(payload,payload['binding'],['different'])
        altered=copy.deepcopy(payload);next(iter(altered['state_dict'].values())).add_(.01)
        with self.assertRaises(ValueError):api.validate_head_payload(altered,payload['binding'],['cache'])

    def test_validation_must_not_refit_missing_classifier(self):
        root=temporary();binding={'source':'frozen'}
        fake=(np.zeros((8192,4),np.float32),np.r_[np.ones(4096,np.float32),np.zeros(4096,np.float32)])
        with patch.object(api,'ARCHIVE',root),patch.object(api,'read_cache',return_value=fake),patch.object(api,'cache_binding',return_value={}),patch.object(Path,'read_bytes',return_value=b'cache'),patch.object(classifier,'fit',side_effect=AssertionError('validation refit forbidden')):
            with self.assertRaises(FileNotFoundError):api.head_from_cache(binding)
