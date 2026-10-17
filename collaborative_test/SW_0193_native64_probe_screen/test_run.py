import contextlib
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid
import numpy as np
import torch
from . import run,score


def temporary():
    p=run.HERE.parents[1]/'collaborative_testing/temporary'/('sw0193_'+uuid.uuid4().hex)
    p.mkdir(parents=True,exist_ok=False);return p


class ScreenTests(unittest.TestCase):
    def test_parent_requires_committed_learning_all_credit_and_Adam_steps(self):
        binding={'source':'fixed'}
        report=dict(status='complete_native64_training_admission',bindings=binding,
            history=[{'credit_failures':[]},{'credit_failures':[]}],accepted_updates=1,
            actual_optimizer_steps=[1],source_CNN_changed=True,validation_data_read=False)
        run.validate_parent_report(report,binding)
        for key,value in [('accepted_updates',0),('actual_optimizer_steps',[2]),('source_CNN_changed',False),('validation_data_read',True),('status','running')]:
            bad=copy.deepcopy(report);bad[key]=value
            with self.assertRaises(ValueError):run.validate_parent_report(bad,binding)
        bad=copy.deepcopy(report);bad['history'][0]['credit_failures']=['dead graph']
        with self.assertRaises(ValueError):run.validate_parent_report(bad,binding)

    def test_saved_native64_row_checks_bytes_geometry_and_exact_binding(self):
        p=temporary();expected={'model':'one'}
        with (p/'labels.npz').open('xb') as stream:np.savez_compressed(stream,labels=np.ones((64,64),dtype=np.int64))
        run.write_json(p/'row.json',dict(binding=expected,prediction_sha256=run.sha((p/'labels.npz').read_bytes())))
        self.assertEqual(run.validate_saved_row(p,expected).shape,(64,64))
        with self.assertRaises(ValueError):run.validate_saved_row(p,{'model':'other'})
        with (p/'labels.npz').open('ab') as stream:stream.write(b'tamper')
        with self.assertRaises(ValueError):run.validate_saved_row(p,expected)

    def test_actual_production_and_independent_metrics_match_at64(self):
        gt=np.zeros((2,64,64),dtype=np.int64)
        gt[:,4:24,4:24]=1;gt[:,30:58,28:59]=2
        pred=gt.copy();pred[0,4:12,4:24]=3;pred[1,35:42,30:40]=0
        vector,error=score.measured_vectors(pred,gt)
        self.assertEqual(vector.shape,(2,3));self.assertLess(error,1e-12)
        perfect,_=score.measured_vectors(gt,gt)
        np.testing.assert_allclose(perfect,np.ones((2,3)),rtol=0,atol=1e-12)

    def test_real_mask_conversion_and_foreground_merge_dependencies_at64(self):
        found=SimpleNamespace(wrapped=SimpleNamespace(core=SimpleNamespace(graph_generator=SimpleNamespace(log_coupling_gain=torch.tensor(0.)))))
        trace={'spikes':torch.zeros(1,1,1),'component_spikes':torch.zeros(1,4,1,1)}
        affinity=torch.zeros(1,4096,4096)
        with patch('snn_kuramoto_bidirectional.spike_classifier.spike_synchrony_components',return_value=[[(65,66,129,130)]]) as groups:
            labels=run.prediction_from_trace(found,trace,affinity,affinity)
        self.assertEqual(labels.shape,(64,64));self.assertEqual(int((labels>0).sum()),4)
        self.assertEqual(groups.call_args.kwargs['spatial_grid_size'],64)
        self.assertEqual(groups.call_args.kwargs['min_group_size'],32)

    def fixture(self):
        wrapped=torch.nn.Linear(1,1,bias=False);wrapped.weight.data.fill_(.25)
        encoder=torch.nn.Linear(1,1,bias=False);encoder.weight.data.fill_(.5)
        found=SimpleNamespace(wrapped=wrapped,encoder=encoder,patcher=torch.nn.Identity())
        baseline=run.native._state_sha(wrapped,encoder)
        candidate={'wrapped':copy.deepcopy(wrapped.state_dict()),'encoder':copy.deepcopy(encoder.state_dict())}
        candidate['encoder']['weight'].fill_(.75)
        encoder.load_state_dict(candidate['encoder']);trained=run.native._state_sha(wrapped,encoder);encoder.weight.data.fill_(.5)
        rgb=np.zeros((2,128,128,3),dtype=np.uint8)
        binding=dict(baseline_model_sha256=baseline,candidate_model_sha256=trained,
            fingerprint=None,development_RGB_row_sha256=[run.sha(row.tobytes()) for row in rgb])
        return dict(found=found,candidate_state=candidate,rgb=rgb,bindings=binding)

    def produce_fixture(self,p):
        prepared=self.fixture();ids=np.arange(1320,1322,dtype=np.int64)
        labels=np.zeros((64,64),dtype=np.int64);labels[5:40,8:49]=1
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(run,'ARCHIVE',p));stack.enter_context(patch.object(run,'IDS',ids))
            stack.enter_context(patch.object(torch.cuda,'set_device'));stack.enter_context(patch.object(torch.cuda,'manual_seed'))
            forward=stack.enter_context(patch.object(run.native,'_native_forward',return_value=(None,{},None,None,None,None)))
            stack.enter_context(patch.object(run,'prediction_from_trace',return_value=labels))
            result=run.predict_pair(prepared,device='cpu')
            self.assertEqual(result['optimizer_updates'],0);self.assertEqual(forward.call_count,4)
            # A yielded worker constructs the original baseline again, then reuses
            # both immutable rows with exactly the same hashes and model bindings.
            restarted=self.fixture();forward.reset_mock();run.predict_pair(restarted,device='cpu')
            self.assertEqual(forward.call_count,0)
        return ids,prepared['bindings']

    def test_real_prediction_loop_preserves_models_and_reuses_only_matching_rows(self):
        p=temporary();self.produce_fixture(p)
        for arm in run.ARMS:self.assertTrue((p/arm/'protocol.json').is_file())

    def test_corrupted_candidate_prevents_first_GT_access(self):
        p=temporary();ids,binding=self.produce_fixture(p)
        path=p/'candidate/rows/0001/labels.npz'
        with path.open('ab') as stream:stream.write(b'broken')
        with patch.object(run,'ARCHIVE',p),patch.object(run,'IDS',ids),patch.object(run,'fingerprint',return_value=None),patch.object(score,'read_ground_truth') as gt:
            with self.assertRaises(ValueError):score.score_pair()
            gt.assert_not_called()

    def test_validated_pair_rejects_RGB_provenance_change(self):
        p=temporary();ids,_binding=self.produce_fixture(p)
        path=p/'baseline/rows/0000/row.json';record=json.loads(path.read_bytes())
        record['binding']['RGB_sha256']='bad';path.write_text(json.dumps(record))
        with patch.object(run,'ARCHIVE',p),patch.object(run,'IDS',ids),patch.object(run,'fingerprint',return_value=None):
            with self.assertRaises(ValueError):score.validated_pair()

if __name__=='__main__':unittest.main()
