import copy
import json
import unittest
from unittest.mock import patch
import numpy as np
import torch
from . import train,evaluate
from .test_train import fixture,temporary


class EvaluationTests(unittest.TestCase):
    def create_rows(self):
        archive=temporary();ids=np.array([1320,1321]);reference={'development_RGB_row_sha256':['first','second']}
        parent_history=[dict(proposal=i+1,indices=list(range(4*i,4*i+4)),accepted=True,parity_error=0.,acceptance_threshold=1e-6) for i in range(2)]
        parent=archive/'parent';(parent/'probe_seed0').mkdir(parents=True)
        train.write_json(parent/'probe_seed0/report.json',dict(history=parent_history))
        bindings=dict(fingerprint=None)
        history=parent_history+[dict(proposal=i+1,indices=list(range(4*i,4*i+4)),accepted=True,parity_error=0.,acceptance_threshold=1e-6) for i in range(2,16)]
        report=dict(bindings=bindings,status='complete_native64_TRAIN64',final_model_state_sha256='trained',history=history,
            inherited_proposals=2,new_proposals=14,total_unique_TRAIN=64,total_exposures=64,accepted_updates=16,actual_optimizer_steps=[16])
        train.write_json(archive/'training/report.json',report)
        labels=np.zeros((2,64,64),dtype=np.int64);labels[:,8:40,5:52]=1
        for i in range(2):
            folder=archive/'candidate/rows'/f'{i:04d}';folder.mkdir(parents=True)
            with (folder/'labels.npz').open('xb') as stream:np.savez_compressed(stream,labels=labels[i])
            train.write_json(folder/'row.json',dict(binding=evaluate.row_binding(bindings,'trained',reference,i),prediction_sha256=train.sha((folder/'labels.npz').read_bytes())))
        train.write_json(archive/'candidate/protocol.json',dict(bindings=bindings,reference_binding=reference,
            model_sha256='trained',image_ids=ids.tolist(),validation_GT_read=False,
            training_report_sha256=train.sha((archive/'training/report.json').read_bytes()),labels_sha256=train.sha(labels.tobytes())))
        return archive,parent,ids,labels

    def test_candidate_training_and_prediction_hashes_validate_before_GT(self):
        archive,parent,ids,labels=self.create_rows()
        with patch.object(train,'ARCHIVE',archive),patch.object(train.probe,'ARCHIVE',parent),patch.object(evaluate.screen,'IDS',ids),patch.object(train,'fingerprint',return_value=None):
            actual,_=evaluate.validated_candidate();np.testing.assert_array_equal(actual,labels)
            with (archive/'candidate/rows/0001/labels.npz').open('ab') as stream:stream.write(b'corrupt')
            with patch.object(evaluate.screen_score,'read_ground_truth') as gt:
                with self.assertRaises(ValueError):evaluate.score_candidate()
                gt.assert_not_called()

    def test_changed_completed_training_report_blocks_candidate_scoring(self):
        archive,parent,ids,_labels=self.create_rows()
        with (archive/'training/report.json').open('ab') as stream:stream.write(b' ')
        with patch.object(train,'ARCHIVE',archive),patch.object(train.probe,'ARCHIVE',parent),patch.object(evaluate.screen,'IDS',ids),patch.object(train,'fingerprint',return_value=None),patch.object(evaluate.screen_score,'read_ground_truth') as gt:
            with self.assertRaises(ValueError):evaluate.score_candidate()
            gt.assert_not_called()

    def test_inference_row_seed_and_source_RGB_bind_to_existing_development_protocol(self):
        reference={'development_RGB_row_sha256':['canonical']}
        row=evaluate.row_binding({'source':'frozen'},'trained',reference,0)
        self.assertEqual(row['inference_seed'],193000);self.assertEqual(row['RGB_sha256'],'canonical')
        self.assertEqual(row['image_id'],1320);self.assertEqual(row['model_sha256'],'trained')

    def test_two_proposals_cannot_be_scored_as_completed_TRAIN64(self):
        archive,parent,ids,_labels=self.create_rows()
        report_path=archive/'training/report.json';report=json.loads(report_path.read_bytes())
        report.update(history=report['history'][:2],accepted_updates=2,actual_optimizer_steps=[2])
        report_path.write_text(json.dumps(report))
        protocol_path=archive/'candidate/protocol.json';protocol=json.loads(protocol_path.read_bytes())
        protocol['training_report_sha256']=train.sha(report_path.read_bytes());protocol_path.write_text(json.dumps(protocol))
        with patch.object(train,'ARCHIVE',archive),patch.object(train.probe,'ARCHIVE',parent),patch.object(evaluate.screen,'IDS',ids),patch.object(train,'fingerprint',return_value=None),patch.object(evaluate.screen_score,'read_ground_truth') as gt:
            with self.assertRaises(ValueError):evaluate.score_candidate()
            gt.assert_not_called()

if __name__=='__main__':unittest.main()
