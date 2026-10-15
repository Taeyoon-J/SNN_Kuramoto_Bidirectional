"""SW0168 v3 prediction producer to score-v4 consumer identity contract."""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock

from collaborative_test.SW_0168_end_to_end_connectivity import (
    prediction_queue_v3 as prediction, score_queue_v4 as scoring, train, recover_completed_training as recovery,
)
from collaborative_test.SW_0168_end_to_end_connectivity.test_training_recovery_v3 import _new_fixture, _remove_fixture

class ScoreV4HandoffTests(TestCase):
    def test_actual_v3_state_producer_is_consumed_for_live_and_complete_states(self):
        temp=_new_fixture("sw168-score-v4-handoff")
        predroot=temp/"pred"; predroot.mkdir()
        receipt=temp/"receipt.json"; receipt.write_text("{}",encoding="utf-8")
        old=(prediction.RUNTIME,recovery.RECEIPT,scoring.PREDICTION_ROOT)
        prediction.RUNTIME=predroot; recovery.RECEIPT=receipt; scoring.PREDICTION_ROOT=predroot
        try:
            arm=train.ARMS[0]
            argv=["python","-m",scoring.PRED_MODULE,"--arm",arm]
            owner=SimpleNamespace(_proc_record=lambda pid:{"state":"S","argv":argv})
            prediction._write_state(arm,"running_prediction",child_pid=1234)
            self.assertIs(scoring._prediction_state(arm,owner,recovery._sha(receipt)),False)
            output=temp/"output"; output.mkdir()
            (output/"predictions.npz").write_bytes(b"predictions")
            (output/"protocol.json").write_text("{}",encoding="utf-8")
            prediction._write_state(arm,"complete",output=str(output),
                prediction_sha256=recovery._sha(output/"predictions.npz"),
                protocol_sha256=recovery._sha(output/"protocol.json"))
            consumed=scoring._prediction_state(arm,owner,recovery._sha(receipt))
            self.assertEqual(consumed["experiment"],prediction.EXPERIMENT)
            self.assertEqual(consumed["arm"],arm)
            self.assertEqual(consumed["status"],"complete")
        finally:
            prediction.RUNTIME,recovery.RECEIPT,scoring.PREDICTION_ROOT=old
            _remove_fixture(temp)

if __name__=="__main__":
    import unittest
    unittest.main()
