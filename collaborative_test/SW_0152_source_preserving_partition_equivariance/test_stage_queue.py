import json
import os
import sys
import uuid
import unittest
from pathlib import Path
from unittest import mock

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from collaborative_test.SW_0152_source_preserving_partition_equivariance import stage_queue as queue


class StageQueueTests(unittest.TestCase):
    def test_registered_six_task_dependencies_and_child_routes(self):
        tasks=queue.task_plan();by={row["task_id"]:row for row in tasks}
        self.assertEqual(set(by),{"train_equivariance","train_mass_control","predict_equivariance",
                                 "predict_mass_control","predict_source_p","score"})
        self.assertEqual(by["train_equivariance"]["depends_on"],["external_preflight"])
        self.assertEqual(by["predict_mass_control"]["depends_on"],["train_mass_control"])
        self.assertEqual(set(by["score"]["depends_on"]),{"predict_equivariance","predict_mass_control","predict_source_p"})
        self.assertIn("SW_0152_source_preserving_partition_equivariance.train", " ".join(queue._argv(by["train_equivariance"])))
        self.assertIn("SW_0152_source_preserving_partition_equivariance.evaluate", " ".join(queue._argv(by["predict_source_p"])))
        fp=queue.implementation_fingerprint()
        self.assertTrue(fp)
        self.assertTrue(all((ROOT / path).is_file() for path in fp))

    def test_preflight_malformed_partial_waits_until_terminal_failure(self):
        temp=queue.HERE/"results_archive"/f"test_preflight_{os.getpid()}_{uuid.uuid4().hex}"
        temp.mkdir(parents=True)
        report=temp/"preflight.json";report.write_text("{partial",encoding="utf-8")
        state=temp/"state.json";state.write_text(json.dumps({"status":"running"}),encoding="utf-8")
        try:
            with mock.patch.object(queue.preflight,"OUTPUT_DEFAULT",report), \
                 mock.patch.object(queue,"preflight_queue_state_path",return_value=state), \
                 mock.patch.object(queue.train,"validate_preflight",side_effect=ValueError("incomplete")):
                self.assertEqual(queue._external_preflight_state()[0],"waiting_for_external_preflight")
                state.write_text(json.dumps({"status":"failed_child"}),encoding="utf-8")
                self.assertEqual(queue._external_preflight_state()[0],"blocked_invalid_external_preflight")
        finally:
            report.unlink(missing_ok=True);state.unlink(missing_ok=True);temp.rmdir()

    def test_failed_external_admission_never_reserves_gpu(self):
        instance=object.__new__(queue.StageQueue)
        instance.stop_event=__import__("threading").Event()
        instance.tasks=queue.task_plan()
        instance.state={"preflight":{"status":"blocked_external_preflight_failed"}}
        finished=[];instance._wait_external_preflight=lambda:None
        instance._finish=lambda status:finished.append(status)
        def forbidden(*_a,**_k):raise AssertionError("GPU task must not start before valid preflight")
        instance._execute_gpu=forbidden
        self.assertEqual(instance.run(),1)
        self.assertEqual(finished,["blocked_external_preflight_failed"])

    def test_score_terminal_schema_requires_all_three_prediction_roles(self):
        task=next(x for x in queue.task_plan() if x["stage"]=="score")
        temp=queue.HERE/"results_archive"/f"test_score_{os.getpid()}_{uuid.uuid4().hex}"
        temp.mkdir(parents=True)
        task["output"]=str(temp/"score.json")
        try:
            report={"status":"complete","experiment":queue.NAME,"seed_scope":[1],
                "ground_truth_opened_after_all_predictions_validated":True,
                "ground_truth_used_for_prediction":False,"ground_truth_used_for_scoring":True,
                "scores":{"equivariance":{},"mass_control":{},"source_p":{},"source97_qcc":{},"slot32_seed1":{}}}
            Path(task["output"]).write_text(json.dumps(report),encoding="utf-8")
            self.assertTrue(queue.valid_result(task))
            del report["scores"]["source_p"]
            Path(task["output"]).write_text(json.dumps(report),encoding="utf-8")
            self.assertFalse(queue.valid_result(task))
        finally:
            Path(task["output"]).unlink(missing_ok=True);temp.rmdir()


if __name__=="__main__":unittest.main()
