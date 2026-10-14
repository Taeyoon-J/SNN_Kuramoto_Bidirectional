"""CPU contracts for SW0163 prediction/score queue handoff."""
from __future__ import annotations
import json, os, subprocess, sys, tempfile, unittest, shutil, uuid
from pathlib import Path
from unittest import mock
from collaborative_test.SW_0163_relation_context_assignment import prediction_queue as pq, score_queue as sq, train

class QueueHandoffTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(__file__).resolve().parents[2]/("tq163"+uuid.uuid4().hex[:7]); self.root.mkdir()
    def tearDown(self):
        if not self.root.resolve().is_relative_to(Path(__file__).resolve().parents[2]): raise RuntimeError("unsafe cleanup")
        shutil.rmtree(self.root)
    def test_training_dependency_waits_only_on_live_registered_owner(self):
        state=self.root/"state.json"
        state.write_text(json.dumps({"status":"training_arms_queued","supervisor_pid":42}),encoding="utf-8")
        self.assertIsNone(pq.training_dependency(state_path=state,process_live=lambda pid:pid==42))
        with self.assertRaisesRegex(ValueError,"not live"):
            pq.training_dependency(state_path=state,process_live=lambda pid:False)
        state.write_text(json.dumps({"status":"complete","queue_fingerprint":{"fp":"x"},"results":{a:True for a in train.ARMS}}),encoding="utf-8")
        with mock.patch.object(pq.training_queue,"implementation_fingerprint",return_value={"fp":"x"}):
            self.assertEqual(pq.training_dependency(state_path=state,process_live=lambda p:False)["state"]["status"],"complete")
        state.write_text(json.dumps({"status":"failed_training"}),encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"terminal"):
            pq.training_dependency(state_path=state,process_live=lambda p:True)

    def test_score_dependency_requires_all_arm_prediction_registry(self):
        state=self.root/"pred.json"
        state.write_text(json.dumps({"status":"running","supervisor_pid":9}),encoding="utf-8")
        self.assertIsNone(sq.prediction_dependency(state_path=state,process_live=lambda p:True))
        with self.assertRaisesRegex(ValueError,"not live"):
            sq.prediction_dependency(state_path=state,process_live=lambda p:False)
        state.write_text(json.dumps({"status":"complete","queue_fingerprint":{"fp":"y"},"prediction_hashes":{a:{} for a in (*pq.train.ARMS,"source_T_control")}}),encoding="utf-8")
        with mock.patch.object(sq.prediction_queue,"implementation_fingerprint",return_value={"fp":"y"}):
            self.assertEqual(sq.prediction_dependency(state_path=state)["state"]["status"],"complete")
        state.write_text(json.dumps({"status":"complete","queue_fingerprint":{"fp":"y"},"prediction_hashes":{"raw_only":{}}}),encoding="utf-8")
        with mock.patch.object(sq.prediction_queue,"implementation_fingerprint",return_value={"fp":"y"}):
            with self.assertRaisesRegex(ValueError,"wrong arm set"):
                sq.prediction_dependency(state_path=state)

    def test_imports_preserve_assigned_cuda_visibility(self):
        code=("import os,sys; os.environ['CUDA_VISIBLE_DEVICES']='3'; "
              "import collaborative_test.SW_0163_relation_context_assignment.prediction_queue; "
              "assert os.environ['CUDA_VISIBLE_DEVICES']=='3'; "
              "import collaborative_test.SW_0163_relation_context_assignment.score_queue; "
              "assert os.environ['CUDA_VISIBLE_DEVICES']=='3'")
        env=dict(os.environ); env["CUDA_VISIBLE_DEVICES"]="3"; env["OMP_NUM_THREADS"]="1"; env["MKL_NUM_THREADS"]="1"
        result=subprocess.run([sys.executable,"-c",code],cwd=pq.ROOT,env=env,capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_registered_child_command_and_dry_plan(self):
        argv=pq.command()
        self.assertEqual(argv[argv.index("--stage")+1],"predict")
        self.assertIn("collaborative_test.SW_0163_relation_context_assignment.predict_evaluate",argv)
        self.assertEqual(pq.MAX_MEMORY_MIB,512)
        self.assertEqual(pq.DEVICE,"cuda:0")

if __name__=="__main__": unittest.main()
