import os
import json
import subprocess
import sys
import shutil
import uuid
import unittest
from pathlib import Path
from unittest import mock

from collaborative_test.SW_0158_membrane_spike_credit import prediction_queue as pq, train
from collaborative_test.SW_0158_membrane_spike_credit import score_queue as sq


class PredictionQueueContractTests(unittest.TestCase):
    def scratch(self):
        path = pq.ROOT.parent / f"unit_queue_{uuid.uuid4().hex}"
        path.mkdir(parents=False)
        self.addCleanup(shutil.rmtree, path, ignore_errors=True)
        return path
    def test_both_registered_arms_have_distinct_outputs_and_fixed_child_argv(self):
        live = pq.configure_arm("live_membrane_credit")
        control = pq.configure_arm("detached_membrane_control")
        self.assertNotEqual(live["output"], control["output"])
        self.assertEqual(pq.command("live_membrane_credit"), [sys.executable, "-m",
            "collaborative_test.SW_0158_membrane_spike_credit.predict_evaluate",
            "--stage", "predict", "--arm", "live_membrane_credit", "--device", "cuda:0"])
        self.assertEqual(pq.command("detached_membrane_control")[6], "detached_membrane_control")
        with self.assertRaises(ValueError):
            pq.configure_arm("unregistered")

    def test_final_gpu_guard_rejects_changed_uuid_owner_memory_and_query_error(self):
        with mock.patch.object(pq.owner.gpu_guard, "_gpu_info", return_value=("uuid-a", 4, [])):
            self.assertTrue(pq._final_gpu_admission(1, "uuid-a")[0])
        cases = [(("uuid-b", 4, []), "uuid_changed"),
                 (("uuid-a", 5, [77]), "owners"),
                 (("uuid-a", 513, []), "memory")]
        for result, reason in cases:
            with mock.patch.object(pq.owner.gpu_guard, "_gpu_info", return_value=result):
                admitted, detail = pq._final_gpu_admission(1, "uuid-a")
                self.assertFalse(admitted)
                self.assertIn(reason, detail["reason"])
        with mock.patch.object(pq.owner.gpu_guard, "_gpu_info", side_effect=RuntimeError("query")):
            self.assertFalse(pq._final_gpu_admission(1, "uuid-a")[0])

    def test_child_import_preserves_queue_assigned_physical_visibility(self):
        script = ("import os,sys; sys.path.insert(0, 'collaborative_test'); "
                  "import collaborative_test.SW_0158_membrane_spike_credit.prediction_queue; "
                  "assert os.environ.get('CUDA_VISIBLE_DEVICES') == '3'; "
                  "import torch; assert not torch.cuda.is_initialized()")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="3", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1")
        result = subprocess.run([sys.executable, "-c", script], cwd=pq.ROOT,
            env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_score_coordinator_library_import_preserves_assigned_visibility(self):
        script = ("import os,sys; sys.path.insert(0, 'collaborative_test'); "
                  "import collaborative_test.SW_0158_membrane_spike_credit.score_queue; "
                  "assert os.environ.get('CUDA_VISIBLE_DEVICES') == '3'; "
                  "import torch; assert not torch.cuda.is_initialized()")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="3", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1")
        result = subprocess.run([sys.executable, "-c", script], cwd=pq.ROOT,
            env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_fingerprint_resolves_prediction_and_training_dependencies(self):
        fingerprint = pq.implementation_fingerprint()
        self.assertIn("collaborative_test/SW_0158_membrane_spike_credit/prediction_queue.py", fingerprint)
        self.assertIn("collaborative_test/SW_0158_membrane_spike_credit/predict_evaluate.py", fingerprint)
        self.assertIn("collaborative_test/SW_0148_checkpointable_boundary_q/stage_queue_v4.py", fingerprint)
        self.assertTrue(all(len(digest) == 64 for digest in fingerprint.values()))

    def test_training_artifact_wait_requires_a_live_matching_arm_owner(self):
        q = object.__new__(pq.PredictionQueue)
        q.arm = "live_membrane_credit"
        q.stop_event = __import__("threading").Event()
        q.existing_complete = False
        directory = self.scratch()
        with mock.patch.object(pq, "TRAINING_QUEUE_ROOT", directory), \
             mock.patch.object(pq, "_process_has_args", return_value=True) as process:
            arm_state = directory / q.arm / "state.json"
            arm_state.parent.mkdir()
            arm_state.write_text(json.dumps({"status": "waiting_for_upstream",
                "experiment": train.EXPERIMENT, "task": {"arm": q.arm},
                "supervisor_pid": 123}), encoding="utf-8")
            self.assertTrue(pq._training_queue_is_live(q.arm))
            process.assert_any_call(123,
                "collaborative_test.SW_0158_membrane_spike_credit.training_queue", q.arm)
            self.assertEqual(process.call_count, 2)
            process.reset_mock(return_value=True)
            self.assertFalse(pq._training_queue_is_live("detached_membrane_control"))

    def test_missing_or_present_corrupt_training_artifact_never_reserves_gpu(self):
        q = object.__new__(pq.PredictionQueue)
        q.arm = "live_membrane_credit"
        q.stop_event = __import__("threading").Event()
        q.existing_complete = False
        q.state_lock = __import__("threading").RLock()
        directory = self.scratch()
        q.state_path = directory / "queue-state.json"
        q.state_path.write_text(json.dumps({"status": "waiting_for_training_artifact"}),
                                encoding="utf-8")
        q._training_ready = lambda: (_ for _ in ()).throw(FileNotFoundError("missing"))
        q._unlock = lambda: None
        with mock.patch.object(pq, "_training_queue_is_live", return_value=False), \
             mock.patch.object(pq.owner, "_acquire") as acquire:
            with self.assertRaisesRegex(RuntimeError, "no live owner"):
                q.run()
            acquire.assert_not_called()

        manifest = directory / "live_membrane_credit_seed1" / "manifest.json"
        manifest.parent.mkdir()
        manifest.write_text("{broken", encoding="utf-8")
        q._training_ready = lambda: pq.evaluation.validate_training_artifact(
            q.arm, directory=manifest.parent)
        with mock.patch.object(pq.train, "TRAIN_ROOT", directory), \
             mock.patch.object(pq, "_training_queue_is_live", return_value=True), \
             mock.patch.object(pq.owner, "_acquire") as acquire:
            with self.assertRaises(json.JSONDecodeError):
                q.run()
            acquire.assert_not_called()

    def test_score_wait_requires_actual_live_prediction_queue_owner(self):
        root = self.scratch()
        state = root / "live_membrane_credit" / "state.json"
        state.parent.mkdir()
        state.write_text(json.dumps({"experiment": train.EXPERIMENT,
            "arm": "live_membrane_credit", "status": "running",
            "supervisor_pid": 4321}), encoding="utf-8")
        with mock.patch.object(sq, "PREDICTION_QUEUE_ROOT", root), \
             mock.patch.object(sq, "_process_has_args", return_value=True) as process:
            self.assertTrue(sq._prediction_queue_is_live("live_membrane_credit"))
            process.assert_any_call(4321,
                "collaborative_test.SW_0158_membrane_spike_credit.prediction_queue",
                "live_membrane_credit")
            self.assertEqual(process.call_count, 2)
            process.reset_mock(return_value=True)
            self.assertFalse(sq._prediction_queue_is_live("detached_membrane_control"))


if __name__ == "__main__":
    unittest.main()
