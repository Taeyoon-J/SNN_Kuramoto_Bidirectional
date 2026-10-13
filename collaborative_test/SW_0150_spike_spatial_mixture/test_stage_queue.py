"""Focused CPU contracts for the SW0150 main-stage supervisor."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
import uuid
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from collaborative_test.SW_0150_spike_spatial_mixture import stage_queue as queue


class _Lease:
    def __init__(self, path):
        self.stream = Path(path).open("w+b")

    def fileno(self):
        return self.stream.fileno()

    def close(self):
        self.stream.close()


class _Proc:
    pid = 12345

    def poll(self):
        return 0

    def wait(self):
        return 0


class StageQueueTests(unittest.TestCase):
    def _temp_root(self):
        # Keep test artifacts beneath the writable workspace; Windows profile
        # temp directories can have transient OneDrive ACLs in this checkout.
        root = HERE / "results_archive" / ("queue_test_" + uuid.uuid4().hex)
        root.mkdir(parents=True)
        self.addCleanup(self._remove_test_tree, root)
        return root

    @staticmethod
    def _remove_test_tree(root):
        for path in sorted(Path(root).rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        Path(root).rmdir()

    def test_registered_dag_and_child_argv_are_fixed(self):
        tasks = queue.task_plan()
        self.assertEqual(len(tasks), 6)
        by_id = {row["task_id"]: row for row in tasks}
        self.assertEqual(by_id["train_joint"]["depends_on"], ["external_preflight"])
        self.assertEqual(by_id["train_encoder_graph"]["depends_on"], ["external_preflight"])
        self.assertEqual(by_id["predict_source_p"]["depends_on"], [])
        self.assertEqual(by_id["predict_joint"]["depends_on"], ["train_joint"])
        self.assertEqual(by_id["score"]["depends_on"],
                         ["predict_encoder_graph", "predict_joint", "predict_source_p"])
        argv = queue._argv(by_id["train_joint"])
        self.assertIn("--arm", argv)
        self.assertEqual(argv[argv.index("--arm") + 1], "joint")
        self.assertEqual(argv[argv.index("--device") + 1], "cuda:0")
        self.assertEqual(queue._argv(by_id["score"])[-1], "score")

    def test_external_preflight_waits_then_admits_only_valid_report(self):
        with self.subTest("wait then admit"):
            root = self._temp_root()
            report, warm, state = root / "preflight.json", root / "warm.pt", root / "state.json"
            with mock.patch.object(queue.train, "PREFLIGHT", report), \
                 mock.patch.object(queue.train, "WARM_ARTIFACT", warm), \
                 mock.patch.object(queue, "preflight_queue_state_path", return_value=state):
                self.assertEqual(queue._external_preflight_state()[0],
                                 "waiting_for_external_preflight")
                state.write_text(json.dumps({"status": "running"}), encoding="utf-8")
                self.assertEqual(queue._external_preflight_state()[0],
                                 "waiting_for_external_preflight")
                report.write_text("{}", encoding="utf-8")
                warm.write_bytes(b"warm")
                with mock.patch.object(queue.train, "validate_training_inputs",
                                       return_value={"report_sha256": "r", "warm_sha256": "w"}) as validate:
                    status, admission, error = queue._external_preflight_state()
                self.assertEqual(status, "passed")
                self.assertEqual(admission["warm_sha256"], "w")
                self.assertIsNone(error)
                validate.assert_called_once_with(report, warm)

    def test_terminal_preflight_failure_or_incomplete_success_blocks(self):
        with self.subTest("terminal state blocks"):
            root = self._temp_root()
            report, warm, state = root / "preflight.json", root / "warm.pt", root / "state.json"
            with mock.patch.object(queue.train, "PREFLIGHT", report), \
                 mock.patch.object(queue.train, "WARM_ARTIFACT", warm), \
                 mock.patch.object(queue, "preflight_queue_state_path", return_value=state):
                state.write_text(json.dumps({"status": "preflight_failed"}), encoding="utf-8")
                self.assertEqual(queue._external_preflight_state()[0],
                                 "blocked_external_preflight_failed")
                state.write_text(json.dumps({"status": "preflight_complete"}), encoding="utf-8")
                self.assertEqual(queue._external_preflight_state()[0],
                                 "blocked_invalid_external_preflight")
                state.write_text("{", encoding="utf-8")
                self.assertEqual(queue._external_preflight_state()[0],
                                 "waiting_for_external_preflight")

    def test_predict_child_uses_reserved_physical_gpu_and_validates_result(self):
        with self.subTest("physical GPU mapping"):
            root = self._temp_root()
            runtime, state_path = root, root / "state.json"
            task = dict(next(row for row in queue.task_plan() if row["task_id"] == "predict_source_p"),
                        output=str(root / "prediction"))
            worker = object.__new__(queue.StageQueue)
            worker.stop_event = threading.Event()
            worker.state_lock = threading.RLock()
            worker.state = {"tasks": {task["task_id"]: {"task": task, "status": "queued"}}}
            with mock.patch.object(queue, "RUNTIME", runtime), \
                 mock.patch.object(queue, "STATE", state_path), \
                 mock.patch.object(queue, "fcntl", SimpleNamespace(LOCK_UN=8, flock=lambda *args: None)), \
                 mock.patch.object(queue, "_required_resume_gpu", return_value=None), \
                 mock.patch.object(queue.owner_queue, "_acquire",
                                   return_value=(2, "GPU-2", 12, _Lease(root / "gpu.lock"))), \
                 mock.patch.object(queue.owner_queue, "_stable_idle", return_value=True), \
                 mock.patch.object(queue, "valid_result", side_effect=[False, True]), \
                 mock.patch.object(queue.subprocess, "Popen", return_value=_Proc()) as popen:
                self.assertTrue(worker._execute_gpu(task))
            args, kwargs = popen.call_args
            self.assertEqual(args[0], queue._argv(task, gpu=2))
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "2")
            self.assertEqual(kwargs["env"]["OMP_NUM_THREADS"], "1")
            self.assertTrue(kwargs["start_new_session"])
            self.assertIn("passed", state_path.read_text(encoding="utf-8"))

    def test_cpu_score_child_hides_cuda_and_requires_frozen_predictions(self):
        with self.subTest("CPU score"):
            root = self._temp_root()
            runtime, state_path = root, root / "state.json"
            task = next(row for row in queue.task_plan() if row["task_id"] == "score")
            worker = object.__new__(queue.StageQueue)
            worker.state_lock = threading.RLock()
            worker.state = {"tasks": {"score": {"task": task, "status": "queued"}}}
            with mock.patch.object(queue, "RUNTIME", runtime), \
                 mock.patch.object(queue, "STATE", state_path), \
                 mock.patch.object(queue, "fcntl", SimpleNamespace(LOCK_UN=8, flock=lambda *args: None)), \
                 mock.patch.object(queue, "valid_result", side_effect=[False, True]), \
                 mock.patch.object(queue.subprocess, "Popen", return_value=_Proc()) as popen:
                self.assertTrue(worker._execute_cpu(task))
            kwargs = popen.call_args.kwargs
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
            self.assertTrue(worker.state["tasks"]["score"]["cuda_hidden"])

    def test_existing_outputs_are_never_relaunched(self):
        with self.subTest("preserve existing output"):
            root = self._temp_root()
            task = next(row for row in queue.task_plan() if row["task_id"] == "predict_source_p")
            output = root / "already-there"
            output.mkdir()
            task = dict(task, output=str(output))
            worker = object.__new__(queue.StageQueue)
            worker.stop_event = threading.Event()
            worker.state_lock = threading.RLock()
            worker.state = {"tasks": {task["task_id"]: {"task": task, "status": "queued"}}}
            with mock.patch.object(queue, "RUNTIME", root), \
                 mock.patch.object(queue, "STATE", root / "state.json"), \
                 mock.patch.object(queue, "valid_result", return_value=False), \
                 mock.patch.object(queue.subprocess, "Popen") as popen:
                self.assertFalse(worker._execute_gpu(task))
            popen.assert_not_called()
            self.assertEqual(worker.state["tasks"][task["task_id"]]["status"],
                             "failed_existing_output_preserved")

    def test_terminal_external_preflight_cancels_only_pending_source_and_never_trains(self):
        class _Lock:
            def fileno(self):
                return 7

            def close(self):
                pass

        with self.subTest("preflight terminal blocks training"):
            root = self._temp_root()
            worker = object.__new__(queue.StageQueue)
            worker.stop_event = threading.Event()
            worker.state_lock = threading.RLock()
            worker.tasks = queue.task_plan()
            worker.lock_stream = _Lock()
            worker.state = {
                "status": "running",
                "preflight": {"status": "blocked_external_preflight_failed"},
                "tasks": {row["task_id"]: {"task": row, "status": "queued"}
                          for row in worker.tasks},
            }
            started = []

            def execute(task):
                started.append(task["task_id"])
                return True

            with mock.patch.object(queue, "RUNTIME", root), \
                 mock.patch.object(queue, "STATE", root / "state.json"), \
                 mock.patch.object(queue, "fcntl", SimpleNamespace(LOCK_UN=8, flock=lambda *args: None)), \
                 mock.patch.object(worker, "_execute_gpu", side_effect=execute), \
                 mock.patch.object(worker, "_wait_external_preflight", return_value=None), \
                 mock.patch.object(worker, "_execute_cpu") as cpu:
                rc = worker.run()
            self.assertEqual(rc, 1)
            self.assertEqual(started, ["predict_source_p"])
            self.assertFalse(cpu.called)
            self.assertEqual(worker.state["status"], "blocked_external_preflight_failed")

    def test_missing_candidate_arm_never_reaches_score(self):
        class _Lock:
            def fileno(self):
                return 7

            def close(self):
                pass

        worker = object.__new__(queue.StageQueue)
        worker.stop_event = threading.Event()
        worker.state_lock = threading.RLock()
        worker.tasks = queue.task_plan()
        worker.lock_stream = _Lock()
        worker.state = {
            "status": "running", "preflight": {"status": "passed"},
            "tasks": {row["task_id"]: {"task": row, "status": "queued"}
                      for row in worker.tasks},
        }
        def execute(task):
            if task["stage"] == "train" and task["arm"] == "joint":
                return False
            return True

        root = self._temp_root()
        with mock.patch.object(queue, "RUNTIME", root), \
             mock.patch.object(queue, "STATE", root / "state.json"), \
             mock.patch.object(queue, "fcntl", SimpleNamespace(LOCK_UN=8, flock=lambda *args: None)), \
             mock.patch.object(worker, "_execute_gpu", side_effect=execute), \
             mock.patch.object(worker, "_wait_external_preflight", return_value={"admitted": True}), \
             mock.patch.object(queue, "valid_result", return_value=True), \
             mock.patch.object(worker, "_execute_cpu") as cpu:
            rc = worker.run()
        self.assertEqual(rc, 1)
        self.assertFalse(cpu.called)
        self.assertEqual(worker.state["status"], "score_blocked_missing_valid_predictions")


if __name__ == "__main__":
    unittest.main()
