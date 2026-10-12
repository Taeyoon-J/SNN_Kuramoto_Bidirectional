from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
import uuid
from pathlib import Path
from unittest import mock

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "collaborative_test")]

from collaborative_test.SW_0143_native32_selective_geodesic import geodesic_queue as queue
from collaborative_test.SW_0143_native32_selective_geodesic import run


class SelectiveGeodesicQueueTests(unittest.TestCase):
    def test_registered_dag_and_independent_reference_then_interventions(self):
        tasks = queue.task_plan()
        self.assertEqual(len(tasks), 13)
        by_id = {row["task_id"]: row for row in tasks}
        for seed in run.SEEDS:
            self.assertEqual(by_id[f"reference_R_s{seed}"]["deps"], [f"preflight_s{seed}"])
            for arm in ("C", "T"):
                self.assertEqual(by_id[f"predict_{arm}_s{seed}"]["deps"],
                                 [f"reference_R_s{seed}"])
        self.assertEqual(len(by_id["score_all"]["deps"]), 9)
        self.assertEqual(queue.task_groups()[-1], ("score_all",))

    def test_score_is_cuda_hidden_and_gpu_tasks_map_registered_physical_device(self):
        score = next(row for row in queue.task_plan() if row["stage"] == "score")
        self.assertEqual(queue.environment(score)["CUDA_VISIBLE_DEVICES"], "")
        task = next(row for row in queue.task_plan() if row["stage"] == "preflight")
        with self.assertRaises(ValueError):
            queue.environment(task, physical_gpu=None)
        self.assertEqual(queue.environment(task, physical_gpu=3)["CUDA_VISIBLE_DEVICES"], "3")

    def test_gpu_admission_requires_empty_memory_and_no_compute_owners(self):
        self.assertTrue(queue.gpu_candidate(512, []))
        self.assertFalse(queue.gpu_candidate(513, []))
        self.assertFalse(queue.gpu_candidate(1, [1234]))

    def test_preflight_validator_accepts_emitted_schema_not_crossbatch_equality(self):
        temp = Path(tempfile.gettempdir()) / ("sw0143_queue_" + uuid.uuid4().hex[:10])
        temp.mkdir()
        path = temp / "seed0.json"
        original = queue.run._preflight_path
        try:
            row = {
                "status": "preflight_complete", "seed": 0,
                "implementation": run.implementation_fingerprint(),
                "ground_truth_used": False, "optimizer_updates": 0,
                "source_r_native_trace_equal": {
                    batch: {key: True for key in run.TRACE_FIELDS} for batch in ("B1", "B4")},
                "native_b1_b4_trace_max_abs_descriptive": {key: 0.25 for key in run.TRACE_FIELDS},
                "r_b1_b4_trace_max_abs_descriptive": {key: 0.5 for key in run.TRACE_FIELDS},
                "intervention_b4_trace_shapes": {arm: {} for arm in ("C", "T")},
                "cuda_peak_allocated_bytes": 1,
            }
            path.write_text(json.dumps(row), encoding="utf-8")
            queue.run._preflight_path = lambda seed: path
            self.assertTrue(queue.validate_result({"stage": "preflight", "seed": 0}))
            row["source_r_native_trace_equal"]["B1"]["theta"] = False
            path.write_text(json.dumps(row), encoding="utf-8")
            self.assertFalse(queue.validate_result({"stage": "preflight", "seed": 0}))
        finally:
            queue.run._preflight_path = original
            path.unlink(missing_ok=True)
            temp.rmdir()

    def _exercise_queue(self, fail_task=None):
        temp = Path(tempfile.gettempdir()) / ("sw0143_run_" + uuid.uuid4().hex[:10])
        archive = temp / "archive"
        archive.mkdir(parents=True)
        state_path, lock_path = archive / "state.json", archive / "queue.lock"
        original_output, original_log, original_fcntl = queue.output_path, queue.log_path, queue.fcntl
        passed = set()
        passed_lock = threading.Lock()
        calls = []

        class FakeFcntl:
            LOCK_EX = 1
            LOCK_NB = 2
            LOCK_UN = 3

            @staticmethod
            def flock(_fd, _operation):
                return None

        def fake_output(task):
            return archive / "artifacts" / (task["task_id"] + ".json")

        def fake_log(task):
            return archive / "logs" / (task["task_id"] + ".log")

        def fake_execute(instance, task):
            calls.append(task["task_id"])
            okay = task["task_id"] != fail_task
            if okay:
                with passed_lock:
                    passed.add(task["task_id"])
            return okay

        def fake_validate(task):
            with passed_lock:
                return task["task_id"] in passed

        try:
            with mock.patch.object(queue, "ARCHIVE", archive), \
                 mock.patch.object(queue, "STATE", state_path), \
                 mock.patch.object(queue, "LOCK", lock_path), \
                 mock.patch.object(queue, "output_path", side_effect=fake_output), \
                 mock.patch.object(queue, "log_path", side_effect=fake_log), \
                 mock.patch.object(queue, "validate_result", side_effect=fake_validate), \
                 mock.patch.object(queue, "fcntl", FakeFcntl):
                with mock.patch.object(queue.SelectiveGeodesicQueue, "_execute", new=fake_execute):
                    instance = queue.SelectiveGeodesicQueue()
                    result = instance.run()
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                    instance.queue_lock.close()
            return result, state, calls
        finally:
            queue.output_path, queue.log_path, queue.fcntl = original_output, original_log, original_fcntl
            for path in (state_path, lock_path):
                path.unlink(missing_ok=True)
            archive.rmdir()
            temp.rmdir()

    def test_constructor_and_run_execute_all_13_tasks_then_score(self):
        result, state, calls = self._exercise_queue()
        self.assertEqual(result, 0)
        self.assertEqual(state["status"], "complete")
        self.assertEqual(len(calls), 13)
        self.assertEqual(calls[-1], "score_all")

    def test_failed_intervention_stops_before_score(self):
        result, state, calls = self._exercise_queue("predict_C_s1")
        self.assertEqual(result, 1)
        self.assertEqual(state["status"], "failed")
        self.assertNotIn("score_all", calls)


if __name__ == "__main__":
    unittest.main()
