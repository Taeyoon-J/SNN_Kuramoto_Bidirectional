from __future__ import annotations

from pathlib import Path
import copy
import tempfile
import unittest
import uuid
import threading
from unittest import mock

import torch
from torch import nn

from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4 as queue
from collaborative_test.SW_0148_checkpointable_boundary_q.checkpoint import (
    runtime_signature, save_boundary,
)


class ResumeGpuAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.gettempdir()) / f"sw0148-queue-v2-{uuid.uuid4().hex}"
        self.root.mkdir()

    def tearDown(self):
        if self.root.parent.resolve() != Path(tempfile.gettempdir()).resolve():
            raise AssertionError("refusing cleanup outside the owned fixture")
        for child in sorted(self.root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        self.root.rmdir()

    @staticmethod
    def _task():
        return {"task_id": "train_joint", "stage": "train", "arm": "joint"}

    def _write_boundary(self, visibility, *, device_type="cuda", device_index=0,
                        completed_updates=0):
        wrapped, encoder = nn.Linear(2, 2), nn.Linear(2, 2)
        optimizer = torch.optim.Adam(list(wrapped.parameters()) + list(encoder.parameters()))
        runtime = runtime_signature("cpu")
        runtime["cuda_visible_devices"] = visibility
        # Synthetic metadata only: this fixture never initializes CUDA.
        runtime["device_type"] = device_type
        runtime["device_index"] = device_index
        runtime["device_name"] = "fixture-cuda"
        runtime["device_capability"] = [8, 0]
        runtime["device_total_memory"] = 16
        checkpoint_dir = self.root / "joint_seed1" / "checkpoints"
        save_boundary(checkpoint_dir, completed_updates=completed_updates, wrapped=wrapped, encoder=encoder,
                      optimizer=optimizer, history=[], bindings={"lineage": "fixed"},
                      runtime=runtime, device="cpu")
        return checkpoint_dir

    def test_committed_checkpoint_requires_its_exact_physical_gpu(self):
        self._write_boundary("2")
        with mock.patch.object(queue.trainer, "WORK", self.root):
            self.assertEqual(queue.required_resume_gpu(self._task()), 2)

    def test_no_commit_may_choose_any_gpu_but_unknown_work_is_rejected(self):
        with mock.patch.object(queue.trainer, "WORK", self.root):
            self.assertIsNone(queue.required_resume_gpu(self._task()))
            checkpoint_dir = self.root / "joint_seed1" / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            self.assertIsNone(queue.required_resume_gpu(self._task()))
            (checkpoint_dir / ".boundary.ptx.interrupted.tmp").write_bytes(b"partial")
            self.assertIsNone(queue.required_resume_gpu(self._task()))
            (checkpoint_dir / "unrecognized.bin").write_bytes(b"preserve")
            with self.assertRaisesRegex(ValueError, "known atomic temp"):
                queue.required_resume_gpu(self._task())

    def test_corrupt_or_unbound_checkpoint_fails_closed(self):
        checkpoint_dir = self._write_boundary("cuda:0")
        with mock.patch.object(queue.trainer, "WORK", self.root):
            with self.assertRaisesRegex(ValueError, "single-physical-GPU"):
                queue.required_resume_gpu(self._task())
        (checkpoint_dir / "boundary.ptx").unlink()
        checkpoint_dir.rmdir()
        checkpoint_dir.parent.rmdir()
        checkpoint_dir = self._write_boundary("1")
        (checkpoint_dir / "boundary.ptx").write_bytes(b"truncated")
        with mock.patch.object(queue.trainer, "WORK", self.root):
            with self.assertRaises((OSError, ValueError, RuntimeError)):
                queue.required_resume_gpu(self._task())

    def test_resume_metadata_rejects_cpu_bad_device_index_and_bad_update_counts(self):
        checkpoint_dir = self._write_boundary("2")
        from collaborative_test.SW_0148_checkpointable_boundary_q.checkpoint import inspect_committed_metadata
        real_metadata = inspect_committed_metadata(checkpoint_dir)
        cases = [
            ({"runtime": {"device_type": "cpu", "device_index": 0,
                          "cuda_visible_devices": "2"}}, "runtime"),
            ({"runtime": {"device_type": "cuda", "device_index": 1,
                          "cuda_visible_devices": "2"}}, "runtime"),
            ({"completed_updates": -1}, "updates"),
            ({"completed_updates": queue.sw146.UPDATES + 1}, "updates"),
            ({"completed_updates": True}, "updates"),
        ]
        with mock.patch.object(queue.trainer, "WORK", self.root):
            for changes, _label in cases:
                metadata = copy.deepcopy(real_metadata)
                metadata.update(changes)
                with self.subTest(changes=changes), \
                     mock.patch("collaborative_test.SW_0148_checkpointable_boundary_q.checkpoint.inspect_committed_metadata",
                                return_value=metadata):
                    with self.assertRaisesRegex(ValueError, "single-physical-GPU"):
                        queue.required_resume_gpu(self._task())

    def test_acquire_pins_to_gpu3_until_lease_is_released(self):
        self._write_boundary("3")
        with mock.patch.object(queue.trainer, "WORK", self.root):
            preferred = queue.required_resume_gpu(self._task())
        self.assertEqual(preferred, 3)

        class FlockStub:
            LOCK_EX, LOCK_NB, LOCK_UN = 1, 2, 8
            @staticmethod
            def flock(_fd, _operation):
                return None

        rows = [("GPU-3", 900, [765]), ("GPU-3", 100, [])]
        queried = []
        def gpu_info(gpu):
            queried.append(gpu)
            return rows.pop(0)

        with mock.patch.object(queue, "GPU_LEASE_DIR", self.root / "leases"), \
             mock.patch.object(queue, "fcntl", FlockStub), \
             mock.patch.object(queue.gpu_guard, "_gpu_info", side_effect=gpu_info):
            gpu, gpu_uuid, memory, lease = queue._acquire(threading.Event(), preferred)
            self.assertEqual((gpu, gpu_uuid, memory), (3, "GPU-3", 100))
            self.assertEqual(queried, [3, 3])
            self.assertEqual(sorted(p.name for p in (self.root / "leases").iterdir()), ["gpu3.lock"])
            queue.fcntl.flock(lease.fileno(), queue.fcntl.LOCK_UN)
            lease.close()

    def test_pinned_busy_gpu_wait_can_be_cancelled_without_fallback(self):
        class FlockStub:
            LOCK_EX, LOCK_NB, LOCK_UN = 1, 2, 8
            @staticmethod
            def flock(_fd, _operation):
                return None

        stop = threading.Event()
        queried = []
        def busy(gpu):
            queried.append(gpu)
            return "GPU-3", 700, [765]

        def cancel(_seconds):
            stop.set()

        with mock.patch.object(queue, "GPU_LEASE_DIR", self.root / "leases"), \
             mock.patch.object(queue, "fcntl", FlockStub), \
             mock.patch.object(queue.gpu_guard, "_gpu_info", side_effect=busy), \
             mock.patch.object(queue.time, "sleep", side_effect=cancel):
            with self.assertRaisesRegex(RuntimeError, "stopped while waiting"):
                queue._acquire(stop, preferred_gpu=3)
        self.assertEqual(queried, [3])

    def test_short_idle_window_uses_two_observations_and_final_recheck(self):
        rows = [("GPU-3", 100, []), ("GPU-3", 120, []), ("GPU-3", 130, [])]
        def gpu_info(gpu):
            self.assertEqual(gpu, 3)
            return rows.pop(0)
        samples = []
        with mock.patch.object(queue.gpu_guard, "_gpu_info", side_effect=gpu_info), \
             mock.patch.object(queue.time, "monotonic", side_effect=[0.0, 5.0]), \
             mock.patch.object(queue.time, "sleep") as sleep:
            self.assertTrue(queue._stable_idle(
                3, "GPU-3", threading.Event(), lambda elapsed, memory: samples.append((elapsed, memory))))
        self.assertEqual(samples, [(0.0, 100), (5.0, 120)])
        sleep.assert_called_once_with(5.0)
        self.assertEqual(rows, [])

    def test_short_idle_window_rejects_owner_at_second_observation(self):
        rows = [("GPU-3", 100, []), ("GPU-3", 100, [765])]
        with mock.patch.object(queue.gpu_guard, "_gpu_info", side_effect=lambda _gpu: rows.pop(0)), \
             mock.patch.object(queue.time, "monotonic", side_effect=[0.0, 5.0]), \
             mock.patch.object(queue.time, "sleep"):
            self.assertFalse(queue._stable_idle(3, "GPU-3", threading.Event(), lambda *_: None))
        self.assertEqual(rows, [])

    def test_execute_gpu_forwards_checkpoint_affinity(self):
        self._write_boundary("3")
        task = dict(self._task(), output=str(self.root / "not-produced"))
        instance = object.__new__(queue.Queue)
        instance.stop_event = threading.Event()
        instance.state = {"tasks": {task["task_id"]: {"status": "queued"}}}
        instance._update = lambda task_id, **fields: instance.state["tasks"][task_id].update(fields)
        forwarded = []
        def stop_after_admission(_event, preferred_gpu=None):
            forwarded.append(preferred_gpu)
            raise RuntimeError("fixture stops after real affinity admission")
        with mock.patch.object(queue.trainer, "WORK", self.root), \
             mock.patch.object(queue, "valid_result", return_value=False), \
             mock.patch.object(queue, "predecessor_clear", return_value=True), \
             mock.patch.object(queue, "_acquire", side_effect=stop_after_admission):
            with self.assertRaisesRegex(RuntimeError, "real affinity admission"):
                instance._execute_gpu(task)
        self.assertEqual(forwarded, [3])

    def test_corrupt_checkpoint_fails_before_gpu_reservation(self):
        checkpoint_dir = self._write_boundary("1")
        (checkpoint_dir / "boundary.ptx").write_bytes(b"broken")
        task = dict(self._task(), output=str(self.root / "not-produced"))
        instance = object.__new__(queue.Queue)
        instance.stop_event = threading.Event()
        instance.state = {"tasks": {task["task_id"]: {"status": "queued"}}}
        instance._update = lambda task_id, **fields: instance.state["tasks"][task_id].update(fields)
        with mock.patch.object(queue.trainer, "WORK", self.root), \
             mock.patch.object(queue, "valid_result", return_value=False), \
             mock.patch.object(queue, "predecessor_clear", return_value=True), \
             mock.patch.object(queue, "_acquire", side_effect=AssertionError("must fail before lease")):
            self.assertFalse(instance._execute_gpu(task))
        self.assertEqual(instance.state["tasks"][task["task_id"]]["status"],
                         "failed_checkpoint_admission")

    def test_completed_training_is_reused_without_gpu_admission(self):
        instance = object.__new__(queue.Queue)
        instance.stop_event = threading.Event()
        instance.state = {"tasks": {"train_joint": {"status": "queued"}}}
        instance._update = lambda task_id, **fields: instance.state["tasks"][task_id].update(fields)
        with mock.patch.object(queue, "predecessor_clear", return_value=True), \
             mock.patch.object(queue, "valid_result", return_value=True), \
             mock.patch.object(queue, "_acquire", side_effect=AssertionError("must not reserve GPU")):
            self.assertTrue(instance._execute_gpu(self._task()))
        self.assertEqual(instance.state["tasks"]["train_joint"]["status"], "verified_existing")

    def test_partial_output_waits_for_predecessor_then_adopts_valid_result(self):
        task = dict(self._task(), output=str(self.root / "joint_seed1"))
        Path(task["output"]).mkdir()
        instance = object.__new__(queue.Queue)
        instance.stop_event = threading.Event()
        instance.state = {"tasks": {task["task_id"]: {"status": "queued"}}}
        instance._update = lambda task_id, **fields: instance.state["tasks"][task_id].update(fields)
        clear = iter([False, True])
        with mock.patch.object(queue, "predecessor_clear", side_effect=lambda _arm: next(clear)), \
             mock.patch.object(queue.time, "sleep"), \
             mock.patch.object(queue, "valid_result", return_value=True), \
             mock.patch.object(queue, "_acquire", side_effect=AssertionError("must adopt after handoff")):
            self.assertTrue(instance._execute_gpu(task))
        self.assertEqual(instance.state["tasks"][task["task_id"]]["status"], "verified_existing")

    def test_train_argv_routes_through_resume_adapter_but_predictions_stay_frozen(self):
        train_argv = queue._argv({"stage": "train", "arm": "joint"})
        predict_argv = queue._argv({"stage": "predict", "arm": "joint"})
        self.assertEqual(train_argv[2],
                         "collaborative_test.SW_0148_checkpointable_boundary_q.recovery_v2")
        self.assertEqual(train_argv[3:], ["--arm", "joint", "--device", "cuda:0"])
        self.assertEqual(predict_argv[2], "collaborative_test.SW_0148_checkpointable_boundary_q.run")
        self.assertIn("--stage", predict_argv)
        self.assertIn("predict", predict_argv)

    def test_joint_handoff_requires_old_joint_pid_and_group_absent(self):
        self.assertTrue(queue.predecessor_clear(
            "joint", proc_reader=lambda _pid: None, group_reader=lambda _pgid: []))

        def reused(pid):
            return {"pid": pid, "state": "R", "start_ticks": 5, "pgid": 769533,
                    "argv": ["foreign-job"]}
        with self.assertRaisesRegex(RuntimeError, "reused or command changed"):
            queue.predecessor_clear("joint", proc_reader=reused, group_reader=lambda _pgid: [])

    def test_encoder_handoff_waits_for_live_predecessor_group_descendants(self):
        captured = [queue.HANDOFF_IDENTITIES["supervisor"],
                    queue.HANDOFF_IDENTITIES["encoder_graph"]]
        rows = {item["pid"]: {**item, "state": "S"} for item in captured}
        self.assertFalse(queue.predecessor_clear(
            "encoder_graph", proc_reader=lambda pid: rows.get(pid),
            group_reader=lambda pgid: [row for row in rows.values() if row["pgid"] == pgid]))
        alien = {"pid": 999999, "state": "R", "start_ticks": 1,
                 "pgid": captured[1]["pgid"], "argv": ["unrelated"]}
        # A short-lived helper such as nvidia-smi can be a process-group
        # descendant without appearing in the captured PID list. It blocks
        # handoff until gone; it is never treated as a predecessor PID identity.
        self.assertFalse(queue.predecessor_clear(
            "encoder_graph", proc_reader=lambda pid: rows.get(pid),
            group_reader=lambda pgid: ([alien] if pgid == alien["pgid"]
                                       else [row for row in rows.values() if row["pgid"] == pgid])))

    def test_permission_failure_for_captured_pid_never_means_missing(self):
        def inaccessible(_pid):
            raise PermissionError("fixture")
        with self.assertRaisesRegex(RuntimeError, "cannot verify predecessor PID"):
            queue.predecessor_clear("encoder_graph", proc_reader=inaccessible,
                                    group_reader=lambda _pgid: [])

    def test_encoder_handoff_loop_waits_for_supervisor_and_writer(self):
        instance = object.__new__(queue.Queue)
        instance.stop_event = threading.Event()
        instance.state = {"tasks": {"train_encoder_graph": {"status": "queued"}}}
        updates = []
        instance._update = lambda task_id, **fields: updates.append(fields)
        answers = iter([False, True])
        sleeps = []
        with mock.patch.object(queue, "predecessor_clear", side_effect=lambda _arm: next(answers)), \
             mock.patch.object(queue.time, "sleep", side_effect=lambda seconds: sleeps.append(seconds)):
            self.assertTrue(instance._wait_for_training_handoff(
                {"task_id": "train_encoder_graph", "stage": "train", "arm": "encoder_graph"}))
        self.assertEqual(sleeps, [queue.SAMPLE_SECONDS])
        self.assertEqual([row["status"] for row in updates],
                         ["waiting_for_predecessor_handoff", "predecessor_handoff_clear"])
        self.assertTrue(all(row["predecessor_manifest_sha256"] == queue._handoff_manifest_sha()
                            for row in updates))


if __name__ == "__main__":
    unittest.main()
