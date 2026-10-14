import numpy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
import uuid
from types import SimpleNamespace
from unittest import mock

from collaborative_test.SW_0161_supervised_spike_partition import training_queue as queue


class _Lease:
    def __init__(self, path):
        self.stream = Path(path).open("w+b")
        self.closed = False
    def fileno(self):
        return self.stream.fileno()
    def close(self):
        self.closed = True
        self.stream.close()


class _Process:
    pid = 45678
    def __init__(self, rc=0):
        self.rc = rc
        self.stopped = False
    def poll(self):
        return 1 if self.stopped else self.rc
    def wait(self):
        return self.rc if not self.stopped else 1


class TrainingQueueTests(unittest.TestCase):
    def setUp(self):
        self.scratch = queue.ROOT / f".sw161q-{os.getpid()}-{uuid.uuid4().hex}"
        self._patches = []

    def _ensure_scratch(self):
        self.scratch.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        for patcher in reversed(self._patches):
            patcher.stop()
        if self.scratch.exists():
            shutil.rmtree(self.scratch)

    def patch(self, target, new=mock.DEFAULT, **kwargs):
        p = mock.patch(target, new=new, **kwargs)
        self._patches.append(p)
        return p.start()

    def test_missing_orphan_and_rejected_preflight_never_admit(self):
        self._ensure_scratch()
        missing = self.scratch / "missing.json"
        with self.assertRaisesRegex(RuntimeError, "state is absent"):
            queue._validated_upstream(missing)
        state_path = self.scratch / "state.json"
        record = {"experiment": queue.cache_queue.EXPERIMENT,
                  "status": "preflight_running", "supervisor_pid": 98765,
                  "stages": {}}
        state_path.write_text(json.dumps(record), encoding="utf-8")
        self.patch("collaborative_test.SW_0161_supervised_spike_partition.training_queue.owner._proc_record",
                   return_value=None)
        with self.assertRaisesRegex(RuntimeError, "owner is absent"):
            queue._validated_upstream(state_path)
        record.update(status="complete_rejected", training_admission=False)
        state_path.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "scientifically rejected"):
            queue._validated_upstream(state_path)

    def test_live_registered_upstream_is_the_only_waitable_case(self):
        self._ensure_scratch()
        state_path = self.scratch / "state.json"
        state_path.write_text(json.dumps({"experiment": queue.cache_queue.EXPERIMENT,
            "status": "preflight_running", "supervisor_pid": 98765, "stages": {}}),
            encoding="utf-8")
        self.patch("collaborative_test.SW_0161_supervised_spike_partition.training_queue.owner._proc_record",
            return_value={"state": "S", "argv": [queue.PREFLIGHT_MODULE]})
        with self.assertRaises(queue.UpstreamPending):
            queue._validated_upstream(state_path)
        self.patch("collaborative_test.SW_0161_supervised_spike_partition.training_queue.owner._proc_record",
            return_value={"state": "S", "argv": ["foreign.module"]})
        with self.assertRaisesRegex(RuntimeError, "unexpected identity"):
            queue._validated_upstream(state_path)

    def test_committed_boundary_pins_physical_gpu_and_rejects_bad_runtime(self):
        self._ensure_scratch()
        work = self.scratch / "work"
        work.mkdir(parents=True)
        self.assertIsNone(queue.preferred_resume_gpu(work))
        (work / "boundary.ptx").write_bytes(b"committed")
        reader = self.patch(
            "collaborative_test.SW_0148_checkpointable_boundary_q.checkpoint.inspect_committed_metadata",
            return_value={"completed_updates": 27, "runtime": {"device_type": "cuda",
                "device_index": 0, "cuda_visible_devices": "2"}})
        self.assertEqual(queue.preferred_resume_gpu(work), 2)
        reader.return_value = {"completed_updates": 27, "runtime": {"device_type": "cpu",
            "device_index": 0, "cuda_visible_devices": "2"}}
        with self.assertRaisesRegex(ValueError, "CUDA runtime"):
            queue.preferred_resume_gpu(work)
        reader.return_value = {"completed_updates": True, "runtime": {"device_type": "cuda",
            "device_index": 0, "cuda_visible_devices": "9"}}
        with self.assertRaisesRegex(ValueError, "cursor or CUDA runtime"):
            queue.preferred_resume_gpu(work)

    def _setup_lifecycle(self, foreign=False):
        self._ensure_scratch()
        runtime = self.scratch / "runtime"
        train_root, work_root = self.scratch / "train", self.scratch / "workroot"
        self.patch("collaborative_test.SW_0161_supervised_spike_partition.train.TRAIN_ROOT", train_root)
        self.patch("collaborative_test.SW_0161_supervised_spike_partition.train.WORK_ROOT", work_root)
        admitted = {"preflight_report_sha256": "a"*64, "cache_manifest_sha256":"b"*64,
                "cache_bindings_sha256":"c"*64, "preflight_implementation_fingerprint":{"x":"d"*64},
                "queue_fingerprint":{"q":"e"*64}}
        self.patch("collaborative_test.SW_0161_supervised_spike_partition.training_queue._validated_upstream",
            side_effect=[queue.UpstreamPending("registered cache queue is still running"), admitted, admitted])
        self.patch("collaborative_test.SW_0161_supervised_spike_partition.training_queue.POLL_SECONDS", 0)
        def validate_artifact(_arm, *, directory=None):
            output = Path(directory)
            output.mkdir(parents=True, exist_ok=True)
            (output / "manifest.json").write_text('{"status":"training_complete"}', encoding="utf-8")
            return {"status":"training_complete", "completed_updates":512}
        self.patch("collaborative_test.SW_0161_supervised_spike_partition.training_queue.train.validate_training_artifact",
            side_effect=validate_artifact)
        self.patch("collaborative_test.SW_0161_supervised_spike_partition.training_queue.owner._stable_idle",
            return_value=True)
        self.patch("collaborative_test.SW_0161_supervised_spike_partition.training_queue._final_gpu_admission",
            return_value=(True,{"reason":"idle"}))
        lease_path = self.scratch / "lease.bin"
        acquire_lease = _Lease(lease_path)
        self.patch("collaborative_test.SW_0161_supervised_spike_partition.training_queue.owner._acquire",
            return_value=(3,"GPU-3",100,acquire_lease))
        events = []
        def fake_flock(_fd, operation):
            if operation == 8:  # fcntl.LOCK_UN on Linux
                events.append("unlock")
        self.patch("collaborative_test.SW_0161_supervised_spike_partition.training_queue.fcntl",
            SimpleNamespace(LOCK_EX=2, LOCK_NB=4, LOCK_UN=8, flock=fake_flock))
        proc = _Process(rc=0 if not foreign else None)
        self.patch("collaborative_test.SW_0161_supervised_spike_partition.training_queue.subprocess.Popen",
            return_value=proc)
        if foreign:
            def stop_owned(p):
                events.append("stop")
                p.stopped = True
            self.patch("collaborative_test.SW_0161_supervised_spike_partition.training_queue.owner._stop_owned",
                side_effect=stop_owned)
            self.patch("collaborative_test.SW_0161_supervised_spike_partition.training_queue._foreign_owners",
                return_value=[99999])
            self.patch("collaborative_test.SW_0161_supervised_spike_partition.training_queue.SAMPLE_SECONDS",0.001)
        return runtime, proc, acquire_lease, events

    def test_run_waits_admitted_then_launches_validates_and_records_complete(self):
        runtime, proc, lease, _events = self._setup_lifecycle()
        statuses = []
        original_atomic_json = queue._atomic_json
        def capture_state(path, payload):
            statuses.append(payload.get("status"))
            return original_atomic_json(path, payload)
        self.patch("collaborative_test.SW_0161_supervised_spike_partition.training_queue._atomic_json",
                   new=capture_state)
        q = queue.TrainingQueue(queue.train.ARMS[0], runtime=runtime)
        try:
            self.assertEqual(q.run(), 0)
            state = json.loads(q.paths["state"].read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["completed_updates"], 512)
            self.assertEqual(state["physical_gpu"], 3)
            self.assertIn("waiting_for_admitted_preflight", statuses)
            self.assertIn("running", statuses)
            self.assertEqual(statuses[-1], "complete")
            self.assertTrue(lease.closed)
            self.assertEqual(proc.poll(), 0)
        finally:
            if not q.lock_stream.closed:
                q._unlock()

    def test_foreign_owner_stops_owned_child_before_lease_unlock(self):
        runtime, proc, lease, events = self._setup_lifecycle(foreign=True)
        q = queue.TrainingQueue(queue.train.ARMS[1], runtime=runtime)
        try:
            self.assertEqual(q.run(), 1)
            state = json.loads(q.paths["state"].read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "failed_foreign_gpu_owner")
            self.assertTrue(lease.closed)
            self.assertLess(events.index("stop"), events.index("unlock"))
        finally:
            if not q.lock_stream.closed:
                q._unlock()

    def test_gpu_child_import_keeps_assigned_visibility(self):
        env = dict(os.environ, PYTHONPATH=str(queue.ROOT), CUDA_VISIBLE_DEVICES="3",
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        code = ("import numpy,os; import collaborative_test.SW_0161_supervised_spike_partition.training_queue; "
                "assert os.environ.get('CUDA_VISIBLE_DEVICES')=='3'")
        result = subprocess.run([sys.executable, "-c", code], cwd=queue.ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout)


if __name__ == "__main__":
    unittest.main()
