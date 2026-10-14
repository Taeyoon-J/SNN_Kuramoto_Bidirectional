"""CPU-only owner-queue contract checks; no CUDA devices are queried or used."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from types import SimpleNamespace
import unittest
import shutil
import uuid
from pathlib import Path
from unittest import mock

import numpy as np

from collaborative_test.SW_0163_relation_context_assignment import context_queue as queue


class ContextQueueTests(unittest.TestCase):
    def test_registered_children_and_fingerprint_resolve_real_inputs(self):
        plan = queue.dry_run_plan()
        self.assertEqual(plan["stages"], ["context_cache", "preflight"])
        self.assertIn(".context_cache", " ".join(plan["commands"]["context_cache"]))
        self.assertIn(".preflight", " ".join(plan["commands"]["preflight"]))
        self.assertEqual(plan["max_memory_mib"], 512)
        for path, digest in plan["queue_fingerprint"].items():
            self.assertTrue((queue.ROOT / path).is_file(), path)
            self.assertEqual(len(digest), 64)

    def test_library_import_preserves_assigned_cuda_visibility(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="3")
        code = ("import os, importlib; os.environ['CUDA_VISIBLE_DEVICES']='3'; "
                "import collaborative_test.SW_0163_relation_context_assignment.context_queue as q; "
                "assert os.environ['CUDA_VISIBLE_DEVICES']=='3'")
        result = subprocess.run([sys.executable, "-c", code], cwd=queue.ROOT,
                                env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_final_gpu_admission_rejects_changed_uuid_owners_and_query_errors(self):
        with mock.patch.object(queue.owner.gpu_guard, "_gpu_info", return_value=("uuid", 2, [])):
            self.assertTrue(queue._final_gpu_admission(1, "uuid")[0])
        with mock.patch.object(queue.owner.gpu_guard, "_gpu_info", return_value=("other", 2, [])):
            self.assertFalse(queue._final_gpu_admission(1, "uuid")[0])
        with mock.patch.object(queue.owner.gpu_guard, "_gpu_info", return_value=("uuid", 2, [123])):
            self.assertFalse(queue._final_gpu_admission(1, "uuid")[0])
        with mock.patch.object(queue.owner.gpu_guard, "_gpu_info", side_effect=OSError("NVML transient")):
            ok, state = queue._final_gpu_admission(1, "uuid")
            self.assertFalse(ok)
            self.assertEqual(state["reason"], "gpu_query_error")

    def test_failed_stable_check_releases_lease_and_retries(self):
        # Exercise the queue's reacquisition branch using controlled CPU fakes.
        class Lease:
            def __init__(self): self.released = False
            def fileno(self): return 88
            def close(self): self.released = True
        q = queue.ContextQueue.__new__(queue.ContextQueue)
        q.stop_event = threading.Event()
        q.child = None; q.lease = None; q.state_lock = threading.RLock()
        q.queue_fp = queue.implementation_fingerprint()
        q.context_fp = queue.context_cache._fingerprint()
        q.preflight_fp = queue.preflight.implementation_fingerprint()
        q._stage_update = mock.Mock()
        leases = [Lease(), Lease()]
        acquire_calls = []
        def acquire(_event):
            idx = len(acquire_calls); acquire_calls.append(idx)
            return 1, "uuid", 1, leases[idx]
        idle = iter((False, True))
        popen = mock.Mock(return_value=mock.Mock(pid=222, poll=mock.Mock(return_value=0), wait=mock.Mock(return_value=0)))
        with mock.patch.object(queue.owner, "_acquire", side_effect=acquire), \
             mock.patch.object(queue.owner, "_stable_idle", side_effect=lambda *a, **k: next(idle)), \
             mock.patch.object(q, "_upstream", return_value={"fixture": True}), \
             mock.patch.object(q, "_existing", return_value=True), \
             mock.patch.object(queue, "fcntl", SimpleNamespace(flock=mock.Mock(), LOCK_UN=8)), \
             mock.patch.object(queue, "_final_gpu_admission") as final_check, \
             mock.patch.object(queue.subprocess, "Popen", popen):
            # Existing validated artifact is adopted on the second stable lease.
            self.assertTrue(q._launch_stage("context_cache"))
        self.assertEqual(len(acquire_calls), 2)
        self.assertTrue(leases[0].released)
        self.assertTrue(leases[1].released)
        final_check.assert_not_called()
        popen.assert_not_called()

    def test_child_launch_uses_registered_argv_and_validates_created_artifact(self):
        class Lease:
            def __init__(self): self.released = False
            def fileno(self): return 89
            def close(self): self.released = True
        class Child:
            pid = 321
            def poll(self): return 0
            def wait(self): return 0
        module_dir = Path(__file__).resolve().parent
        root = module_dir / f"_c163q_{uuid.uuid4().hex[:8]}"
        root.mkdir()
        self.addCleanup(lambda: shutil.rmtree(root) if root.exists() else None)
        log = root / "child.log"
        lease = Lease()
        q = queue.ContextQueue.__new__(queue.ContextQueue)
        q.stop_event = threading.Event(); q.child = None; q.lease = None
        q.state_lock = threading.RLock()
        q.queue_fp = queue.implementation_fingerprint()
        q.context_fp = queue.context_cache._fingerprint()
        q.preflight_fp = queue.preflight.implementation_fingerprint()
        q._stage_update = mock.Mock()
        prepared = {"context_bindings": {"binding": "x"}}
        validated = {"status": "complete", "count": 512, "manifest_sha256": "a" * 64}
        with mock.patch.object(queue.owner, "_acquire", return_value=(1, "GPU-1", 2, lease)), \
             mock.patch.object(queue.owner, "_stable_idle", return_value=True), \
             mock.patch.object(q, "_upstream", return_value={"source": "fixed"}), \
             mock.patch.object(q, "_existing", return_value=False), \
             mock.patch.object(queue, "_final_gpu_admission", return_value=(True, {"reason": "idle"})), \
             mock.patch.object(queue, "LOGS", {"context_cache": log, "preflight": root / "pref.log"}), \
             mock.patch.object(queue, "_atomic_json"), \
             mock.patch.object(queue, "fcntl", SimpleNamespace(flock=mock.Mock(), LOCK_UN=8)), \
             mock.patch.object(queue.context_cache, "prepare_context_inputs_cpu", return_value=prepared), \
             mock.patch.object(queue.context_cache, "validate_final_cache", return_value=validated) as validate, \
             mock.patch.object(queue.subprocess, "Popen", return_value=Child()) as popen:
            self.assertTrue(q._launch_stage("context_cache"))
        args, kwargs = popen.call_args
        self.assertEqual(args[0], queue.stage_command("context_cache"))
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(kwargs["pass_fds"], (89,))
        self.assertTrue(lease.released)
        validate.assert_called_once_with(queue.context_cache.CACHE_ROOT,
                                         expected_bindings=prepared["context_bindings"])
        self.assertTrue(any(call.args[1] == "complete" for call in q._stage_update.call_args_list))


if __name__ == "__main__":
    unittest.main()
