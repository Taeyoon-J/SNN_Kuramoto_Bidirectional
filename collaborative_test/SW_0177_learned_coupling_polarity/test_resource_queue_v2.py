"""CPU lifecycle regressions for retrying SW0177 preflight ownership."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
import uuid
from unittest import mock

import numpy as np

from collaborative_test.SW_0177_learned_coupling_polarity import resource_queue_v2 as queue


class ResourceQueueV2Tests(unittest.TestCase):
    def test_report_validator_preserves_frozen_preflight_contract(self):
        proof = {"source_cache_state_sha256": "a" * 64}
        fingerprint = {"preflight.py": "b" * 64}
        common = {"source_proof": proof, "implementation_fingerprint": fingerprint,
                  "optimizer_updates": 0, "validation_data_read": False,
                  "cuda_visible_devices": "1"}
        complete = {**common, "status": "complete_source_preflight"}
        rejected = {**common, "status": "scientific_rejection",
                    "rejection_reasons": ["registered gradient criterion failed"]}
        self.assertEqual(queue._validate_worker_report(
            complete, preflight_fingerprint=fingerprint,
            source_proof=proof, physical_gpu=1), "complete")
        self.assertEqual(queue._validate_worker_report(
            rejected, preflight_fingerprint=fingerprint,
            source_proof=proof, physical_gpu=1), "complete_rejected")
        with self.assertRaises(ValueError):
            queue._validate_worker_report({**complete, "optimizer_updates": 1},
                preflight_fingerprint=fingerprint,
                source_proof=proof, physical_gpu=1)

    def test_stable_idle_uses_real_four_argument_api_and_worker_module_is_v2(self):
        from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4

        callback = mock.create_autospec(stage_queue_v4._stable_idle, return_value=True)
        owner = SimpleNamespace(_stable_idle=callback)
        stop_event = object()
        self.assertTrue(queue._wait_for_stable_idle(owner, 1, "uuid-1", stop_event))
        callback.assert_called_once()
        self.assertEqual(len(callback.call_args.args), 4)
        command = queue.worker_command(1, "attempt.json")
        self.assertEqual(command[command.index("-m") + 1], queue.MODULE)
        self.assertTrue(queue.MODULE.endswith("resource_queue_v2"))

    def test_foreign_owner_yields_only_our_child_then_reacquires_and_completes(self):
        central = queue.ROOT / "collaborative_testing" / "temporary"
        central.mkdir(parents=True, exist_ok=True)
        runtime = (central / f"sw177_v2_{os.getpid()}_{uuid.uuid4().hex}").resolve()
        self.assertEqual(runtime.parent, central.resolve())
        runtime.mkdir()
        proof = {"source_cache_state_sha256": "c" * 64, "selected_ids": [4, 9]}
        prep_fp = {"preflight.py": "d" * 64}
        queue_fp = {"preflight": prep_fp, "queue": "e" * 64}
        fake_prep = SimpleNamespace(
            prepare_inputs_cpu=lambda: {"source_proof": proof,
                                         "implementation_fingerprint": prep_fp},
            implementation_fingerprint=lambda: prep_fp)
        foreign_pid = 424242
        gpu_observations = iter([
            ("uuid-1", 0, []),                    # first lease initial check
            ("uuid-1", 800, [9876, foreign_pid]), # foreign arrives while our worker lives
            ("uuid-1", 0, []),                    # second lease initial check
        ])
        released = []

        class Lease:
            def __init__(self, token):
                self.token = token
                self.closed = False
            def fileno(self):
                return 10 + self.token
            def close(self):
                self.closed = True
                released.append(self.token)

        leases = [Lease(1), Lease(2)]
        acquire_count = {"n": 0}
        clock = [0.0]
        def acquire(_stop):
            index = acquire_count["n"]
            acquire_count["n"] += 1
            # Simulate a long wait before each lease; that must not consume an
            # active worker's timeout budget.
            clock[0] += 1000.0
            return (1, "uuid-1", 0, leases[index])

        class Child:
            def __init__(self, pid, live):
                self.pid, self.returncode, self.live = pid, None, live
            def poll(self):
                if not self.live and self.returncode is None:
                    self.returncode = 0
                return None if self.live else self.returncode
            def wait(self, timeout=None):
                if self.returncode is None:
                    self.returncode = -15 if self.pid == 9876 else 0
                self.live = False
                return self.returncode

        children = [Child(9876, True), Child(9877, False)]
        launched = {"n": 0}
        def popen(command, **kwargs):
            child = children[launched["n"]]
            launched["n"] += 1
            if child.pid == 9877:
                out = Path(command[command.index("--output") + 1])
                out.write_text(json.dumps({
                    "status": "complete_source_preflight",
                    "source_proof": proof,
                    "implementation_fingerprint": prep_fp,
                    "optimizer_updates": 0,
                    "validation_data_read": False,
                    "cuda_visible_devices": "1",
                }), encoding="utf-8")
            return child

        class StopEvent:
            def is_set(self): return False
            def wait(self, _seconds): return False

        fake_owner = SimpleNamespace(
            MAX_MEMORY_MIB=512,
            _acquire=mock.Mock(side_effect=acquire),
            _stable_idle=mock.Mock(return_value=True),
            _stop_owned=mock.Mock(side_effect=lambda child: child.wait()),
            _proc_record=mock.Mock(return_value={"state": "S", "argv": []}),
            gpu_guard=SimpleNamespace(_gpu_info=mock.Mock(side_effect=lambda _gpu: next(gpu_observations))))
        fake_fcntl = SimpleNamespace(LOCK_EX=1, LOCK_NB=2, LOCK_UN=8, flock=lambda *_: None)
        names = {
            "collaborative_test.SW_0177_learned_coupling_polarity.preflight": fake_prep,
            "collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4": fake_owner,
        }
        previous_runtime = queue.RUNTIME
        previous_state, previous_report = queue.STATE, queue.REPORT
        previous_stop = queue.STOP
        previous_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            queue.RUNTIME = runtime
            queue.STATE, queue.REPORT = runtime / "state.json", runtime / "preflight.json"
            queue.STOP = StopEvent()
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            with mock.patch.dict("sys.modules", {"fcntl": fake_fcntl}), \
                 mock.patch.object(queue.importlib, "import_module", side_effect=lambda name: names[name]), \
                 mock.patch.object(queue, "implementation_fingerprint", return_value=queue_fp), \
                 mock.patch.object(queue.time, "monotonic", side_effect=lambda: clock[0]), \
                 mock.patch.object(queue.subprocess, "Popen", side_effect=popen):
                queue.supervise(timeout_seconds=60)

            state = json.loads((runtime / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["experiment"], "SW0177_learned_coupling_polarity_preflight_v2")
            self.assertEqual([x["status"] for x in state["attempts"]],
                             ["yielded_foreign_owner", "complete"])
            self.assertEqual(state["attempts"][0]["owners"], [9876, foreign_pid])
            self.assertEqual(state["attempts"][1]["physical_gpu"], 1)
            self.assertEqual(fake_owner._stop_owned.call_count, 1)
            fake_owner._stop_owned.assert_called_once_with(children[0])
            self.assertEqual(acquire_count["n"], 2)
            self.assertEqual(sorted(released), [1, 2])
            self.assertTrue((runtime / "attempts" / "worker_attempt01.log").exists())
            self.assertTrue((runtime / "attempts" / "worker_attempt02.log").exists())
            report_bytes = (runtime / "preflight.json").read_bytes()
            self.assertEqual(hashlib.sha256(report_bytes).hexdigest(), state["report_file_sha256"])
            self.assertEqual(json.loads(report_bytes)["status"], "complete_source_preflight")
        finally:
            queue.RUNTIME, queue.STATE, queue.REPORT = previous_runtime, previous_state, previous_report
            queue.STOP = previous_stop
            if previous_visible is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous_visible
            resolved = runtime.resolve()
            if resolved.parent != central.resolve():
                raise AssertionError("refusing to clean outside central temporary directory")
            for child in resolved.rglob("*"):
                if child.is_file(): child.unlink()
            for child in sorted(resolved.rglob("*"), reverse=True):
                if child.is_dir(): child.rmdir()
            resolved.rmdir()


if __name__ == "__main__":
    unittest.main()
