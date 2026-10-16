"""CPU-only contract checks for SW0176's owner-guarded GPU worker."""
from __future__ import annotations

import inspect
import json
import os
from types import SimpleNamespace
from pathlib import Path
import uuid
import unittest
from unittest import mock

import numpy as np  # Initialize NumPy/OpenMP before torch imports.

from collaborative_test.SW_0176_local_phase_credit import resource_queue_v4 as queue


class ResourceQueueTests(unittest.TestCase):
    def test_terminal_report_classifies_complete_and_scientific_rejection(self):
        proof = {"source_cache_state_sha256": "a" * 64}
        fingerprint = {"preflight_v4.py": "b" * 64}
        common = {"source_proof": proof, "implementation_fingerprint": fingerprint,
                  "optimizer_updates": 0, "validation_data_read": False,
                  "cuda_visible_devices": "2"}
        complete = {**common, "status": "complete_source_preflight"}
        rejected = {**common, "status": "scientific_rejection",
                    "rejection_reasons": ["full-window family credit was not positive"]}
        self.assertEqual(queue._validate_worker_report(complete, fingerprint=fingerprint,
            source_proof=proof, physical_gpu=2), "complete")
        self.assertEqual(queue._validate_worker_report(rejected, fingerprint=fingerprint,
            source_proof=proof, physical_gpu=2), "complete_rejected")
        with self.assertRaisesRegex(ValueError, "reason"):
            queue._validate_worker_report({**rejected, "rejection_reasons": []},
                fingerprint=fingerprint, source_proof=proof, physical_gpu=2)

    def test_worker_command_names_exact_module_and_physical_device(self):
        command = queue.worker_command(2, "report.json")
        self.assertEqual(command[command.index("-m") + 1], queue.MODULE)
        self.assertIn("--worker", command)
        self.assertEqual(command[command.index("--physical-gpu") + 1], "2")
        self.assertEqual(command[command.index("--output") + 1], "report.json")

    def test_stable_idle_uses_actual_four_argument_lease_api(self):
        from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4

        parameters = tuple(inspect.signature(stage_queue_v4._stable_idle).parameters)
        self.assertEqual(parameters, ("gpu", "expected_uuid", "stop_event", "on_sample"))
        callback = mock.create_autospec(stage_queue_v4._stable_idle, return_value=True)
        owner = SimpleNamespace(_stable_idle=callback)
        stop = object()
        self.assertTrue(queue._wait_for_stable_idle(owner, 3, "uuid-3", stop))
        callback.assert_called_once()
        self.assertEqual(callback.call_args.args[:3], (3, "uuid-3", stop))
        self.assertTrue(callable(callback.call_args.args[3]))

    def test_worker_runs_cpu_admission_before_cuda_query_and_preserves_lease(self):
        class FakeCuda:
            def __init__(self):
                self.initialized_checks = 0
                self.device_count_checks = 0

            def is_initialized(self):
                self.initialized_checks += 1
                return False

            def device_count(self):
                self.device_count_checks += 1
                return 1

        fake_torch = SimpleNamespace(cuda=FakeCuda())
        prepared = {"cpu": True}
        fake = SimpleNamespace(torch=fake_torch,
            prepare_inputs_cpu=mock.Mock(return_value=prepared),
            run_preflight=mock.Mock(return_value={"status": "complete_source_preflight"}))
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}), \
             mock.patch.object(queue, "_load_preflight", return_value=fake):
            result = queue.run_worker(1, "review.json")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "1")
        fake.prepare_inputs_cpu.assert_called_once_with()
        fake.run_preflight.assert_called_once_with(device="cuda:0", prepared=prepared,
                                                   output="review.json")
        self.assertGreaterEqual(fake_torch.cuda.initialized_checks, 1)
        self.assertEqual(fake_torch.cuda.device_count_checks, 1)

    def test_worker_rejects_foreign_or_unassigned_visibility_before_import(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), \
             mock.patch.object(queue, "_load_preflight") as load:
            with self.assertRaisesRegex(RuntimeError, "exclusive physical GPU"):
                queue.run_worker(1, "unused")
        load.assert_not_called()

    def test_fingerprint_pins_preflight_queue_and_contract_test(self):
        fingerprint = queue.implementation_fingerprint()
        self.assertIn("collaborative_test/SW_0176_local_phase_credit/resource_queue_v4.py", fingerprint)
        self.assertIn("collaborative_test/SW_0176_local_phase_credit/test_resource_queue_v4.py", fingerprint)
        self.assertIn("collaborative_test/SW_0176_local_phase_credit/preflight_v4.py", fingerprint)

    def test_supervisor_records_owned_lifecycle_and_publishes_bound_worker_report(self):
        from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4

        central = queue.ROOT / "collaborative_testing" / "temporary"
        central.mkdir(parents=True, exist_ok=True)
        runtime = (central / f"sw176_queue_{os.getpid()}_{uuid.uuid4().hex}").resolve()
        self.assertEqual(runtime.parent, central.resolve())
        runtime.mkdir()
        try:
            proof = {"seed": 1, "source_cache_state_sha256": "a" * 64}
            fingerprint = {"dependency.py": "b" * 64}
            fake_prep = SimpleNamespace(prepare_inputs_cpu=lambda: {"source_proof": proof,
                "implementation_fingerprint": fingerprint},
                implementation_fingerprint=lambda: fingerprint)
            fake_owner = SimpleNamespace(MAX_MEMORY_MIB=512,
                _acquire=mock.Mock(), _stable_idle=mock.Mock(return_value=True),
                _stop_owned=mock.Mock(),
                gpu_guard=SimpleNamespace(_gpu_info=mock.Mock(return_value=("uuid-2", 0, []))))
            lease = open(os.devnull, "rb")
            fake_owner._acquire.return_value = (2, "uuid-2", None, lease)
            names = {
                "collaborative_test.SW_0176_local_phase_credit.preflight_v4": fake_prep,
                "collaborative_test.SW_0148_checkpointable_boundary_q.stage_queue_v4": fake_owner,
            }
            class Child:
                pid = 9876
                returncode = 0
                def poll(self):
                    return 0

            def launch(command, **kwargs):
                path = Path(command[command.index("--output") + 1])
                path.write_text(json.dumps({"status": "complete_source_preflight",
                    "source_proof": proof, "implementation_fingerprint": fingerprint,
                    "optimizer_updates": 0, "validation_data_read": False,
                    "cuda_visible_devices": "2"}), encoding="utf-8")
                return Child()

            fake_fcntl = SimpleNamespace(LOCK_EX=1, LOCK_NB=2, LOCK_UN=8,
                flock=lambda *_: None)
            prior_runtime = queue.RUNTIME
            prior_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            try:
                with mock.patch.object(queue, "RUNTIME", runtime), \
                     mock.patch.dict("sys.modules", {"fcntl": fake_fcntl}), \
                     mock.patch.object(queue.importlib, "import_module",
                                       side_effect=lambda name: names[name]), \
                     mock.patch.object(queue.subprocess, "Popen", side_effect=launch):
                    queue.supervise()
                state = json.loads((runtime / "state.json").read_text(encoding="utf-8"))
                self.assertEqual(state["status"], "complete")
                self.assertEqual(state["supervisor_pid"], os.getpid())
                self.assertEqual(state["source_proof"], proof)
                self.assertEqual(state["implementation_fingerprint"], fingerprint)
                fake_owner._stable_idle.assert_called_once()
                self.assertEqual(len(fake_owner._stable_idle.call_args.args), 4)
                self.assertEqual(state["report_file_sha256"],
                    __import__("hashlib").sha256((runtime / "preflight.json").read_bytes()).hexdigest())
            finally:
                lease.close()
                queue.STOP.clear()
                if prior_visible is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = prior_visible
                self.assertEqual(queue.RUNTIME, prior_runtime)
        finally:
            resolved = runtime.resolve()
            if resolved.parent != central.resolve():
                raise AssertionError("refusing to clean test files outside central temporary root")
            for item in resolved.iterdir():
                if item.is_file():
                    item.unlink()
            resolved.rmdir()


if __name__ == "__main__":
    unittest.main()
