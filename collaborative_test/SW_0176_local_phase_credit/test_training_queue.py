"""CPU lifecycle contracts for the exclusive SW0176 training queue."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
import uuid

import numpy as np  # Keep NumPy/OpenMP before torch imports.

from collaborative_test.SW_0176_local_phase_credit import training_queue as queue


ROOT = Path(__file__).resolve().parents[2]
TEMP_ROOT = ROOT / "collaborative_testing" / "temporary"


class TrainingQueueTests(unittest.TestCase):
    def test_worker_command_uses_exact_registered_module(self):
        command = queue.worker_command(2)
        self.assertEqual(command[command.index("-m") + 1], queue.MODULE)
        self.assertEqual(command[command.index("--physical-gpu") + 1], "2")
        self.assertIn("--worker", command)
        with self.assertRaises(ValueError):
            queue.worker_command(4)

    def test_worker_admits_and_builds_fresh_cpu_factory_before_training(self):
        calls = []
        admitted = {"source_proof": {"source_cache_state_sha256": "a" * 64}}
        class FakeCuda:
            @staticmethod
            def is_initialized():
                calls.append("cuda_initialized_check")
                return False
        class FakeTrainer:
            ARM = "local_phase"
            UPDATES = 64
            torch = SimpleNamespace(cuda=FakeCuda())
            @staticmethod
            def prepare_training_inputs_cpu():
                calls.append("prepare_cpu")
                return admitted
            @staticmethod
            def prepare_fresh_training_state_cpu(value):
                calls.append("fresh_cpu_factory")
                assert value is admitted
                return {"status": "complete_fresh_cpu_factory"}
            @staticmethod
            def run_training(*, physical_gpu, admitted):
                calls.append(("run_training", physical_gpu, admitted is admitted_ref))
                return {"status": "scientific_rejection", "completed_updates": 2,
                        "rejection_reasons": ["test terminal"], "arm": "local_phase"}
        admitted_ref = admitted
        fake_owner = object()
        with mock.patch.object(queue, "_runtime_modules", return_value=(FakeTrainer, fake_owner)), \
             mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2"}):
            result = queue._worker(2)
        self.assertEqual(result["status"], "scientific_rejection")
        self.assertEqual(calls[:2], ["prepare_cpu", "fresh_cpu_factory"])
        self.assertIn(("run_training", 2, True), calls)
        self.assertLess(calls.index("fresh_cpu_factory"), calls.index(("run_training", 2, True)))

    def test_mocked_supervisor_uses_real_four_argument_lease_and_publishes(self):
        from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4

        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        runtime = (TEMP_ROOT / f"sw176_train_queue_{os.getpid()}_{uuid.uuid4().hex}").resolve()
        self.assertEqual(runtime.parent, TEMP_ROOT.resolve())
        runtime.mkdir()
        output_dir = (TEMP_ROOT / f"sw176_artifact_{os.getpid()}_{uuid.uuid4().hex}_seed1").resolve()
        self.assertEqual(output_dir.parent, TEMP_ROOT.resolve())
        output_dir.mkdir()
        proof = {"source_cache_state_sha256": "a" * 64}
        admitted = {"source_proof": proof, "preflight_fingerprint": {"p": "b" * 64},
            "preflight_report_sha256": "c" * 64, "preflight_state_sha256": "d" * 64,
            "control_receipt": {"manifest_sha256": "e" * 64},
            "preflight_report": {"status": "complete_source_preflight"}}
        class FakeTrainer:
            EXPERIMENT = "SW0176_local_phase_fullwindow_training"
            OUTPUT_ROOT = TEMP_ROOT
            ARM = output_dir.name[:-6]
            @staticmethod
            def prepare_training_inputs_cpu(): return admitted
            @staticmethod
            def implementation_fingerprint(): return {"train.py": "f" * 64}
            @staticmethod
            def validate_training_artifact(path):
                return {"status": "training_complete", "completed_updates": 64}

        lease = open(os.devnull, "rb")
        owner = SimpleNamespace(MAX_MEMORY_MIB=512,
            _acquire=mock.Mock(return_value=(2, "uuid-2", 0, lease)),
            _stable_idle=mock.Mock(return_value=True),
            _stop_owned=mock.Mock(),
            gpu_guard=SimpleNamespace(_gpu_info=mock.Mock(return_value=("uuid-2", 0, []))),
            _proc_record=mock.Mock())
        class Child:
            pid = 9876
            returncode = 0
            def poll(self): return 0
            def wait(self): return 0
        def launch(command, **kwargs):
            (output_dir / "manifest.json").write_text("{}", encoding="utf-8")
            self.assertEqual(command[command.index("-m") + 1], queue.MODULE)
            return Child()
        fields = {"source_proof": proof,
            "preflight_report_sha256": admitted["preflight_report_sha256"],
            "trainer_fingerprint": FakeTrainer.implementation_fingerprint()}
        queue_fp = "1" * 64
        fake_fcntl = SimpleNamespace(LOCK_EX=1, LOCK_NB=2, LOCK_UN=8, flock=lambda *_: None)
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            with mock.patch.object(queue, "RUNTIME", runtime), \
                 mock.patch.object(queue, "_runtime_modules", return_value=(FakeTrainer, owner)), \
                 mock.patch.object(queue, "_fingerprint", return_value=(fields, queue_fp)), \
                 mock.patch.object(queue, "fcntl", fake_fcntl), \
                 mock.patch.object(queue.subprocess, "Popen", side_effect=launch):
                result = queue.supervise(timeout_seconds=10)
            state = json.loads((runtime / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "complete")
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["supervisor_pid"], os.getpid())
            owner._stable_idle.assert_called_once()
            self.assertEqual(len(owner._stable_idle.call_args.args), 4)
        finally:
            lease.close(); queue.STOP.clear()
            if previous is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous
            resolved_output = output_dir.resolve()
            if resolved_output.parent != TEMP_ROOT.resolve():
                raise AssertionError("refusing to clean artifact test files outside central temp")
            for child in resolved_output.iterdir():
                if child.is_file(): child.unlink()
            resolved_output.rmdir()
        resolved = runtime.resolve()
        if resolved.parent != TEMP_ROOT.resolve():
            raise AssertionError("refusing to clean queue test files outside central temp")
        for child in resolved.iterdir():
            if child.is_file(): child.unlink()
        resolved.rmdir()


if __name__ == "__main__":
    unittest.main()
