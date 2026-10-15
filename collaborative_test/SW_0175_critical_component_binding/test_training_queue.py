from __future__ import annotations

import json
import os
from pathlib import Path
import unittest
import uuid
from types import SimpleNamespace
from unittest import mock

import numpy
import torch

from collaborative_test.SW_0175_critical_component_binding import training_queue as queue


class _FakeFcntl:
    LOCK_EX = 1
    LOCK_NB = 2
    LOCK_UN = 4

    @staticmethod
    def flock(_fd, _op):
        return None


class TrainingQueueTests(unittest.TestCase):
    def test_worker_command_names_real_module_arm_and_physical_gpu(self):
        trainer = SimpleNamespace(ARMS=("native_Q", "critical_component"))
        with mock.patch.object(queue, "_runtime_modules", return_value=(trainer, object())):
            command = queue.worker_command("critical_component", 3)
        self.assertEqual(command[1:3], ["-m", queue.TRAINER_MODULE.rsplit(".", 1)[0] + ".training_queue"])
        self.assertEqual(command[3:], ["--worker", "--arm", "critical_component", "--physical-gpu", "3"])
        with self.assertRaises(ValueError):
            queue.worker_command("unknown", 3)

    def _exercise_run(self, terminal):
        fixture_root = Path(__file__).resolve().parents[2] / "collaborative_testing" / "temporary"
        fixture_root.mkdir(parents=True, exist_ok=True)
        root = fixture_root / f"sw171_queue_{terminal}_{os.getpid()}_{uuid.uuid4().hex}"
        root.mkdir()
        runtime = root / "runtime"
        output_root = root / "training_output"
        proof = {"source_cache_state_sha256": "a" * 64, "ordered_train_ids": list(range(512))}
        admission = {"proof": proof, "queue_fingerprint_sha256": "b" * 64,
            "preflight_fingerprint": {"preflight.py": "c" * 64},
            "admission_report_sha256": "d" * 64, 
            "admission_report": {"training_admission": True}}
        bindings = {"admission_report_sha256": admission["admission_report_sha256"],
            "admission_queue_fingerprint_sha256": admission["queue_fingerprint_sha256"],
            "preflight_implementation_fingerprint": admission["preflight_fingerprint"],
            "source_proof": proof, "source_state_sha256": proof["source_cache_state_sha256"],
            "training_fingerprint": {"trainer": "e" * 64}, }
        manifest = {"status": "training_complete", "completed_updates": 64, "bindings": bindings}
        trainer = SimpleNamespace(ARMS=("native_Q", "critical_component"), OUTPUT_ROOT=output_root,
            _load_admitted_inputs=lambda: admission,
            _training_fingerprint=lambda: bindings["training_fingerprint"],
            validate_training_artifact=lambda arm, _path: manifest)
        owner = SimpleNamespace(MAX_MEMORY_MIB=512)
        owner.gpu_guard = SimpleNamespace(_gpu_info=lambda _gpu: ("GPU-TEST", 2048, [child.pid]))
        lease_path = root / "lease"
        with lease_path.open("xb") as lease_stream:
            lease_stream.write(b"")
        lease = lease_path.open("rb")
        owner._acquire = lambda _stop: (2, "GPU-TEST", 0, lease)
        stable_args = []

        def stable_idle(gpu, gpu_uuid, stop_event, on_sample):
            stable_args.append((gpu, gpu_uuid, stop_event, callable(on_sample)))
            on_sample(.1, 0)
            return True

        owner._stable_idle = stable_idle
        owner._proc_record = lambda pid: {"state": "S", "argv": child.command}
        owner._stop_owned = lambda process: setattr(process, "returncode", -9)

        class Child:
            pid = 45678
            command = None
            returncode = None

            def __init__(self, command, **kwargs):
                self.command = list(command)
                Child.command = self.command
                stream = kwargs["stdout"]
                stream.write(json.dumps({"status": "training_complete", "arm": "native_Q",
                                         "completed_updates": 64}) + "\n")
                (output_root / "native_Q_seed1").mkdir(parents=True)
                (output_root / "native_Q_seed1" / "manifest.json").write_text("{}", encoding="utf-8")

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                self.returncode = 0
                return 0

        child = Child
        # GPU owner is empty for the final idle admission sample; once the
        # worker is alive, only its PID may appear and its memory can exceed
        # the 512 MiB idle-admission ceiling.
        gpu_calls = iter([("GPU-TEST", 10, []), ("GPU-TEST", 2500, [Child.pid])])
        owner.gpu_guard._gpu_info = lambda _gpu: next(gpu_calls)
        old_runtime, old_stop = queue.RUNTIME, queue._stop
        queue.RUNTIME = runtime
        queue._stop = __import__("threading").Event()
        try:
            with (mock.patch.object(queue, "fcntl", _FakeFcntl),
                  mock.patch.object(queue, "_runtime_modules", return_value=(trainer, owner)),
                  mock.patch.object(queue, "_admit", return_value=(trainer, admission)),
                  mock.patch.object(queue, "_load_preserving_visibility", return_value=owner),
                  mock.patch.object(queue.subprocess, "Popen", side_effect=Child),
                  mock.patch.object(queue.time, "sleep", side_effect=lambda _seconds: setattr(Child, "returncode", 0))):
                result = queue.run("native_Q", timeout_seconds=30)
            state = json.loads((runtime / "native_Q_state.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "complete")
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["physical_gpu"], 2)
            self.assertEqual(state["queue_fingerprint_sha256"], result["queue_fingerprint_sha256"])
            self.assertEqual(len(stable_args), 1)
            self.assertTrue(stable_args[0][3])
            self.assertEqual(Child.command[Child.command.index("--physical-gpu") + 1], "2")
        finally:
            queue.RUNTIME, queue._stop = old_runtime, old_stop
            lease.close()
            if os.path.commonpath([str(fixture_root.resolve()), str(root.resolve())]) != str(fixture_root.resolve()):
                raise RuntimeError("refusing to remove fixture outside central temporary root")
            import shutil
            shutil.rmtree(root)

    def test_run_revalidates_admission_under_lease_and_accepts_owned_memory_over_idle_limit(self):
        original = queue._validate_training_result
        try:
            queue._validate_training_result = lambda trainer, arm, admitted: (
                {"status": "training_complete", "completed_updates": 64,
                 "bindings": {"admission_report_sha256": admitted["admission_report_sha256"],
                    "admission_queue_fingerprint_sha256": admitted["queue_fingerprint_sha256"],
                    "preflight_implementation_fingerprint": admitted["preflight_fingerprint"],
                    "source_proof": admitted["proof"],
                    "source_state_sha256": admitted["proof"]["source_cache_state_sha256"],
                    "training_fingerprint": trainer._training_fingerprint(), }},
                "f" * 64)
            self._exercise_run("complete")
        finally:
            queue._validate_training_result = original

    def test_rejection_consumer_requires_committed_boundary_and_current_provenance(self):
        from collaborative_test.SW_0175_critical_component_binding import train as train
        from collaborative_test.SW_0148_checkpointable_boundary_q import checkpoint
        fixture_root = Path(__file__).resolve().parents[2] / "collaborative_testing" / "temporary"
        fixture_root.mkdir(parents=True, exist_ok=True)
        root = fixture_root / f"sw171_queue_rejection_{os.getpid()}_{__import__('uuid').uuid4().hex}"
        root.mkdir()
        arm = "native_Q"
        admitted = {"proof": {"source_cache_state_sha256": "a" * 64},
                    "admission_report_sha256": "b" * 64}
        bindings = {"source_proof": admitted["proof"],
                    "admission_report_sha256": admitted["admission_report_sha256"],
                    "training_fingerprint": {"trainer": "c" * 64}}
        history = [{"update": 1, "loss": 1.0}]
        work = root / "work" / f"{arm}_seed1"
        work.mkdir(parents=True)
        wrapped, encoder = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
        optimizer = torch.optim.Adam(list(wrapped.parameters()) + list(encoder.parameters()), lr=1e-3)
        checkpoint.save_boundary(work, completed_updates=1, wrapped=wrapped, encoder=encoder,
            optimizer=optimizer, history=history, bindings=bindings,
            runtime={"cpu": True}, device="cpu")
        boundary = work / "boundary.ptx"
        rejection = {"experiment": train.EXPERIMENT, "status": "scientific_rejection",
            "arm": arm, "completed_updates": 1, "history_sha256": train._history_sha(history),
            "bindings_sha256": train._binding_sha(bindings), "boundary_sha256": train._sha_file(boundary),
            "source_proof": bindings["source_proof"],
            "training_fingerprint": bindings["training_fingerprint"],
            "rejection_reasons": ["dead Q-to-S credit"], "validation_data_read": False}
        (work / "credit_rejection.json").write_text(json.dumps(rejection), encoding="utf-8")
        trainer = SimpleNamespace(OUTPUT_ROOT=root,
            _training_fingerprint=lambda: bindings["training_fingerprint"],
            _read_credit_rejection=train._read_credit_rejection)
        runtime = checkpoint
        old_load = queue._load_preserving_visibility
        try:
            queue._load_preserving_visibility = lambda _name: runtime
            report, report_sha = queue._validate_rejection(trainer, arm, admitted)
            self.assertEqual(report["status"], "scientific_rejection")
            self.assertEqual(report["source_proof"], admitted["proof"])
            self.assertEqual(len(report_sha), 64)
            bad = dict(admitted, proof={"source_cache_state_sha256": "d" * 64})
            with self.assertRaises(ValueError):
                queue._validate_rejection(trainer, arm, bad)
        finally:
            queue._load_preserving_visibility = old_load
            if os.path.commonpath([str(fixture_root.resolve()), str(root.resolve())]) != str(fixture_root.resolve()):
                raise RuntimeError("refusing to remove fixture outside central temporary root")
            import shutil
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
