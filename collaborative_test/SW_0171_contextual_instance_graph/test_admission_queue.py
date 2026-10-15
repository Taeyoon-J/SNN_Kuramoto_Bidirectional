from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy  # Keep NumPy before Torch imports in Windows CPU test processes.

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from collaborative_test.SW_0171_contextual_instance_graph import admission_queue as queue


class AdmissionQueueTests(unittest.TestCase):
    def test_worker_command_survives_python_m_main_module_name(self):
        with mock.patch.dict(queue.worker_command.__globals__, {"__name__": "__main__"}):
            command = queue.worker_command(1)
        self.assertEqual(command[1:3], ["-m", "collaborative_test.SW_0171_contextual_instance_graph.admission_queue"])

    def test_worker_command_and_repo_root_are_exact(self):
        command = queue.worker_command(3)
        self.assertEqual(command[1:3], ["-m", queue.__name__])
        self.assertEqual(command[command.index("--physical-gpu") + 1], "3")
        self.assertEqual(command[command.index("--worker")], "--worker")
        self.assertTrue((queue.ROOT / "collaborative_test/SW_0171_contextual_instance_graph/preflight.py").is_file())
        self.assertTrue((queue.ROOT / "snn_kuramoto_bidirectional/spike_classifier.py").is_file())

    def test_stable_idle_uses_registered_four_argument_callback(self):
        calls = []

        class Owner:
            def _stable_idle(self, gpu, uuid, stop, on_sample):
                calls.append((gpu, uuid, stop, on_sample))
                on_sample(6.0, 14)
                return True

        marker = object()
        sample = lambda elapsed, used: elapsed >= 6 and used == 14
        self.assertTrue(queue._stable_idle(Owner(), 2, "GPU-uuid", marker, sample))
        self.assertEqual(calls, [(2, "GPU-uuid", marker, sample)])

    def test_report_consumer_accepts_only_registered_terminal_outcomes(self):
        proof = {"seed": 1, "ordered_train_ids": list(range(512)),
                 "validation_data_read": False, "training_ground_truth_scope": "TRAIN512 only"}
        implementation = {"preflight.py": "a" * 64}
        common = {"experiment": "SW0171_contextual_instance_graph_preflight",
                  "source_proof": proof, "queue_fingerprint_sha256": "b" * 64,
                  "implementation_fingerprint": implementation,
                  "validation_data_read": False, "optimizer_updates": 0}
        admitted = {**common, "status": "complete_admitted", "training_admission": True}
        rejected = {**common, "status": "complete_rejected", "training_admission": False,
                    "rejection_reasons": ["lambda_calibration_norms_not_finite_positive"]}
        args = dict(source_proof=proof, queue_fingerprint="b" * 64,
                    preflight_fingerprint=implementation)
        self.assertEqual(queue.validate_report(admitted, **args), "complete_admitted")
        self.assertEqual(queue.validate_report(rejected, **args), "complete_rejected")
        for broken in (
            {**rejected, "training_admission": True},
            {**rejected, "source_proof": {**proof, "seed": 2}},
            {**rejected, "optimizer_updates": 1},
            {**rejected, "status": "failed_preflight"},
        ):
            with self.assertRaises(ValueError):
                queue.validate_report(broken, **args)

    def test_cpu_preparation_hides_then_restores_leased_visibility(self):
        events = []

        class Cuda:
            @staticmethod
            def is_initialized():
                events.append(("is_initialized", os.environ.get("CUDA_VISIBLE_DEVICES")))
                return False

        class Prep:
            torch = types.SimpleNamespace(cuda=Cuda())

            @staticmethod
            def prepare_inputs_cpu():
                events.append(("prepare", os.environ.get("CUDA_VISIBLE_DEVICES")))
                return {"source_proof": {"ok": True}}

        old_worker = queue._WORKER
        queue._WORKER = True
        try:
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3"}), \
                    mock.patch.object(queue.importlib, "import_module", return_value=Prep):
                imported, prepared = queue._cpu_prepare()
                self.assertIs(imported, Prep)
                self.assertEqual(prepared["source_proof"], {"ok": True})
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "3")
        finally:
            queue._WORKER = old_worker
        self.assertIn(("prepare", ""), events)
        self.assertIn(("is_initialized", "3"), events)

    def test_worker_calls_real_preflight_api_and_preserves_assigned_device(self):
        events = []
        published = {}
        proof = {"seed": 1, "ordered_train_ids": list(range(512)),
                 "validation_data_read": False, "training_ground_truth_scope": "TRAIN512 only"}

        class Cuda:
            @staticmethod
            def is_initialized():
                return False

            @staticmethod
            def device_count():
                events.append(("device_count", os.environ.get("CUDA_VISIBLE_DEVICES")))
                return 1

        class Torch:
            cuda = Cuda()

            @staticmethod
            def set_num_threads(_n):
                pass

        class Prep:
            torch = Torch()

            @staticmethod
            def prepare_inputs_cpu():
                events.append(("cpu_prepare", os.environ.get("CUDA_VISIBLE_DEVICES")))
                return {"source_proof": proof}

            @staticmethod
            def implementation_fingerprint():
                return {"preflight.py": "a" * 64}

            @staticmethod
            def run_preflight(*, device, prepared, output):
                events.append(("run", device, os.environ.get("CUDA_VISIBLE_DEVICES"), output.name))
                self = {"experiment": "SW0171_contextual_instance_graph_preflight",
                        "status": "complete_admitted", "training_admission": True,
                        "source_proof": proof, "implementation_fingerprint": Prep.implementation_fingerprint(),
                        "validation_data_read": False, "optimizer_updates": 0}
                published[str(output)] = dict(self)
                return self

        old_worker = queue._WORKER
        queue._WORKER = True
        try:
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2"}), \
                    mock.patch.object(queue, "_preflight", return_value=Prep), \
                    mock.patch.object(queue, "RAW_REPORT", Path("raw.json")), \
                    mock.patch.object(queue, "_atomic_json", side_effect=lambda path, obj: published.__setitem__(str(path), dict(obj))), \
                    mock.patch.object(queue, "fingerprint_sha256", return_value="b" * 64):
                raw = queue.RAW_REPORT
                final = Path("final.json")
                report = queue._worker(2, final)
                self.assertEqual(report["status"], "complete_admitted")
                self.assertIn(str(raw), published)  # raw preflight is separately persisted.
                self.assertIn(str(final), published)
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2")
        finally:
            queue._WORKER = old_worker
        self.assertIn(("cpu_prepare", ""), events)
        self.assertIn(("device_count", "2"), events)
        self.assertTrue(any(event[0] == "run" and event[1:3] == ("cuda:0", "2") for event in events))

    def test_worker_identity_requires_exact_registered_argv(self):
        command = queue.worker_command(1)

        class Child:
            pid = 4321

        class Owner:
            row = {"state": "R", "argv": list(command)}

            @classmethod
            def _proc_record(cls, _pid):
                return cls.row

        self.assertTrue(queue._proc_identity(Owner, Child(), command))
        for marker, replacement in (("--physical-gpu", "2"), ("--output", "other.json")):
            altered = list(command)
            altered[altered.index(marker) + 1] = replacement
            Owner.row = {"state": "R", "argv": altered}
            self.assertFalse(queue._proc_identity(Owner, Child(), command))


if __name__ == "__main__":
    unittest.main()
