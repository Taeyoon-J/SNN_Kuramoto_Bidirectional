from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from collaborative_test.SW_0170_causal_synaptic_readout import admission_queue as queue


class AdmissionQueueTests(unittest.TestCase):
    def test_worker_command_names_real_module_and_physical_gpu(self):
        command = queue.worker_command(2)
        self.assertEqual(command[1:3], ["-m", queue.__name__])
        self.assertEqual(command[command.index("--physical-gpu") + 1], "2")
        self.assertEqual(command[command.index("--worker")], "--worker")

    def test_runtime_root_resolves_registered_repo_dependencies(self):
        self.assertTrue((queue.ROOT / "collaborative_test/SW_0170_causal_synaptic_readout/preflight.py").is_file())
        self.assertTrue((queue.ROOT / "snn_kuramoto_bidirectional/spike_classifier.py").is_file())

    def test_stable_idle_uses_registered_four_argument_callback(self):
        calls = []

        class Owner:
            def _stable_idle(self, gpu, uuid, stop, on_sample):
                calls.append((gpu, uuid, stop, on_sample))
                return bool(on_sample(5.0, 12))

        marker = object()
        callback = lambda elapsed, used: elapsed >= 5 and used == 12
        self.assertTrue(queue._stable_idle(Owner(), 1, "uuid-1", marker, callback))
        self.assertEqual(calls, [(1, "uuid-1", marker, callback)])

    def test_terminal_scientific_admitted_and_rejected_reports(self):
        source = {"selected_ids": list(range(8)), "ground_truth_read": False}
        fp = {"preflight.py": "a" * 64}
        base = {"experiment": "SW0170_causal_synaptic_readout",
                "source": {"source_proof": source}, "validation_data_read": False,
                "queue_fingerprint_sha256": "b" * 64,
                "implementation_fingerprint": fp}
        admitted = {**base, "status": "complete_admitted", "training_admission": True,
                    "stage1_gate_failures": []}
        rejected = {**base, "status": "scientific_rejection", "training_admission": False,
                    "stage1_gate_failures": ["filtered_maximin_loss_not_lower"]}
        self.assertEqual(queue.validate_report(admitted, source_proof=source,
                         queue_fingerprint="b" * 64, preflight_fingerprint=fp),
                         "complete_admitted")
        self.assertEqual(queue.validate_report(rejected, source_proof=source,
                         queue_fingerprint="b" * 64, preflight_fingerprint=fp),
                         "scientific_rejection")
        bad = {**rejected, "training_admission": True}
        with self.assertRaises(ValueError):
            queue.validate_report(bad, source_proof=source,
                                  queue_fingerprint="b" * 64,
                                  preflight_fingerprint=fp)

    def test_visibility_is_restored_after_owner_import(self):
        name = queue.OWNER
        with mock.patch.dict(sys.modules, {name: types.ModuleType(name)}):
            old = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = "physical-2"
            try:
                queue._owner_preserving_visibility()
                self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "physical-2")
            finally:
                if old is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = old

    def test_preflight_import_cannot_change_assigned_visibility(self):
        fake = types.ModuleType(queue.PREFLIGHT)

        def changing_import(_name):
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            return fake

        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2"}), \
                mock.patch.object(queue.importlib, "import_module", side_effect=changing_import):
            self.assertIs(queue._preflight(), fake)
            self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "2")

    def test_worker_enters_preflight_without_early_cuda_initialization(self):
        events = []

        class Cuda:
            def is_initialized(self):
                return False

            def set_device(self, _device):
                events.append("unexpected_early_set_device")

        class Torch:
            cuda = Cuda()

        class Prep:
            def run_preflight(self, **kwargs):
                events.append(("preflight_entry", kwargs["device"]))
                return {"status": "complete_admitted"}

        queue._invoke_preflight(Prep(), {}, Path("unused.json"), Torch())
        self.assertEqual(events, [("preflight_entry", "cuda:0")])

    def test_mutable_queue_state_replaces_atomically_and_preserves_fields(self):
        import json
        saved = {}

        class StatePath:
            def exists(self):
                return bool(saved)

            def read_text(self, encoding=None):
                return json.dumps(saved)

        def capture(_path, value):
            saved.clear()
            saved.update(value)

        with mock.patch.object(queue, "STATE", StatePath()), \
                mock.patch.object(queue, "_atomic_json", side_effect=capture):
            queue._write_state("validating_source_cpu", marker="kept")
            queue._write_state("running_preflight", child_pid=123)
        self.assertEqual(saved["status"], "running_preflight")
        self.assertEqual(saved["marker"], "kept")
        self.assertEqual(saved["child_pid"], 123)

    def test_worker_identity_requires_exact_registered_arguments(self):
        command = queue.worker_command(1)

        class Child:
            pid = 4321

        class Owner:
            row = {"state": "R", "argv": list(command)}

            @classmethod
            def _proc_record(cls, _pid):
                return cls.row

        self.assertTrue(queue._proc_identity(Owner, Child(), command))
        altered_gpu = list(command)
        altered_gpu[altered_gpu.index("--physical-gpu") + 1] = "2"
        Owner.row = {"state": "R", "argv": altered_gpu}
        self.assertFalse(queue._proc_identity(Owner, Child(), command))
        altered_output = list(command)
        altered_output[altered_output.index("--output") + 1] = "other.json"
        Owner.row = {"state": "R", "argv": altered_output}
        self.assertFalse(queue._proc_identity(Owner, Child(), command))


if __name__ == "__main__":
    unittest.main()
