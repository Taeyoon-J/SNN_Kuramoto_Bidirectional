import numpy as np  # Keep NumPy/OpenMP initialization before torch on Windows.
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from collaborative_test.SW_0179_qce_seed_replication import training_queue as queue
from collaborative_test.SW_0179_qce_seed_replication import train


class SW0179TrainingQueueContracts(unittest.TestCase):
    def test_worker_command_binds_objective_source_seed_and_physical_gpu(self):
        expected = [queue.sys.executable, "-m",
            "collaborative_test.SW_0179_qce_seed_replication.training_queue",
            "--worker", "--arm", "product_q_ce", "--source-seed", "2",
            "--physical-gpu", "3"]
        self.assertEqual(queue.worker_command("product_q_ce", 2, 3), expected)
        with self.assertRaises(ValueError):
            queue.worker_command("not-registered", 2, 3)
        with self.assertRaises(ValueError):
            queue.worker_command("product_q_ce", 1, 3)

    def test_process_identity_requires_full_registered_worker_argv(self):
        command = queue.worker_command("product_q_ce", 0, 1)
        owner = SimpleNamespace(_proc_record=lambda _pid: {"state": "S", "argv": command})
        self.assertTrue(queue._proc_matches(owner, 123, command))
        self.assertFalse(queue._proc_matches(owner, 123,
            command[:-3] + ["--source-seed", "2", "--physical-gpu", "1"]))

    def test_cpu_admission_binds_shared_data_and_seed_specific_fresh_state(self):
        inputs = {"dataset_manifest_sha256": "a" * 64,
            "dataset_arrays_sha256": "b" * 64,
            "dataset_producer_proof": {"source_seed": 1, "scope": "shared data only"},
            "ordered_train_ids": list(range(512))}
        proof = {"seed": 2, "source_state_sha256": "c" * 64,
                 "initial_model_state_sha256": "d" * 64}
        trainer = SimpleNamespace(prepare_training_inputs_cpu=lambda: inputs,
            prepare_fresh_training_state_cpu=lambda seed, prepared: {
                "proof": proof, "found": None, "optimizer": None,
                "families": {}, "bindings": {}, "inputs": prepared})
        got_trainer, got_inputs, identity = queue._input_admission(2, trainer)
        self.assertIs(got_trainer, trainer)
        self.assertIs(got_inputs, inputs)
        self.assertEqual(identity["source_factory_proof"], proof)
        self.assertEqual(identity["dataset_producer_proof"]["source_seed"], 1)
        self.assertNotIn("source_state_sha256", identity["dataset_producer_proof"])

    def test_queue_fingerprint_binds_seed_objective_trainer_and_queue_bytes(self):
        identity = {"source_factory_proof": {"seed": 0,
            "initial_model_state_sha256": "a" * 64}, "ordered_train_ids": list(range(512))}
        payload, digest = queue._queue_fingerprint(train, "product_q_ce", 0, identity)
        self.assertEqual(payload["experiment"], "SW0179_training_queue")
        self.assertEqual(payload["seed"], 0)
        self.assertEqual(payload["source_input_identity"], identity)
        self.assertEqual(len(payload["queue_sha256"]), 64)
        self.assertEqual(len(payload["trainer_sha256"]), 64)
        self.assertEqual(len(payload["trainer_fingerprint_sha256"]), 64)
        self.assertEqual(len(digest), 64)

    def test_resume_gpu_uses_seed_specific_committed_boundary(self):
        trainer = SimpleNamespace(OUTPUT_ROOT=Path("unused"), CHECKPOINT_MODULE=train.CHECKPOINT_MODULE)
        with mock.patch.object(Path, "is_file", return_value=False):
            self.assertIsNone(queue._resume_gpu(trainer, "product_q_ce", 2))
        with mock.patch.object(Path, "is_file", return_value=True):
            observed = []
            def inspect(path):
                observed.append(str(path))
                return {"runtime": {"cuda_visible_devices": "2"}}
            with mock.patch.object(queue, "_import_preserving_visibility",
                return_value=SimpleNamespace(inspect_committed_metadata=inspect)):
                self.assertEqual(queue._resume_gpu(trainer, "product_q_ce", 2), 2)
                self.assertIn("product_q_ce_seed2", observed[0])

    def test_worker_passes_seed_and_assigned_visibility_to_trainer(self):
        calls = []
        trainer = SimpleNamespace(train=lambda arm, *, seed, physical_gpu: calls.append(
            (arm, seed, physical_gpu)) or {"status": "training_complete", "completed_updates": 64},
            ARMS=("product_q_ce",), UPDATES=64)
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}, clear=False):
            with mock.patch.object(queue, "_runtime_modules", return_value=(trainer, None)):
                result = queue._worker("product_q_ce", 0, 1)
        self.assertEqual(calls, [("product_q_ce", 0, 1)])
        self.assertEqual(result["status"], "training_complete")


if __name__ == "__main__":
    unittest.main()
