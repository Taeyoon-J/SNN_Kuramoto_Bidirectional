from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from collaborative_test.SW_0182_unique1024_actual_descent import training_queue as queue


class TrainingQueueTests(unittest.TestCase):
    def test_registered_worker_uses_canonical_module_and_gpu(self):
        command = queue.worker_command(2, 3)
        self.assertEqual(command[-5:], ["--worker", "--source-seed", "2", "--physical-gpu", "3"])
        self.assertIn("collaborative_test.SW_0182_unique1024_actual_descent.training_queue", command)
        with self.assertRaises(ValueError):
            queue.worker_command(1, 4)

    def test_cpu_admission_binds_data_and_completed_control(self):
        inputs = {"dataset_manifest_sha256": "a" * 64,
            "dataset_arrays_sha256": "b" * 64, "dataset_producer_proof": {"rows": 1024},
            "ordered_train_ids": list(range(10000, 11024))}
        fresh = {"proof": {"cuda_initialized": False, "initial_model_state_sha256": "c" * 64,
                           "optimizer_groups": [{"name": "graph"}]}}
        control = {"manifest_sha256": "d" * 64, "history_sha256": "e" * 64,
                   "manifest": {"initial_model_state_sha256": "c" * 64}}
        trainer = SimpleNamespace(prepare_training_inputs_cpu=lambda: inputs,
            prepare_fresh_training_state_cpu=lambda seed, data: fresh,
            _control_prefix=lambda seed: control)
        result, identity = queue._cpu_admission(2, trainer)
        self.assertIs(result, inputs)
        self.assertEqual(identity["control_manifest_sha256"], "d" * 64)
        self.assertEqual(identity["control_history_sha256"], "e" * 64)
        self.assertEqual(identity["source_factory_proof"], fresh["proof"])

    def test_queue_fingerprint_pins_actual_trainer_and_cache_proof(self):
        trainer = SimpleNamespace(implementation_fingerprint=lambda: {"fp": "x"},
                                  fingerprint_sha256=lambda: "a" * 64)
        identity = {"dataset_manifest_sha256": "b" * 64, "source_factory_proof": {"seed": 1}}
        payload, digest = queue._queue_fingerprint(trainer, 1, identity)
        self.assertEqual(payload["trainer_sha256"], queue._sha(queue.HERE / "train.py"))
        self.assertEqual(payload["input_identity"], identity)
        self.assertEqual(len(digest), 64)

    def test_cli_help_and_parent_dispatch(self):
        with self.assertRaises(SystemExit) as help_exit:
            queue.main(["--help"])
        self.assertEqual(help_exit.exception.code, 0)
        with mock.patch.object(queue, "run", return_value={"status": "queued"}) as run:
            self.assertEqual(queue.main(["--source-seed", "0"]), 0)
        run.assert_called_once_with(0)


if __name__ == "__main__":
    unittest.main()
