import numpy as np  # Keep NumPy/OpenMP initialization before torch on Windows.
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from collaborative_test.SW_0178_temporal_class_competition import training_queue as queue
from collaborative_test.SW_0178_temporal_class_competition.train import ARMS, CHECKPOINT_MODULE


class SW178TrainingQueueContracts(unittest.TestCase):
    def test_worker_command_and_process_identity_bind_arm_and_gpu(self):
        command = ["python", "-m", "collaborative_test.SW_0178_temporal_class_competition.training_queue",
                   "--worker", "--arm", ARMS[0], "--physical-gpu", "2"]
        with mock.patch.object(queue.sys, "executable", "python"):
            self.assertEqual(queue.worker_command(ARMS[0], 2), command)
        owner = SimpleNamespace(_proc_record=lambda _pid: {"state": "S", "argv": command})
        self.assertTrue(queue._proc_matches(owner, 42, command))
        self.assertFalse(queue._proc_matches(owner, 42, command[:-1] + ["1"]))

    def test_resume_uses_boundary_physical_visibility(self):
        trainer = SimpleNamespace(OUTPUT_ROOT=Path("ignored"), CHECKPOINT_MODULE=CHECKPOINT_MODULE)
        with mock.patch.object(Path, "is_file", return_value=False):
            self.assertIsNone(queue._resume_gpu(trainer, ARMS[0]))
        with mock.patch.object(Path, "is_file", return_value=True):
            with mock.patch.object(queue, "_import_preserving_visibility",
                return_value=SimpleNamespace(inspect_committed_metadata=lambda path: {
                    "runtime": {"cuda_visible_devices": "2"}})):
                self.assertEqual(queue._resume_gpu(trainer, ARMS[0]), 2)

    def test_foreign_yield_rechecks_preferred_gpu_after_new_boundary(self):
        trainer = SimpleNamespace()
        with mock.patch.object(queue, "_resume_gpu", side_effect=[None, 2]) as resume:
            preferred = [queue._resume_gpu(trainer, ARMS[0]), queue._resume_gpu(trainer, ARMS[0])]
        self.assertEqual(preferred, [None, 2])
        self.assertEqual(resume.call_count, 2)

    def test_queue_fingerprint_binds_queue_and_trainer_files(self):
        trainer = queue._import_preserving_visibility(queue.TRAINER_MODULE)
        identity = {"source_state_sha256": "a" * 64, "ordered_train_ids": list(range(512))}
        payload, digest = queue._queue_fingerprint(trainer, ARMS[1], identity)
        self.assertEqual(payload["experiment"], "SW0178_training_queue")
        self.assertEqual(payload["source_input_identity"], identity)
        self.assertEqual(len(payload["queue_sha256"]), 64)
        self.assertEqual(len(payload["trainer_sha256"]), 64)
        self.assertEqual(len(digest), 64)


if __name__ == "__main__":
    unittest.main()
