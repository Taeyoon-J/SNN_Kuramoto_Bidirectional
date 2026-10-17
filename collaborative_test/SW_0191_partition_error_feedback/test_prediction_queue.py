"""CPU contracts for SW0191 prediction queue state and process identity."""
from __future__ import annotations

import unittest
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from collaborative_test.SW_0191_partition_error_feedback import prediction_queue as q


class PredictionQueueTests(unittest.TestCase):
    def test_worker_command_binds_module_seed_and_physical_device(self):
        self.assertEqual(q.worker_command(1, 2), [q.sys.executable, "-u", "-m", q.MODULE,
            "--worker", "--source-seed", "1", "--physical-gpu", "2"])
        for seed, gpu in ((3, 1), (1, 4)):
            with self.assertRaises(ValueError): q.worker_command(seed, gpu)

    def test_training_parent_requires_registered_live_module_and_seed(self):
        temp_root = q.ROOT / "collaborative_testing" / "temporary"
        temp_root.mkdir(parents=True, exist_ok=True)
        state_path = temp_root / f"sw181-training-state-{os.getpid()}.json"
        try:
            state = {"experiment": "SW0191_training_queue", "seed": 1, "arm": "product_q_ce",
                "status": "waiting_for_stable_gpu", "supervisor_pid": 234}
            state_path.write_text(json.dumps(state), encoding="utf-8")
            training = SimpleNamespace(_state_path=lambda seed: state_path)
            owner = SimpleNamespace(_proc_record=lambda pid: {"state": "S", "argv": [
                "python", "-m", q.TRAIN_MODULE, "--source-seed", "1"]})
            with mock.patch.object(q, "_load", return_value=training):
                self.assertIs(q._training_terminal(owner, 1), False)
                with self.assertRaises(ValueError): q._training_terminal(owner, 2)
            bad = SimpleNamespace(_proc_record=lambda pid: {"state": "S", "argv": [
                "python", "-m", q.TRAIN_MODULE, "--worker", "--source-seed", "1"]})
            self.assertFalse(q._module_live(bad, 234, q.TRAIN_MODULE, 1, worker=False))
        finally:
            if state_path.is_file(): state_path.unlink()
            if temp_root.is_dir() and not any(temp_root.iterdir()): temp_root.rmdir()

    def test_cli_parent_does_not_need_worker_physical_gpu_argument(self):
        with self.assertRaises(SystemExit) as help_exit:
            q.main(["--help"])
        self.assertEqual(help_exit.exception.code, 0)
        with mock.patch.object(q, "run", return_value={"status": "waiting_for_training"}) as run:
            self.assertEqual(q.main(["--source-seed", "2"]), 0)
        run.assert_called_once_with(2)


if __name__ == "__main__": unittest.main()
