"""CPU-only supervisor identity and CLI contracts for SW0183."""
import contextlib
import io
import unittest
from pathlib import Path
import uuid
from unittest import mock

from collaborative_test.SW_0183_shared_prefix_unique_scaling import training_queue as queue


class QueueContracts(unittest.TestCase):
    def test_worker_command_binds_seed_gpu_and_canonical_module(self):
        command = queue.worker_command(2, 3)
        self.assertEqual(command[command.index("-m") + 1], queue.__name__)
        self.assertEqual(command[command.index("--source-seed") + 1], "2")
        self.assertEqual(command[command.index("--physical-gpu") + 1], "3")
        with self.assertRaises(ValueError):
            queue.worker_command(2, 4)

    def test_parent_cli_dispatches_seed_once_without_worker_gpu(self):
        with mock.patch.object(queue, "run", return_value={"status": "complete", "seed": 0}) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                code = queue.main(["--source-seed", "0"])
        self.assertEqual(code, 0)
        run.assert_called_once_with(0)

    def test_worker_cli_requires_gpu_and_dispatches_exact_assignment(self):
        with mock.patch.object(queue, "_worker", return_value={"status": "training_complete"}) as worker:
            with contextlib.redirect_stdout(io.StringIO()):
                code = queue.main(["--worker", "--source-seed", "1", "--physical-gpu", "2"])
        self.assertEqual(code, 0)
        worker.assert_called_once_with(1, 2)

    def test_resume_gpu_returns_none_before_any_boundary_exists(self):
        central = queue.ROOT / "collaborative_testing" / "temporary"
        central.mkdir(parents=True, exist_ok=True)
        trainer, _owner = queue._modules()
        output_root = central / f"sw0183-queue-{uuid.uuid4().hex}"
        with mock.patch.object(trainer, "OUTPUT_ROOT", output_root):
            self.assertIsNone(queue._resume_gpu(trainer, 0))

    def test_cli_help_is_loadable_without_worker_arguments(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                queue.main(["--help"])
        self.assertEqual(caught.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
