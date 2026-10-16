"""CPU-only contracts for SW0179 per-seed prediction queue handoff."""
from __future__ import annotations

import json
import uuid
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from collaborative_test.SW_0179_qce_seed_replication import prediction_queue as pq
from collaborative_test.SW_0179_qce_seed_replication import predict_evaluate as evaluate
from collaborative_test.SW_0179_qce_seed_replication import train


@contextmanager
def _json_file(root: Path, payload):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"queue_fixture_{uuid.uuid4().hex}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        yield path
    finally:
        if path.parent.resolve() != root.resolve():
            raise RuntimeError("queue fixture cleanup escaped its designated results_archive")
        path.unlink(missing_ok=True)


class SW0179PredictionQueueContracts(unittest.TestCase):
    def test_all_literal_training_active_states_require_exact_seeded_supervisor(self):
        producer_active = {
            "validating_train_inputs", "waiting_for_exclusive_gpu",
            "waiting_for_stable_gpu", "running_worker", "yielding_foreign_owner",
        }
        self.assertTrue(producer_active.issubset(pq.ACTIVE_TRAIN))
        owner = SimpleNamespace(_proc_record=lambda pid: {
            "state": "S", "argv": ["python", "-u", "-m", pq.TRAIN_MODULE,
                "--arm", train.ARMS[0], "--source-seed", "2"]})
        for status in sorted(producer_active):
            state = {"status": status, "seed": 2, "supervisor_pid": 43210}
            self.assertTrue(pq._training_pending(owner, state, 2), status)
        wrong_arm = SimpleNamespace(_proc_record=lambda pid: {
            "state": "S", "argv": ["python", "-m", pq.TRAIN_MODULE,
                "--arm", "wrong", "--source-seed", "2"]})
        self.assertFalse(pq._training_pending(wrong_arm, {
            "status": "waiting_for_stable_gpu", "seed": 2, "supervisor_pid": 43210}, 2))
        worker = SimpleNamespace(_proc_record=lambda pid: {
            "state": "S", "argv": ["python", "-m", pq.TRAIN_MODULE, "--worker",
                "--arm", train.ARMS[0], "--source-seed", "2"]})
        self.assertFalse(pq._training_pending(worker, {
            "status": "running_worker", "seed": 2, "supervisor_pid": 43210}, 2))

    def test_worker_command_binds_seed_arm_and_assigned_physical_gpu(self):
        command = pq.worker_command(0, 3)
        self.assertEqual(command[command.index("--source-seed") + 1], "0")
        self.assertEqual(command[command.index("--arm") + 1], train.ARMS[0])
        self.assertEqual(command[command.index("--physical-gpu") + 1], "3")
        self.assertIn(pq.MODULE, command)
        with self.assertRaises(ValueError):
            pq.worker_command(1, 0)

    def test_training_rejection_publishes_terminal_state_without_duplicate_seed(self):
        root = pq.HERE / "results_archive"
        state_path = root / f"prediction_queue_fixture_{uuid.uuid4().hex}.json"
        with _json_file(root, {"bindings": {"seed": 2}, "updates": 2}) as rejection:
            terminal = {"rejection_path": rejection, "rejection_sha256": pq._sha(rejection)}
            with mock.patch.object(pq, "_state_path", lambda seed: state_path):
                result = pq._publish_training_rejection(2, terminal)
                state = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "complete_training_rejected")
        self.assertEqual(state["status"], "complete_training_rejected")
        self.assertEqual(state["seed"], 2)
        self.assertFalse(state["ground_truth_used_for_prediction"])
        self.assertEqual(state["training_rejection_sha256"], terminal["rejection_sha256"])
        if not state_path.parent.samefile(root):
            raise RuntimeError("queue fixture cleanup escaped results_archive")
        state_path.unlink()


if __name__ == "__main__":
    unittest.main()
