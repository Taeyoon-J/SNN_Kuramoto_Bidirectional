"""Queue contracts for SW0177 producer identity, module identity, and CPU admission."""
import json
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np  # NumPy/OpenMP before importing train/Torch.

from collaborative_test.SW_0177_learned_coupling_polarity import (
    prediction_queue, score_queue, train,
)


class FakeOwner:
    def __init__(self, argv): self.argv = argv
    def _proc_record(self, _pid): return {"state": "S", "argv": self.argv}


class PredictionQueueContracts(unittest.TestCase):
    def test_worker_command_binds_arm_physical_gpu_and_canonical_module(self):
        for arm in train.ARMS:
            command = prediction_queue.worker_command(arm, 2)
            self.assertEqual(command[command.index("-m") + 1], prediction_queue.MODULE)
            self.assertEqual(command[command.index("--arm") + 1], arm)
            self.assertEqual(command[command.index("--physical-gpu") + 1], "2")
        with self.assertRaises(ValueError):
            prediction_queue.worker_command("other", 2)
        with self.assertRaises(ValueError):
            prediction_queue.worker_command("signed", 4)

    def test_complete_training_state_consumes_actual_queue_producer_schema(self):
        manifest_path = Path(__file__)
        state = {"experiment": "SW0177_learned_coupling_polarity_training_queue",
            "arm": "signed", "status": "complete", "completed_updates": train.UPDATES,
            "result_manifest_sha256": "a" * 64, "validation_data_read": False}
        self.assertTrue(prediction_queue._validate_training_receipt("signed", state,
            manifest_path=manifest_path, manifest_sha256="a" * 64))
        for bad in ({**state, "completed_updates": 63},
                    {**state, "validation_data_read": True},
                    {**state, "result_manifest_sha256": "b" * 64}):
            with self.assertRaisesRegex(ValueError, "does not bind final artifact"):
                prediction_queue._validate_training_receipt("signed", bad,
                    manifest_path=manifest_path, manifest_sha256="a" * 64)

    def test_pending_training_statuses_and_arm_identity_are_exact(self):
        owner = FakeOwner([sys.executable, "-u", "-m", prediction_queue.TRAIN_MODULE,
                           "--arm", "signed"])
        for status in sorted(prediction_queue.ACTIVE_TRAIN):
            self.assertTrue(prediction_queue._training_is_pending(owner, {
                "status": status, "supervisor_pid": 1234}, "signed"), status)
        self.assertFalse(prediction_queue._training_is_pending(owner, {
            "status": "waiting_for_stable_gpu", "supervisor_pid": 1234}, "nonnegative"))
        self.assertFalse(prediction_queue._training_is_pending(FakeOwner(
            [sys.executable, "-m", prediction_queue.TRAIN_MODULE, "--arm", "signed",
             "--worker", "--physical-gpu", "2"]), {
                 "status": "running_worker", "supervisor_pid": 1234}, "signed"))

    def test_actual_waiting_for_stable_gpu_producer_state_is_pending(self):
        # This is an emitted producer state, not a value enumerated from the consumer set.
        producer_active = {"waiting_for_v2_source_preflight", "waiting_for_exclusive_gpu",
            "waiting_for_stable_gpu", "running_worker", "yielding_foreign_owner"}
        self.assertTrue(producer_active.issubset(prediction_queue.ACTIVE_TRAIN))
        central = Path(__file__).resolve().parents[2] / "collaborative_testing" / "temporary"
        central.mkdir(parents=True, exist_ok=True)
        state_path = central / f"sw177_waiting_{uuid.uuid4().hex}.json"
        state = {"experiment": "SW0177_learned_coupling_polarity_training_queue",
            "arm": "signed", "status": "waiting_for_stable_gpu", "supervisor_pid": 1234}
        state_path.write_text(json.dumps(state), encoding="utf-8")
        owner = FakeOwner([sys.executable, "-u", "-m", prediction_queue.TRAIN_MODULE,
                           "--arm", "signed"])
        try:
            self.assertIn("waiting_for_stable_gpu", prediction_queue.ACTIVE_TRAIN)
            self.assertTrue(prediction_queue._training_is_pending(owner, state, "signed"))
            queue = type("Queue", (), {"_state_path": staticmethod(lambda _arm: state_path)})
            with mock.patch.object(prediction_queue, "_load", return_value=queue), \
                 mock.patch.object(train, "ARMS", ("signed",)):
                self.assertIs(prediction_queue._training_states(owner), False)

            completed = {**state, "status": "complete", "completed_updates": train.UPDATES,
                         "validation_data_read": False}
            self.assertFalse(prediction_queue._training_is_pending(owner, completed, "signed"))
            self.assertFalse(prediction_queue._training_is_pending(owner, state, "nonnegative"))
            self.assertFalse(prediction_queue._training_is_pending(FakeOwner(
                [sys.executable, "-m", prediction_queue.TRAIN_MODULE, "--arm", "signed",
                 "--worker"]), state, "signed"))
        finally:
            self.assertEqual(state_path.resolve().parent, central.resolve())
            state_path.unlink(missing_ok=True)

    def test_live_prediction_supervisor_consumed_by_score_queue_exactly(self):
        owner = FakeOwner([sys.executable, "-u", "-m", prediction_queue.MODULE,
                           "--arm", "nonnegative"])
        self.assertTrue(score_queue._module_live(owner, 5678, score_queue.PREDICTION_MODULE,
                                                 "nonnegative"))
        self.assertFalse(score_queue._module_live(owner, 5678, score_queue.PREDICTION_MODULE,
                                                  "signed"))

    def test_score_queue_fingerprint_tracks_both_prediction_consumers(self):
        fingerprint = train._module(
            "collaborative_test.SW_0177_learned_coupling_polarity.predict_evaluate"
        ).implementation_fingerprint()
        expected = {
            "collaborative_test/SW_0177_learned_coupling_polarity/prediction_queue.py",
            "collaborative_test/SW_0177_learned_coupling_polarity/score_queue.py",
            "collaborative_test/SW_0177_learned_coupling_polarity/test_predict_evaluate.py",
            "collaborative_test/SW_0177_learned_coupling_polarity/test_prediction_score_queue.py",
            "collaborative_test/SW_0177_learned_coupling_polarity/test_score_queue.py",
        }
        self.assertTrue(expected.issubset(fingerprint))
        self.assertTrue(all(len(fingerprint[name]) == 64 for name in expected))


if __name__ == "__main__":
    unittest.main()
