"""Producer/consumer and owner-liveness tests for SW0169 evaluation queues."""
from __future__ import annotations

import json
import os
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from collaborative_test.SW_0169_normalized_hard_reset import (
    prediction_queue as prediction, score_queue as scoring, train,
)


def _central_temp():
    root = Path(__file__).resolve().parents[2] / "collaborative_testing" / "temporary"
    root.mkdir(parents=True, exist_ok=True)
    owned = root / f"sw169-queue-{os.getpid()}-{time.time_ns()}"
    owned.mkdir(exist_ok=False)
    return owned


class QueueContractTests(unittest.TestCase):
    def test_mutable_prediction_state_updates_replace_atomically(self):
        root = _central_temp()
        old_runtime = prediction.RUNTIME
        prediction.RUNTIME = root
        try:
            arm = train.ARMS[0]
            prediction._write_state(arm, "waiting_for_training")
            prediction._write_state(arm, "running_prediction", child_pid=123)
            state = json.loads(prediction._state_path(arm).read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "running_prediction")
            self.assertEqual(state["experiment"], "SW0169_prediction_queue")
            self.assertEqual(state["child_pid"], 123)
        finally:
            prediction.RUNTIME = old_runtime
            import shutil
            shutil.rmtree(root)

    def test_worker_command_and_exact_module_arm_liveness(self):
        module = "collaborative_test.SW_0169_normalized_hard_reset.prediction_queue"
        cmd = prediction._worker_command(train.ARMS[0], 2)
        self.assertEqual(cmd[1:3], ["-m", module])
        self.assertEqual(cmd[-5:], ["--worker", "--arm", train.ARMS[0], "--physical-gpu", "2"])
        owner = SimpleNamespace(_proc_record=lambda _pid: {"state": "S", "argv": cmd})
        self.assertTrue(prediction._module_live(owner, 77, module, train.ARMS[0]))
        self.assertFalse(prediction._module_live(owner, 77, module, train.ARMS[1]))
        self.assertFalse(prediction._module_live(owner, 0, module, train.ARMS[0]))

    def test_prediction_wait_requires_real_live_training_owner(self):
        root = _central_temp()
        state_root = root / "training_queue_20261015"
        state_root.mkdir()
        old_root = train.TRAIN_ROOT
        train.TRAIN_ROOT = root / "training_20261015"
        try:
            arm = train.ARMS[0]
            (state_root / f"{arm}_state.json").write_text(json.dumps({
                "experiment": "SW0169_training_queue", "arm": arm,
                "status": "running_training", "supervisor_pid": 501}), encoding="utf-8")
            module = "collaborative_test.SW_0169_normalized_hard_reset.training_queue"
            argv = ["python", "-m", module, "--arm", arm]
            owner = SimpleNamespace(_proc_record=lambda _pid: {"state": "S", "argv": argv})
            self.assertFalse(prediction._training_ready(arm, owner))
            owner._proc_record = lambda _pid: {"state": "Z", "argv": argv}
            with self.assertRaises(RuntimeError):
                prediction._training_ready(arm, owner)
        finally:
            train.TRAIN_ROOT = old_root
            import shutil
            shutil.rmtree(root)

    def test_prediction_requires_both_training_arms_before_preparation(self):
        with mock.patch.object(prediction, "_training_ready",
                               side_effect=[{"arm": train.ARMS[0]}, False]) as ready:
            self.assertIs(prediction._all_training_ready(object()), False)
            self.assertEqual(ready.call_count, 2)
        complete = [{"arm": arm} for arm in train.ARMS]
        with mock.patch.object(prediction, "_training_ready", side_effect=complete):
            self.assertEqual(prediction._all_training_ready(object()), dict(zip(train.ARMS, complete)))

    def test_stable_idle_uses_registered_four_argument_callback(self):
        calls = []
        def stable_idle(gpu, uuid, stop, on_sample):
            calls.append((gpu, uuid, stop.is_set()))
            on_sample(1.0, 0)
            return True
        event = prediction.threading.Event()
        self.assertTrue(stable_idle(1, "gpu-1", event, lambda *_: None))
        self.assertEqual(calls, [(1, "gpu-1", False)])

    def test_score_consumer_accepts_completed_prediction_producer_state(self):
        root = _central_temp()
        old_root = scoring.PREDICTION_STATE_ROOT
        scoring.PREDICTION_STATE_ROOT = root
        try:
            arm = train.ARMS[0]
            out = root / f"{arm}_seed1"; out.mkdir()
            (out / "predictions.npz").write_bytes(b"pred")
            (out / "protocol.json").write_bytes(b"protocol")
            with mock.patch.object(scoring, "_sha", side_effect=lambda path:
                    {out / "predictions.npz": "p", out / "protocol.json": "q"}[Path(path)]):
                (root / f"{arm}_state.json").write_text(json.dumps({
                    "experiment": "SW0169_prediction_queue", "arm": arm, "status": "complete",
                    "output": str(out), "prediction_sha256": "p", "protocol_sha256": "q"}), encoding="utf-8")
                owner = SimpleNamespace(_proc_record=lambda _pid: None)
                done = scoring._prediction_complete(arm, owner)
                self.assertEqual(done["status"], "complete")
                self.assertEqual(done["prediction_sha256"], "p")
        finally:
            scoring.PREDICTION_STATE_ROOT = old_root
            import shutil
            shutil.rmtree(root)

    def test_score_wait_rejects_orphan_and_wrong_namespace(self):
        root = _central_temp()
        old_root = scoring.PREDICTION_STATE_ROOT
        scoring.PREDICTION_STATE_ROOT = root
        try:
            arm = train.ARMS[0]
            path = root / f"{arm}_state.json"
            path.write_text(json.dumps({"experiment": "SW0169_prediction_queue", "arm": arm,
                "status": "running_prediction", "supervisor_pid": 502}), encoding="utf-8")
            owner = SimpleNamespace(_proc_record=lambda _pid: None)
            with self.assertRaises(RuntimeError): scoring._prediction_complete(arm, owner)
            path.write_text(json.dumps({"experiment": "SW0168_prediction_queue", "arm": arm,
                "status": "running_prediction", "supervisor_pid": 502}), encoding="utf-8")
            with self.assertRaises(ValueError): scoring._prediction_complete(arm, owner)
        finally:
            scoring.PREDICTION_STATE_ROOT = old_root
            import shutil
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
