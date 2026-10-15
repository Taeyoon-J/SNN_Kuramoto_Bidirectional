"""Dependency and command contracts for the SW0168 prediction/score owners."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from collaborative_test.SW_0168_end_to_end_connectivity import (
    prediction_queue as prediction, score_queue as scoring, train,
)


class OwnerHandoffTests(unittest.TestCase):
    def test_mutable_state_advances_atomically(self):
        root = Path(__file__).resolve().parents[1] / "temporary" / f"sw168-state-cycle-{os.getpid()}"
        root.mkdir(parents=True, exist_ok=False)
        old_runtime = prediction.RUNTIME
        prediction.RUNTIME = root
        state_path = root / f"{train.ARMS[0]}_state.json"
        try:
            arm = train.ARMS[0]
            prediction._write_state(arm, "waiting_for_training")
            prediction._write_state(arm, "running_prediction", child_pid=123)
            state = json.loads(prediction._state_path(arm).read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "running_prediction")
            self.assertEqual(state["child_pid"], 123)
            self.assertEqual(state["experiment"], "SW0168_prediction_queue")
        finally:
            prediction.RUNTIME = old_runtime
            state_path.unlink(missing_ok=True)
            root.rmdir()

    def test_gpu_worker_import_preserves_assigned_visibility_without_cuda_init(self):
        code = (
            "import json, os, sys; sys.argv.append('--worker'); "
            "import collaborative_test.SW_0168_end_to_end_connectivity.prediction_queue; "
            "import torch; print(json.dumps({'visible':os.environ.get('CUDA_VISIBLE_DEVICES'),"
            "'initialized':torch.cuda.is_initialized()}))"
        )
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="3", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2],
                                env=env, capture_output=True, text=True, check=True, timeout=60)
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(payload, {"visible": "3", "initialized": False})

    def test_prediction_waits_until_both_training_arms_are_complete(self):
        with mock.patch.object(prediction, "_training_ready", side_effect=[{"manifest": 1}, False]) as ready:
            self.assertIs(prediction._all_training_ready(object()), False)
            self.assertEqual(ready.call_count, len(train.ARMS))
        values = [{"arm": arm} for arm in train.ARMS]
        with mock.patch.object(prediction, "_training_ready", side_effect=values):
            result = prediction._all_training_ready(object())
        self.assertEqual(result, dict(zip(train.ARMS, values)))

    def test_prediction_worker_command_and_exact_module_arm_liveness(self):
        cmd = prediction._worker_command(train.ARMS[0], 2)
        self.assertEqual(cmd[1:3], ["-m", "collaborative_test.SW_0168_end_to_end_connectivity.prediction_queue"])
        self.assertEqual(cmd[-5:], ["--worker", "--arm", train.ARMS[0], "--physical-gpu", "2"])
        owner = SimpleNamespace(_proc_record=lambda pid: {"state": "S", "argv": cmd})
        self.assertTrue(prediction._module_live(owner, 77,
            "collaborative_test.SW_0168_end_to_end_connectivity.prediction_queue", train.ARMS[0]))
        self.assertFalse(prediction._module_live(owner, 77,
            "collaborative_test.SW_0168_end_to_end_connectivity.prediction_queue", train.ARMS[1]))
        self.assertFalse(prediction._module_live(owner, 0,
            "collaborative_test.SW_0168_end_to_end_connectivity.prediction_queue", train.ARMS[0]))

    def test_prediction_wait_requires_real_live_training_supervisor(self):
        root = Path(__file__).resolve().parents[1] / "temporary" / f"sw168-train-dep-{os.getpid()}"
        runtime = root / "results_archive" / "training_queue_20261015"
        runtime.mkdir(parents=True, exist_ok=False)
        old_train_root = train.TRAIN_ROOT
        train.TRAIN_ROOT = root / "results_archive" / "training_20261015"
        try:
            state = runtime / f"{train.ARMS[0]}_state.json"
            state.write_text(json.dumps({"experiment": "SW0168_training_queue", "arm": train.ARMS[0],
                "status": "running_training", "supervisor_pid": 501}), encoding="utf-8")
            argv = ["python", "-m", "collaborative_test.SW_0168_end_to_end_connectivity.training_queue",
                    "--worker", "--arm", train.ARMS[0]]
            owner = SimpleNamespace(_proc_record=lambda pid: {"state": "S", "argv": argv})
            self.assertFalse(prediction._training_ready(train.ARMS[0], owner))
            owner._proc_record = lambda pid: {"state": "Z", "argv": argv}
            with self.assertRaises(RuntimeError): prediction._training_ready(train.ARMS[0], owner)
        finally:
            train.TRAIN_ROOT = old_train_root
            state.unlink(missing_ok=True)
            runtime.rmdir(); (root / "results_archive").rmdir(); root.rmdir()

    def test_score_wait_rejects_missing_or_orphaned_prediction_state(self):
        root = Path(__file__).resolve().parents[1] / "temporary" / f"sw168-score-dep-{os.getpid()}"
        root.mkdir(exist_ok=False)
        old_runtime = scoring.PREDICTION_STATE_ROOT
        scoring.PREDICTION_STATE_ROOT = root
        try:
            owner = SimpleNamespace(_proc_record=lambda pid: None)
            with self.assertRaises(FileNotFoundError): scoring._prediction_complete(train.ARMS[0], owner)
            path = root / f"{train.ARMS[0]}_state.json"
            path.write_text(json.dumps({"experiment": "SW0168_prediction_queue", "arm": train.ARMS[0],
                "status": "running_prediction", "supervisor_pid": 502}), encoding="utf-8")
            with self.assertRaises(RuntimeError): scoring._prediction_complete(train.ARMS[0], owner)
        finally:
            scoring.PREDICTION_STATE_ROOT = old_runtime
            for path in root.iterdir(): path.unlink()
            root.rmdir()


if __name__ == "__main__":
    unittest.main()
