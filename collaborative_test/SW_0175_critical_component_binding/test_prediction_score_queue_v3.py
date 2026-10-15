"""V2 handoff tests keep the prediction chain attached to the recovered trainer namespace."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import uuid
import unittest
from pathlib import Path
from unittest import mock

from collaborative_test.SW_0175_critical_component_binding import prediction_queue_v3 as predq
from collaborative_test.SW_0175_critical_component_binding import score_queue_v3 as scoreq


def _safe_cleanup(root: Path, central_temp: Path) -> None:
    resolved_root = root.resolve()
    resolved_temp = central_temp.resolve()
    if resolved_root.parent != resolved_temp:
        raise AssertionError(f"temporary fixture escaped central temp root: {resolved_root}")
    for item in sorted(resolved_root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            item.rmdir()
    resolved_root.rmdir()


class QueueHandoffTests(unittest.TestCase):
    def test_prediction_worker_uses_registered_v2_module(self):
        command = predq._worker_command("native_Q", 2)
        self.assertEqual(command[1:3], ["-m", "collaborative_test.SW_0175_critical_component_binding.prediction_queue_v3"])
        self.assertEqual(command[-5:], ["--worker", "--arm", "native_Q", "--physical-gpu", "2"])

    def test_stable_idle_uses_real_four_argument_owner_contract(self):
        from collaborative_test.SW_0148_checkpointable_boundary_q import stage_queue_v4 as owner_api
        self.assertEqual(len(inspect.signature(owner_api._stable_idle).parameters), 4)
        received = []
        owner = type("Owner", (), {})()
        def stable(gpu, gpu_uuid, stop_event, on_sample):
            received.append((gpu, gpu_uuid, on_sample))
            on_sample(2.0, 128)
            return True
        owner._stable_idle = stable
        with mock.patch.object(predq, "_write_state") as write_state:
            self.assertTrue(predq._stable_idle(owner, "native_Q", 2, "uuid-2"))
        self.assertEqual(received[0][0:2], (2, "uuid-2"))
        write_state.assert_called_once()

    def test_owned_worker_memory_is_not_misclassified_as_foreign(self):
        self.assertFalse(predq._foreign_during_owned_child(44, "gpu-uuid", "gpu-uuid", [44]))
        self.assertTrue(predq._foreign_during_owned_child(44, "gpu-uuid", "gpu-uuid", [44, 99]))
        self.assertTrue(predq._foreign_during_owned_child(44, "gpu-uuid", "other-uuid", [44]))

    def test_training_dependency_consumes_v2_state_and_manifest(self):
        base = Path(__file__).resolve().parents[2] / "collaborative_testing" / "temporary"
        base.mkdir(parents=True, exist_ok=True)
        root = base / f"sw175-preddep-{os.getpid()}-{uuid.uuid4().hex}"
        root.mkdir()
        try:
            output = root / "results_archive" / "training_v2_20261015"
            artifact = output / "native_Q_seed1"
            artifact.mkdir(parents=True)
            manifest = artifact / "manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")
            state_root = output.parent / "training_queue_v2_20261015"
            state_root.mkdir()
            state = {"experiment": "SW0175_training_queue_v2", "arm": "native_Q",
                     "status": "complete", "manifest_sha256": predq._sha(manifest)}
            (state_root / "native_Q_state.json").write_text(json.dumps(state), encoding="utf-8")
            with (mock.patch.object(predq.train, "OUTPUT_ROOT", output),
                  mock.patch.object(predq.train, "validate_training_artifact", return_value={"status": "validated"}) as validate):
                result = predq._training_ready("native_Q", owner=object())
            self.assertEqual(result, {"status": "validated"})
            validate.assert_called_once_with("native_Q", artifact)
        finally:
            _safe_cleanup(root, base)

    def test_training_dependency_waits_only_for_live_v2_supervisor(self):
        base = Path(__file__).resolve().parents[2] / "collaborative_testing" / "temporary"
        root = base / f"sw175-predlive-{os.getpid()}-{uuid.uuid4().hex}"
        root.mkdir()
        try:
            output = root / "results_archive" / "training_v2_20261015"
            state_root = output.parent / "training_queue_v2_20261015"
            state_root.mkdir(parents=True)
            state_path = state_root / "native_Q_state.json"
            state = {"experiment": "SW0175_training_queue_v2", "arm": "native_Q",
                     "status": "running_training", "supervisor_pid": 54321}
            state_path.write_text(json.dumps(state), encoding="utf-8")
            owner = mock.Mock()
            owner._proc_record.return_value = {"state": "S", "argv": ["python", "-m",
                "collaborative_test.SW_0175_critical_component_binding.training_queue_v2",
                "--arm", "native_Q"]}
            with mock.patch.object(predq.train, "OUTPUT_ROOT", output):
                self.assertIs(predq._training_ready("native_Q", owner), False)
            owner._proc_record.return_value["argv"][2] = "collaborative_test.SW_0175_critical_component_binding.training_queue"
            with mock.patch.object(predq.train, "OUTPUT_ROOT", output):
                with self.assertRaises(RuntimeError):
                    predq._training_ready("native_Q", owner)
        finally:
            _safe_cleanup(root, base)

    def test_score_wait_consumes_v2_prediction_state_hashes(self):
        base = Path(__file__).resolve().parents[2] / "collaborative_testing" / "temporary"
        root = base / f"sw175-scoredep-{os.getpid()}-{uuid.uuid4().hex}"
        root.mkdir()
        try:
            out = root / "arm"
            out.mkdir()
            archive, protocol = out / "predictions.npz", out / "protocol.json"
            archive.write_bytes(b"predictions"); protocol.write_text("{}\n", encoding="utf-8")
            state = {"experiment": "SW0175_prediction_queue_v3", "arm": "native_Q", "status": "complete",
                "output": str(out), "prediction_sha256": scoreq._sha(archive),
                "protocol_sha256": scoreq._sha(protocol)}
            state_root = root / "states"; state_root.mkdir()
            (state_root / "native_Q_state.json").write_text(json.dumps(state), encoding="utf-8")
            with mock.patch.object(scoreq, "PREDICTION_STATE_ROOT", state_root):
                self.assertEqual(scoreq._prediction_complete("native_Q", object()), state)
        finally:
            _safe_cleanup(root, base)

    def test_score_wait_does_not_accept_v1_prediction_supervisor(self):
        base = Path(__file__).resolve().parents[2] / "collaborative_testing" / "temporary"
        root = base / f"sw175-scorelive-{os.getpid()}-{uuid.uuid4().hex}"
        root.mkdir()
        try:
            state_root = root / "states"; state_root.mkdir()
            state = {"experiment": "SW0175_prediction_queue_v3", "arm": "native_Q",
                     "status": "running_prediction", "supervisor_pid": 54321}
            (state_root / "native_Q_state.json").write_text(json.dumps(state), encoding="utf-8")
            owner = mock.Mock()
            owner._proc_record.return_value = {"state": "S", "argv": ["python", "-m",
                "collaborative_test.SW_0175_critical_component_binding.prediction_queue_v3", "--arm", "native_Q"]}
            with mock.patch.object(scoreq, "PREDICTION_STATE_ROOT", state_root):
                self.assertIs(scoreq._prediction_complete("native_Q", owner), False)
            owner._proc_record.return_value["argv"][2] = "collaborative_test.SW_0168_end_to_end_connectivity.prediction_queue"
            with mock.patch.object(scoreq, "PREDICTION_STATE_ROOT", state_root):
                with self.assertRaises(RuntimeError):
                    scoreq._prediction_complete("native_Q", owner)
        finally:
            _safe_cleanup(root, base)


if __name__ == "__main__":
    unittest.main()
