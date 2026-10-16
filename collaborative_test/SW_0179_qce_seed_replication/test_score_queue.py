"""CPU contracts for the GT-last SW0179 score handoff."""
import json
import uuid
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from collaborative_test.SW_0179_qce_seed_replication import score_queue as sq
from collaborative_test.SW_0179_qce_seed_replication import prediction_queue as predq
from collaborative_test.SW_0178_temporal_class_competition import prediction_queue as seed1q
from collaborative_test.SW_0179_qce_seed_replication import train


class SW0179ScoreQueueContracts(unittest.TestCase):
    def test_active_prediction_wait_requires_exact_supervisor_seed(self):
        root = sq.HERE / "results_archive"
        path = root / f"active_prediction_{uuid.uuid4().hex}.json"
        path.write_text(json.dumps({"experiment": "SW0179_prediction_queue", "arm": train.ARMS[0],
            "seed": 0, "status": "running_prediction", "supervisor_pid": 54321}), encoding="utf-8")
        record = {"state": "S", "argv": ["python", "-m", predq.MODULE, "--arm",
            train.ARMS[0], "--source-seed", "0"]}
        wrong = {"state": "S", "argv": ["python", "-m", predq.MODULE, "--arm",
            train.ARMS[0], "--source-seed", "2"]}
        try:
            with mock.patch.object(predq, "_state_path", side_effect=lambda seed: path if seed == 0 else root / "missing.json"), \
                 mock.patch.object(sq, "_load", return_value=predq), \
                 mock.patch.object(seed1q, "_state_path", return_value=root / "missing_seed1.json"):
                self.assertIs(sq._prediction_dependencies(SimpleNamespace(_proc_record=lambda pid: record)), False)
                with self.assertRaisesRegex(RuntimeError, "failed/orphaned"):
                    sq._prediction_dependencies(SimpleNamespace(_proc_record=lambda pid: wrong))
        finally:
            if not path.parent.samefile(root): raise RuntimeError("fixture escaped results_archive")
            path.unlink()

    def test_three_prediction_protocols_are_required_before_scoring(self):
        root = sq.HERE / "results_archive"
        token = uuid.uuid4().hex
        state_paths = {seed: root / f"prediction_state_{token}_{seed}.json" for seed in (0, 2)}
        seed1_state = root / f"prediction_state_{token}_1.json"
        outputs = {seed: root / f"prediction_output_{token}_{seed}" for seed in (0, 1, 2)}
        try:
            state_inputs = []
            for seed in (0, 2, 1):
                output = outputs[seed]
                output.mkdir()
                archive = output / "predictions.npz"
                archive.write_bytes(b"registered fixed320 prediction fixture")
                protocol_path = output / "protocol.json"
                protocol = {"experiment": ("SW0179_fixed320_prediction" if seed != 1 else
                            "SW0178_temporal_class_competition_fixed320_prediction"),
                    "arm": train.ARMS[0], "status": "complete", "ground_truth_used_for_prediction": False,
                    "prediction_sha256": sq._sha(archive)}
                if seed != 1: protocol["seed"] = seed
                protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
                state = {"experiment": ("SW0179_prediction_queue" if seed != 1 else
                            "SW0178_prediction_queue_v1"), "arm": train.ARMS[0],
                    "status": "complete", "ground_truth_used_for_prediction": False,
                    "prediction_sha256": sq._sha(archive), "protocol_sha256": sq._sha(protocol_path),
                    "output": str(output)}
                if seed != 1: state["seed"] = seed
                state_path = state_paths[seed] if seed != 1 else seed1_state
                state_path.write_text(json.dumps(state), encoding="utf-8")
                state_inputs.append((output, archive, protocol_path, state_path))
            with mock.patch.object(predq, "_state_path", side_effect=lambda seed: state_paths[seed]), \
                 mock.patch.object(seed1q, "_state_path", return_value=seed1_state), \
                 mock.patch.object(sq, "_load", side_effect=lambda name: predq if name == predq.MODULE else seed1q):
                deps = sq._prediction_dependencies(SimpleNamespace())
            self.assertEqual(set(deps), {0, 1, 2})
            for seed in (0, 1, 2):
                self.assertEqual(deps[seed]["prediction_sha256"], sq._sha(outputs[seed] / "predictions.npz"))
                self.assertFalse(deps[seed]["state"]["ground_truth_used_for_prediction"])
        finally:
            for state_path in (*state_paths.values(), seed1_state):
                if state_path.exists():
                    if not state_path.parent.samefile(root): raise RuntimeError("state fixture escaped results_archive")
                    state_path.unlink()
            for output in outputs.values():
                if output.exists():
                    if not output.parent.samefile(root): raise RuntimeError("prediction fixture escaped results_archive")
                    for child in output.iterdir(): child.unlink()
                    output.rmdir()


if __name__ == "__main__":
    unittest.main()
